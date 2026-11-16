"""Treinamento e avaliacao reproduziveis para a Issue #4.

Este modulo nao conhece arquivos NIfTI, fases, cortes ou augmentations. Ele recebe
``DataLoader`` prontos e exige que cada lote identifique os pacientes. Assim, a
preparacao do dataset pode evoluir de forma independente sem duplicar regras da
Issue #2.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ConfiguracaoTreinamento:
    """Hiperparametros que precisam acompanhar cada checkpoint."""

    arquitetura: str
    canais_entrada: int
    configuracao_temporal: str
    estrategia_corte: str
    epocas: int = 30
    tamanho_lote: int = 8
    taxa_aprendizado: float = 1e-3
    decaimento_pesos: float = 1e-4
    pos_weight: float = 3.0
    paciencia: int = 5
    limiar: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.canais_entrada not in (1, 3):
            raise ValueError("canais_entrada deve ser 1 ou 3")
        if self.epocas < 1 or self.tamanho_lote < 1:
            raise ValueError("epocas e tamanho_lote devem ser positivos")
        if self.taxa_aprendizado <= 0 or self.decaimento_pesos < 0:
            raise ValueError("taxas de otimizacao invalidas")
        if self.pos_weight <= 0 or self.paciencia < 1:
            raise ValueError("pos_weight e paciencia devem ser positivos")
        if not 0 < self.limiar < 1:
            raise ValueError("limiar deve estar entre 0 e 1")


@dataclass
class ResultadoEpoca:
    epoca: int
    loss_treino: float
    loss_validacao: float
    tempo_segundos: float
    metricas_validacao: dict[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistroPredicao:
    paciente: str
    rotulo: int
    probabilidade: float


def configurar_reprodutibilidade(seed: int) -> None:
    """Configura geradores aleatorios sem prometer determinismo absoluto."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def escolher_dispositivo(nome: str | None = None) -> torch.device:
    if nome is not None:
        dispositivo = torch.device(nome)
        if dispositivo.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA solicitado, mas indisponivel para o PyTorch")
        return dispositivo
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _desempacotar_lote(lote: Any) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Aceita tupla ou mapping e aplica o contrato minimo da Issue #4."""

    if isinstance(lote, Mapping):
        try:
            imagens = lote["imagens"] if "imagens" in lote else lote["imagem"]
            rotulos = lote["rotulos"] if "rotulos" in lote else lote["rotulo"]
            pacientes = lote["pacientes"] if "pacientes" in lote else lote["paciente"]
        except KeyError as exc:
            raise ValueError(
                "Lote mapping deve conter imagem(ns), rotulo(s) e paciente(s)"
            ) from exc
    elif isinstance(lote, Sequence) and len(lote) == 3:
        imagens, rotulos, pacientes = lote
    else:
        raise ValueError(
            "Cada lote deve ser (imagens, rotulos, pacientes) ou mapping equivalente"
        )

    if not isinstance(imagens, torch.Tensor) or not isinstance(rotulos, torch.Tensor):
        raise TypeError("imagens e rotulos devem ser tensores PyTorch")
    if imagens.ndim != 4:
        raise ValueError(f"imagens devem ter shape [N,C,H,W], recebido {imagens.shape}")

    rotulos = rotulos.float().reshape(-1, 1)
    if imagens.shape[0] != rotulos.shape[0]:
        raise ValueError("quantidades de imagens e rotulos diferentes")
    if not torch.isfinite(imagens).all():
        raise ValueError("imagens contem NaN ou infinito")
    if not torch.isfinite(rotulos).all() or not torch.all((rotulos == 0) | (rotulos == 1)):
        raise ValueError("rotulos devem conter somente 0 ou 1")

    if isinstance(pacientes, (str, bytes)):
        pacientes_lista = [str(pacientes)]
    else:
        pacientes_lista = [str(p) for p in pacientes]
    if len(pacientes_lista) != imagens.shape[0]:
        raise ValueError("cada imagem precisa de exatamente um ID de paciente")
    return imagens, rotulos, pacientes_lista


def calcular_pos_weight_treino(
    carregador_treino: Iterable[Any],
) -> tuple[float, int, int]:
    """Calcula N negativos/N positivos, contando cada paciente uma unica vez."""

    rotulos_por_paciente: dict[str, int] = {}
    for lote in carregador_treino:
        _, rotulos, pacientes = _desempacotar_lote(lote)
        for paciente, rotulo in zip(
            pacientes, rotulos.reshape(-1).tolist(), strict=True
        ):
            atual = int(round(rotulo))
            anterior = rotulos_por_paciente.get(paciente)
            if anterior is not None and anterior != atual:
                raise ValueError(f"rotulos conflitantes para o paciente {paciente}")
            rotulos_por_paciente[paciente] = atual
    negativos = sum(rotulo == 0 for rotulo in rotulos_por_paciente.values())
    positivos = sum(rotulo == 1 for rotulo in rotulos_por_paciente.values())
    if positivos == 0 or negativos == 0:
        raise ValueError("treino precisa conter as duas classes para calcular pos_weight")
    return negativos / positivos, negativos, positivos


def _batchnorm_congelado_em_eval(modelo: nn.Module) -> None:
    """Impede atualizacao de estatisticas de BatchNorm em backbones congelados."""

    for modulo in modelo.modules():
        if isinstance(modulo, nn.modules.batchnorm._BatchNorm):
            parametros = list(modulo.parameters(recurse=False))
            if parametros and not any(p.requires_grad for p in parametros):
                modulo.eval()


def _executar_epoca(
    modelo: nn.Module,
    carregador: Iterable[Any],
    criterio: nn.Module,
    dispositivo: torch.device,
    otimizador: torch.optim.Optimizer | None,
) -> tuple[float, list[RegistroPredicao]]:
    treinando = otimizador is not None
    modelo.train(treinando)
    if treinando:
        _batchnorm_congelado_em_eval(modelo)

    soma_loss = 0.0
    quantidade = 0
    registros: list[RegistroPredicao] = []

    contexto = torch.enable_grad() if treinando else torch.no_grad()
    with contexto:
        for lote in carregador:
            imagens, rotulos, pacientes = _desempacotar_lote(lote)
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)

            if treinando:
                otimizador.zero_grad(set_to_none=True)
            logits = modelo(imagens)
            if logits.shape != rotulos.shape:
                raise ValueError(
                    f"modelo deve devolver {rotulos.shape}; devolveu {logits.shape}"
                )
            loss = criterio(logits, rotulos)
            if treinando:
                loss.backward()
                otimizador.step()

            tamanho = imagens.shape[0]
            soma_loss += float(loss.detach()) * tamanho
            quantidade += tamanho
            probabilidades = torch.sigmoid(logits.detach()).cpu().reshape(-1).tolist()
            rotulos_cpu = rotulos.detach().cpu().reshape(-1).tolist()
            registros.extend(
                RegistroPredicao(pid, int(round(rotulo)), float(probabilidade))
                for pid, rotulo, probabilidade in zip(
                    pacientes, rotulos_cpu, probabilidades, strict=True
                )
            )

    if quantidade == 0:
        raise ValueError("DataLoader vazio")
    return soma_loss / quantidade, registros


def agregar_por_paciente(
    registros: Iterable[RegistroPredicao],
) -> list[RegistroPredicao]:
    """Faz media das probabilidades das variacoes/cortes do mesmo paciente."""

    agrupados: dict[str, tuple[int, list[float]]] = {}
    for registro in registros:
        if registro.paciente in agrupados:
            rotulo, probabilidades = agrupados[registro.paciente]
            if rotulo != registro.rotulo:
                raise ValueError(
                    f"rotulos conflitantes para o paciente {registro.paciente}"
                )
            probabilidades.append(registro.probabilidade)
        else:
            agrupados[registro.paciente] = (
                registro.rotulo,
                [registro.probabilidade],
            )
    return [
        RegistroPredicao(pid, rotulo, float(np.mean(probabilidades)))
        for pid, (rotulo, probabilidades) in sorted(agrupados.items())
    ]


def calcular_metricas(
    registros: Iterable[RegistroPredicao], limiar: float = 0.5
) -> dict[str, float | int]:
    """Calcula as cinco metricas exigidas e a matriz de confusao por paciente."""

    pacientes = agregar_por_paciente(registros)
    if not pacientes:
        raise ValueError("nenhuma predicao para avaliar")

    tp = tn = fp = fn = 0
    for registro in pacientes:
        previsto = int(registro.probabilidade >= limiar)
        if registro.rotulo == 1 and previsto == 1:
            tp += 1
        elif registro.rotulo == 0 and previsto == 0:
            tn += 1
        elif registro.rotulo == 0 and previsto == 1:
            fp += 1
        else:
            fn += 1

    def divisao(numerador: float, denominador: float) -> float:
        return numerador / denominador if denominador else 0.0

    sensibilidade = divisao(tp, tp + fn)
    especificidade = divisao(tn, tn + fp)
    precisao = divisao(tp, tp + fp)
    acuracia = divisao(tp + tn, tp + tn + fp + fn)
    f1 = divisao(2 * precisao * sensibilidade, precisao + sensibilidade)
    return {
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "precisao": precisao,
        "acuracia": acuracia,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "numero_pacientes": len(pacientes),
        "limiar": limiar,
    }


def selecionar_limiar_validacao(
    registros: Iterable[RegistroPredicao],
    candidatos: Iterable[float] | None = None,
) -> tuple[float, dict[str, float | int]]:
    """Seleciona por F1; empates favorecem o limiar mais proximo de 0,5."""

    pacientes = agregar_por_paciente(registros)
    lista = list(candidatos) if candidatos is not None else np.linspace(0.05, 0.95, 19).tolist()
    if not lista or any(not 0 < valor < 1 for valor in lista):
        raise ValueError("candidatos de limiar devem estar entre 0 e 1")
    avaliacoes = [(valor, calcular_metricas(pacientes, valor)) for valor in lista]
    return max(
        avaliacoes,
        key=lambda item: (
            float(item[1]["f1"]),
            float(item[1]["sensibilidade"]),
            -abs(item[0] - 0.5),
        ),
    )


def avaliar_modelo(
    modelo: nn.Module,
    carregador: Iterable[Any],
    criterio: nn.Module,
    dispositivo: torch.device,
    limiar: float = 0.5,
) -> tuple[float, dict[str, float | int], list[RegistroPredicao]]:
    loss, registros = _executar_epoca(
        modelo, carregador, criterio, dispositivo, otimizador=None
    )
    pacientes = agregar_por_paciente(registros)
    return loss, calcular_metricas(pacientes, limiar), pacientes


def _metadados_ambiente(dispositivo: torch.device) -> dict[str, Any]:
    metadados: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "dispositivo": str(dispositivo),
        "cuda_pytorch": torch.version.cuda,
    }
    if dispositivo.type == "cuda":
        metadados["gpu"] = torch.cuda.get_device_name(dispositivo)
    return metadados


def salvar_checkpoint(
    caminho: str | Path,
    modelo: nn.Module,
    otimizador: torch.optim.Optimizer,
    configuracao: ConfiguracaoTreinamento,
    epoca: int,
    melhor_loss_validacao: float,
    historico: Sequence[ResultadoEpoca],
    metadados_preprocessamento: Mapping[str, Any],
    dispositivo: torch.device,
) -> None:
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    conteudo = {
        "formato_checkpoint": 1,
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": otimizador.state_dict(),
        "configuracao": asdict(configuracao),
        "epoca": epoca,
        "melhor_loss_validacao": melhor_loss_validacao,
        "historico": [asdict(item) for item in historico],
        "preprocessamento": dict(metadados_preprocessamento),
        "ambiente": _metadados_ambiente(dispositivo),
    }
    torch.save(conteudo, temporario)
    os.replace(temporario, destino)


def carregar_checkpoint(
    caminho: str | Path,
    modelo: nn.Module,
    dispositivo: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(caminho, map_location=dispositivo, weights_only=False)
    if checkpoint.get("formato_checkpoint") != 1:
        raise ValueError("formato de checkpoint desconhecido")
    modelo.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def registrar_validacao_checkpoint(
    caminho: str | Path,
    limiar: float,
    metricas: Mapping[str, float | int],
) -> None:
    """Acrescenta ao melhor checkpoint a decisao tomada apenas na validacao."""

    if not 0 < limiar < 1:
        raise ValueError("limiar deve estar entre 0 e 1")
    destino = Path(caminho)
    checkpoint = torch.load(destino, map_location="cpu", weights_only=False)
    if checkpoint.get("formato_checkpoint") != 1:
        raise ValueError("formato de checkpoint desconhecido")
    checkpoint["selecao_validacao"] = {
        "limiar": float(limiar),
        "metricas": dict(metricas),
    }
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    torch.save(checkpoint, temporario)
    os.replace(temporario, destino)


def treinar_experimento(
    modelo: nn.Module,
    carregador_treino: Iterable[Any],
    carregador_validacao: Iterable[Any],
    configuracao: ConfiguracaoTreinamento,
    caminho_checkpoint: str | Path,
    metadados_preprocessamento: Mapping[str, Any],
    dispositivo: str | torch.device | None = None,
) -> list[ResultadoEpoca]:
    """Treina com selecao do checkpoint exclusivamente pela validacao."""

    chaves_obrigatorias = {
        "fases",
        "normalizacao",
        "corte",
        "augmentation",
        "particoes",
    }
    ausentes = sorted(chaves_obrigatorias - set(metadados_preprocessamento))
    if ausentes:
        raise ValueError(
            "metadados de preprocessamento ausentes: " + ", ".join(ausentes)
        )
    for nome, carregador in (
        ("treino", carregador_treino),
        ("validacao", carregador_validacao),
    ):
        batch_size = getattr(carregador, "batch_size", None)
        if batch_size is not None and batch_size != configuracao.tamanho_lote:
            raise ValueError(
                f"batch do carregador de {nome} ({batch_size}) difere da configuracao "
                f"({configuracao.tamanho_lote})"
            )

    configurar_reprodutibilidade(configuracao.seed)
    device = (
        escolher_dispositivo(dispositivo)
        if isinstance(dispositivo, str) or dispositivo is None
        else dispositivo
    )
    modelo.to(device)

    parametros = [p for p in modelo.parameters() if p.requires_grad]
    if not parametros:
        raise ValueError("modelo nao possui parametros treinaveis")
    otimizador = torch.optim.Adam(
        parametros,
        lr=configuracao.taxa_aprendizado,
        weight_decay=configuracao.decaimento_pesos,
    )
    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([configuracao.pos_weight], device=device)
    )

    historico: list[ResultadoEpoca] = []
    melhor_loss = float("inf")
    epocas_sem_melhora = 0
    for epoca in range(1, configuracao.epocas + 1):
        inicio = time.perf_counter()
        loss_treino, _ = _executar_epoca(
            modelo, carregador_treino, criterio, device, otimizador
        )
        loss_validacao, metricas, _ = avaliar_modelo(
            modelo,
            carregador_validacao,
            criterio,
            device,
            configuracao.limiar,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resultado = ResultadoEpoca(
            epoca=epoca,
            loss_treino=loss_treino,
            loss_validacao=loss_validacao,
            tempo_segundos=time.perf_counter() - inicio,
            metricas_validacao=metricas,
        )
        historico.append(resultado)
        print(
            f"Epoca {epoca:03d}/{configuracao.epocas:03d} | "
            f"loss treino={loss_treino:.6f} | loss val={loss_validacao:.6f} | "
            f"F1 val={float(metricas['f1']):.4f} | {resultado.tempo_segundos:.1f}s"
        )

        if loss_validacao < melhor_loss:
            melhor_loss = loss_validacao
            epocas_sem_melhora = 0
            salvar_checkpoint(
                caminho_checkpoint,
                modelo,
                otimizador,
                configuracao,
                epoca,
                melhor_loss,
                historico,
                metadados_preprocessamento,
                device,
            )
            print(f"Checkpoint atualizado: {caminho_checkpoint}")
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= configuracao.paciencia:
                break
    return historico


def salvar_historico_csv(
    caminho: str | Path, historico: Sequence[ResultadoEpoca]
) -> None:
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "epoca",
        "loss_treino",
        "loss_validacao",
        "tempo_segundos",
        "sensibilidade",
        "especificidade",
        "precisao",
        "acuracia",
        "f1",
    ]
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for item in historico:
            linha = {
                "epoca": item.epoca,
                "loss_treino": item.loss_treino,
                "loss_validacao": item.loss_validacao,
                "tempo_segundos": item.tempo_segundos,
            }
            linha.update({campo: item.metricas_validacao[campo] for campo in campos[4:]})
            escritor.writerow(linha)


def salvar_resultado_json(
    caminho: str | Path,
    metricas: Mapping[str, float | int],
    predicoes: Sequence[RegistroPredicao],
) -> None:
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    conteudo = {
        "metricas": dict(metricas),
        "predicoes_por_paciente": [
            {
                **asdict(item),
                "previsto": int(
                    item.probabilidade >= float(metricas.get("limiar", 0.5))
                ),
                "correto": int(
                    item.probabilidade >= float(metricas.get("limiar", 0.5))
                )
                == item.rotulo,
            }
            for item in predicoes
        ],
    }
    destino.write_text(
        json.dumps(conteudo, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def salvar_grafico_convergencia(
    caminho: str | Path, historico: Sequence[ResultadoEpoca]
) -> None:
    if not historico:
        raise ValueError("historico vazio")
    import matplotlib.pyplot as plt

    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, eixo = plt.subplots(figsize=(7, 4))
    epocas = [item.epoca for item in historico]
    eixo.plot(epocas, [item.loss_treino for item in historico], label="Treino")
    eixo.plot(epocas, [item.loss_validacao for item in historico], label="Validacao")
    eixo.set(xlabel="Epoca", ylabel="BCEWithLogitsLoss", title="Convergencia")
    eixo.grid(alpha=0.25)
    eixo.legend()
    figura.tight_layout()
    figura.savefig(destino, dpi=160)
    plt.close(figura)


def salvar_matriz_confusao(
    caminho: str | Path, metricas: Mapping[str, float | int]
) -> None:
    import matplotlib.pyplot as plt

    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    matriz = np.array(
        [[metricas["tn"], metricas["fp"]], [metricas["fn"], metricas["tp"]]],
        dtype=int,
    )
    figura, eixo = plt.subplots(figsize=(4, 4))
    imagem = eixo.imshow(matriz, cmap="Blues")
    for linha in range(2):
        for coluna in range(2):
            eixo.text(coluna, linha, str(matriz[linha, coluna]), ha="center", va="center")
    eixo.set_xticks([0, 1], labels=["pCR=0", "pCR=1"])
    eixo.set_yticks([0, 1], labels=["pCR=0", "pCR=1"])
    eixo.set(xlabel="Previsto", ylabel="Real", title="Matriz de confusao por paciente")
    figura.colorbar(imagem, ax=eixo)
    figura.tight_layout()
    figura.savefig(destino, dpi=160)
    plt.close(figura)
