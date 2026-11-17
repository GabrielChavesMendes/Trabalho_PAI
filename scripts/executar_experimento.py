"""CLI da Issue #4 com fabrica de DataLoaders fornecida pela Issue #2.

Exemplo futuro:
    python scripts/executar_experimento.py \
      --loader-factory modulos.fabrica_carregadores:criar_carregadores \
      --arquitetura cnn --canais 1 --fases post_early --corte central

A funcao indicada por ``--loader-factory`` deve aceitar os argumentos nomeados
``configuracao_temporal``, ``estrategia_corte``, ``tamanho_lote`` e ``seed`` e
devolver um mapping com ``treino``, ``validacao``, ``teste`` (opcional) e
``metadados``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.modulo_redes import criar_modelo  # noqa: E402
from modulos.modulo_treinamento import (  # noqa: E402
    ConfiguracaoTreinamento,
    avaliar_modelo,
    calcular_metricas,
    calcular_pos_weight_treino,
    carregar_checkpoint,
    escolher_dispositivo,
    salvar_grafico_convergencia,
    salvar_historico_csv,
    salvar_matriz_confusao,
    registrar_validacao_checkpoint,
    salvar_resultado_json,
    selecionar_limiar_validacao,
    treinar_experimento,
)


def _carregar_fabrica(referencia: str) -> Callable[..., Mapping[str, Any]]:
    try:
        nome_modulo, nome_funcao = referencia.split(":", maxsplit=1)
        modulo = importlib.import_module(nome_modulo)
        fabrica = getattr(modulo, nome_funcao)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            "--loader-factory deve usar o formato pacote.modulo:funcao"
        ) from exc
    if not callable(fabrica):
        raise TypeError("loader factory nao e chamavel")
    return fabrica


def _nome_experimento(configuracao: ConfiguracaoTreinamento) -> str:
    temporal = configuracao.configuracao_temporal.replace("_", "-")
    corte = configuracao.estrategia_corte.replace("_", "-")
    return f"{configuracao.arquitetura}__{temporal}__{corte}__seed-{configuracao.seed}"


def _avaliar_e_exportar(
    modelo: torch.nn.Module,
    carregador: Any,
    criterio: torch.nn.Module,
    dispositivo: torch.device,
    limiar: float,
    prefixo: Path,
    selecionar_limiar: bool,
    criterio_limiar: str = "f1",
) -> tuple[float, dict[str, float | int]]:
    loss, _, predicoes = avaliar_modelo(
        modelo, carregador, criterio, dispositivo, limiar
    )
    if selecionar_limiar:
        limiar, metricas = selecionar_limiar_validacao(predicoes, criterio=criterio_limiar)
    else:
        metricas = calcular_metricas(predicoes, limiar)
    metricas = {**metricas, "loss": loss}
    salvar_resultado_json(prefixo.with_suffix(".json"), metricas, predicoes)
    salvar_matriz_confusao(
        prefixo.with_name(prefixo.name + "__confusion-matrix.png"), metricas
    )
    return limiar, metricas


def main() -> int:
    parser = argparse.ArgumentParser(description="Executar uma configuracao da Issue #4")
    parser.add_argument("--loader-factory", required=True)
    parser.add_argument("--arquitetura", choices=("cnn", "resnet50", "mobilenetv2"), required=True)
    parser.add_argument("--canais", type=int, choices=(1, 3), required=True)
    parser.add_argument("--fases", required=True)
    parser.add_argument("--corte", required=True)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, help="Sobrescreve peso de classe calculado do treino; registrar no artigo")
    parser.add_argument("--finetune-resnet", action="store_true", help="Descongelar layer4 da ResNet50")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate para layer4")
    parser.add_argument("--criterio-limiar", choices=("f1", "acuracia_balanceada"), default="acuracia_balanceada")
    parser.add_argument("--criterio-checkpoint", choices=("loss", "auc_roc", "acuracia_balanceada"), default="loss")
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    parser.add_argument("--checkpoints", type=Path, default=RAIZ / "checkpoints")
    parser.add_argument("--avaliar-teste", action="store_true")
    args = parser.parse_args()

    fabrica = _carregar_fabrica(args.loader_factory)
    carregadores = fabrica(
        configuracao_temporal=args.fases,
        estrategia_corte=args.corte,
        tamanho_lote=args.batch,
        seed=args.seed,
    )
    obrigatorios = {"treino", "validacao", "metadados"}
    ausentes = sorted(obrigatorios - set(carregadores))
    if ausentes:
        raise ValueError("loader factory nao devolveu: " + ", ".join(ausentes))

    if "pos_weight" in carregadores:
        pos_weight = float(carregadores["pos_weight"])
        negativos = int(carregadores["negativos_treino"])
        positivos = int(carregadores["positivos_treino"])
    else:
        pos_weight, negativos, positivos = calcular_pos_weight_treino(
            carregadores["treino"]
        )
    peso_original = pos_weight
    if args.pos_weight is not None:
        if args.pos_weight <= 0:
            raise ValueError("--pos-weight precisa ser positivo")
        pos_weight = args.pos_weight
    if args.finetune_resnet and args.arquitetura != "resnet50":
        raise ValueError("--finetune-resnet exige --arquitetura resnet50")
    print(
        f"Treino por paciente: negativos={negativos}, positivos={positivos}, "
        f"peso_observado={peso_original:.6g}, pos_weight_usado={pos_weight:.6g}"
    )
    taxa = args.lr if args.lr is not None else (1e-3 if args.arquitetura == "cnn" else 1e-4)
    configuracao = ConfiguracaoTreinamento(
        arquitetura=args.arquitetura,
        canais_entrada=args.canais,
        configuracao_temporal=args.fases,
        estrategia_corte=args.corte,
        epocas=args.epocas,
        tamanho_lote=args.batch,
        taxa_aprendizado=taxa,
        decaimento_pesos=args.weight_decay,
        pos_weight=pos_weight,
        paciencia=args.paciencia,
        seed=args.seed,
        taxa_aprendizado_backbone=args.lr_backbone if args.finetune_resnet else None,
        criterio_checkpoint=args.criterio_checkpoint,
    )
    nome = _nome_experimento(configuracao)
    checkpoint = args.checkpoints / f"{nome}.pt"
    dispositivo = escolher_dispositivo(args.device)
    modelo = criar_modelo(
        args.arquitetura,
        args.canais,
        pre_treinado=args.arquitetura != "cnn",
    )
    if args.finetune_resnet:
        for parametro in modelo.layer4.parameters():
            parametro.requires_grad = True
    carregadores["metadados"]["ajuste_hyperparametros"] = {
        "pos_weight_observado": peso_original,
        "pos_weight_utilizado": pos_weight,
        "fine_tuning_resnet_layer4": bool(args.finetune_resnet),
        "lr_backbone": args.lr_backbone if args.finetune_resnet else None,
        "criterio_limiar": args.criterio_limiar,
    }
    historico = treinar_experimento(
        modelo,
        carregadores["treino"],
        carregadores["validacao"],
        configuracao,
        checkpoint,
        carregadores["metadados"],
        dispositivo,
    )
    salvar_historico_csv(args.saida / f"{nome}__history.csv", historico)
    salvar_grafico_convergencia(args.saida / f"{nome}__loss.png", historico)

    dados = carregar_checkpoint(checkpoint, modelo, dispositivo)
    criterio = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=dispositivo)
    )
    limiar, metricas_validacao = _avaliar_e_exportar(
        modelo,
        carregadores["validacao"],
        criterio,
        dispositivo,
        configuracao.limiar,
        args.saida / f"{nome}__validation",
        selecionar_limiar=True,
        criterio_limiar=args.criterio_limiar,
    )
    registrar_validacao_checkpoint(checkpoint, limiar, {**metricas_validacao, "criterio_selecao_limiar": args.criterio_limiar})
    print(
        f"Checkpoint selecionado na epoca {dados['epoca']}; "
        f"limiar escolhido na validacao={limiar:.3f}"
    )

    if args.avaliar_teste:
        if "teste" not in carregadores:
            raise ValueError("--avaliar-teste requer o carregador 'teste'")
        _avaliar_e_exportar(
            modelo,
            carregadores["teste"],
            criterio,
            dispositivo,
            limiar,
            args.saida / f"{nome}__test",
            selecionar_limiar=False,
        )
        print("Teste reservado avaliado com o limiar fixado pela validacao.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
