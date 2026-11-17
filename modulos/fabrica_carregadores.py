"""Integra os datasets das Issues #1/#2 ao treinamento da Issue #4.

Os volumes ficam fora do repositorio. Defina ``PAI_DATASET_DIR`` apontando para a
pasta que contem diretamente ``dce/`` e ``mask/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from modulos.modulo_balanceamento import calcular_pos_weight_treino
from modulos.modulo_dataset import (
    FASES,
    DatasetDCE,
    carregar_metadados,
    criar_manifesto,
    selecionar_particao,
    verificar_arquivos,
)
from modulos.modulo_dataset_issue1 import DatasetDCEAumentado


RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]
CSV_PADRAO = RAIZ_REPOSITORIO / "dataset" / "BreastDCEDL_SPY1_min_crop_metadata.csv"


class DatasetCacheMemoria(Dataset):
    """Evita descompactar o mesmo NIfTI dez vezes durante o augmentation."""

    def __init__(self, dataset_base: Dataset):
        self.dataset_base = dataset_base
        if hasattr(dataset_base, "pacientes"):
            self.pacientes = dataset_base.pacientes
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.dataset_base)

    def __getitem__(self, indice: int) -> dict[str, Any]:
        if indice not in self._cache:
            self._cache[indice] = self.dataset_base[indice]
        return self._cache[indice]


class DatasetPadronizado(Dataset):
    """Aplica estatisticas calculadas exclusivamente nos pacientes de treino."""

    def __init__(self, dataset_base: Dataset, medias: torch.Tensor, desvios: torch.Tensor):
        self.dataset_base = dataset_base
        self.pacientes = dataset_base.pacientes
        self.medias = medias.reshape(-1, 1, 1)
        self.desvios = desvios.reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.dataset_base)

    def __getitem__(self, indice: int) -> dict[str, Any]:
        amostra = dict(self.dataset_base[indice])
        amostra["imagem"] = (amostra["imagem"] - self.medias) / self.desvios
        return amostra


class DatasetComMascara(Dataset):
    """Zera o fundo usando a segmentacao, opcao explicitamente prevista no enunciado."""

    def __init__(self, dataset_base: Dataset):
        self.dataset_base = dataset_base
        self.pacientes = dataset_base.pacientes

    def __len__(self) -> int:
        return len(self.dataset_base)

    def __getitem__(self, indice: int) -> dict[str, Any]:
        amostra = dict(self.dataset_base[indice])
        amostra["imagem"] = amostra["imagem"] * amostra["mascara"]
        return amostra


class DatasetRecorteROI(Dataset):
    """Amplia a regiao tumoral mantendo contexto e o tamanho original da entrada."""

    def __init__(self, dataset_base: Dataset, fator_contexto: float = 2.0):
        if fator_contexto < 1:
            raise ValueError("fator_contexto deve ser pelo menos 1")
        self.dataset_base = dataset_base
        self.pacientes = dataset_base.pacientes
        self.fator_contexto = fator_contexto

    def __len__(self) -> int:
        return len(self.dataset_base)

    def __getitem__(self, indice: int) -> dict[str, Any]:
        amostra = dict(self.dataset_base[indice])
        imagem = amostra["imagem"]
        mascara = amostra["mascara"]
        posicoes = torch.nonzero(mascara[0] > 0, as_tuple=False)
        if posicoes.numel() == 0:
            raise ValueError("recorte ROI requer mascara nao vazia no corte selecionado")
        altura, largura = imagem.shape[-2:]
        y0, x0 = posicoes.min(dim=0).values.tolist()
        y1, x1 = (posicoes.max(dim=0).values + 1).tolist()
        centro_y = (y0 + y1) / 2
        centro_x = (x0 + x1) / 2
        lado = min(max(int(round(max(y1 - y0, x1 - x0) * self.fator_contexto)), 64), altura, largura)
        inicio_y = min(max(int(round(centro_y - lado / 2)), 0), altura - lado)
        inicio_x = min(max(int(round(centro_x - lado / 2)), 0), largura - lado)
        fim_y, fim_x = inicio_y + lado, inicio_x + lado
        imagem_roi = imagem[:, inicio_y:fim_y, inicio_x:fim_x].unsqueeze(0)
        mascara_roi = mascara[:, inicio_y:fim_y, inicio_x:fim_x].unsqueeze(0)
        amostra["imagem"] = F.interpolate(
            imagem_roi, size=(altura, largura), mode="bilinear", align_corners=False
        ).squeeze(0)
        amostra["mascara"] = F.interpolate(
            mascara_roi, size=(altura, largura), mode="nearest"
        ).squeeze(0)
        amostra["recorte_roi"] = {
            "inicio_y": inicio_y, "inicio_x": inicio_x, "lado": lado,
            "fator_contexto": self.fator_contexto,
        }
        return amostra


def _estatisticas_treino(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    soma = soma_quadrados = None
    quantidade_pixels = 0
    for indice in range(len(dataset)):
        imagem = dataset[indice]["imagem"].double()
        atual = imagem.sum(dim=(1, 2))
        atual_quadrados = imagem.square().sum(dim=(1, 2))
        soma = atual if soma is None else soma + atual
        soma_quadrados = atual_quadrados if soma_quadrados is None else soma_quadrados + atual_quadrados
        quantidade_pixels += imagem.shape[1] * imagem.shape[2]
    if soma is None or quantidade_pixels == 0:
        raise ValueError("treino vazio para calcular normalizacao")
    medias = soma / quantidade_pixels
    variancias = soma_quadrados / quantidade_pixels - medias.square()
    desvios = variancias.clamp_min(1e-12).sqrt()
    return medias.float(), desvios.float()


def _resolver_fase(configuracao_temporal: str) -> str:
    aliases = {"tres_fases": "tres", "tres-fases": "tres"}
    fase = aliases.get(configuracao_temporal, configuracao_temporal)
    if fase not in (*FASES, "tres"):
        raise ValueError("Use pre, post_early, post_late ou tres como configuracao temporal")
    return fase


def _resolver_corte(estrategia_corte: str) -> str:
    aliases = {
        "maior_area": "maior_tumor",
        "maior-area": "maior_tumor",
        "maior_area_tumoral": "maior_tumor",
    }
    corte = aliases.get(estrategia_corte, estrategia_corte)
    if corte not in ("central", "maior_tumor"):
        raise ValueError("Use central ou maior_tumor como estrategia de corte")
    return corte


def _opcoes_dataloader(tamanho_lote: int, num_workers: int) -> dict[str, Any]:
    opcoes: dict[str, Any] = {
        "batch_size": tamanho_lote,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        opcoes.update(persistent_workers=True, prefetch_factor=2)
    return opcoes


def criar_carregadores(
    configuracao_temporal: str,
    estrategia_corte: str,
    tamanho_lote: int,
    seed: int,
    normalizacao_entrada: str = "percentil",
    usar_mascara: bool = False,
    recorte_roi: bool = False,
) -> dict[str, Any]:
    """Cria treino/validacao aumentados e teste reservado sem augmentation."""

    fase = _resolver_fase(configuracao_temporal)
    corte = _resolver_corte(estrategia_corte)
    pasta_dataset = Path(
        os.environ.get(
            "PAI_DATASET_DIR",
            RAIZ_REPOSITORIO / "dataset" / "BreastDCEDL_ISPY1_min_crop",
        )
    ).expanduser()
    caminho_csv = Path(os.environ.get("PAI_CSV", CSV_PADRAO)).expanduser()
    num_workers = int(os.environ.get("PAI_NUM_WORKERS", "0"))
    if tamanho_lote < 1 or num_workers < 0:
        raise ValueError("tamanho_lote deve ser positivo e PAI_NUM_WORKERS nao negativo")
    if normalizacao_entrada not in ("percentil", "zscore_treino", "imagenet"):
        raise ValueError("normalizacao_entrada deve ser percentil, zscore_treino ou imagenet")

    tabela = carregar_metadados(caminho_csv)
    manifesto = criar_manifesto(tabela, pasta_dataset)
    erros = verificar_arquivos(manifesto)
    if erros:
        exemplos = "\n".join(erros[:5])
        raise FileNotFoundError(
            f"Dataset incompleto em {pasta_dataset}: {len(erros)} arquivo(s) ausente(s).\n"
            f"Primeiros erros:\n{exemplos}"
        )

    pacientes_treino = selecionar_particao(manifesto, "treino")
    pacientes_validacao = selecionar_particao(manifesto, "validacao")
    pacientes_teste = selecionar_particao(manifesto, "teste")
    conjuntos_ids = [
        {p.identificador for p in pacientes}
        for pacientes in (pacientes_treino, pacientes_validacao, pacientes_teste)
    ]
    if any(conjuntos_ids[i] & conjuntos_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Paciente presente em mais de uma particao")

    treino_base = DatasetCacheMemoria(
        DatasetDCE(pacientes_treino, fase=fase, estrategia=corte)
    )
    validacao_base = DatasetCacheMemoria(
        DatasetDCE(pacientes_validacao, fase=fase, estrategia=corte)
    )
    teste_base: Dataset = DatasetDCE(pacientes_teste, fase=fase, estrategia=corte)
    if recorte_roi:
        if corte != "maior_tumor":
            raise ValueError("recorte ROI requer estrategia maior_tumor")
        treino_base = DatasetRecorteROI(treino_base)
        validacao_base = DatasetRecorteROI(validacao_base)
        teste_base = DatasetRecorteROI(teste_base)
    if usar_mascara:
        treino_base = DatasetComMascara(treino_base)
        validacao_base = DatasetComMascara(validacao_base)
        teste_base = DatasetComMascara(teste_base)
    estatisticas_normalizacao: dict[str, Any] = {"tipo": "percentil_1_99_conjunto"}
    if normalizacao_entrada == "zscore_treino":
        medias, desvios = _estatisticas_treino(treino_base)
        treino_base = DatasetPadronizado(treino_base, medias, desvios)
        validacao_base = DatasetPadronizado(validacao_base, medias, desvios)
        teste_base = DatasetPadronizado(teste_base, medias, desvios)
        estatisticas_normalizacao = {
            "tipo": "percentil_1_99_conjunto_seguido_zscore_treino_por_canal",
            "medias_treino": medias.tolist(),
            "desvios_treino": desvios.tolist(),
        }
    elif normalizacao_entrada == "imagenet":
        if fase == "tres":
            medias = torch.tensor([0.485, 0.456, 0.406])
            desvios = torch.tensor([0.229, 0.224, 0.225])
        else:
            # Aproximacao para a convolucao RGB adaptada por soma dos filtros.
            medias = torch.tensor([0.449])
            desvios = torch.tensor([0.226])
        treino_base = DatasetPadronizado(treino_base, medias, desvios)
        validacao_base = DatasetPadronizado(validacao_base, medias, desvios)
        teste_base = DatasetPadronizado(teste_base, medias, desvios)
        estatisticas_normalizacao = {
            "tipo": "percentil_1_99_seguido_normalizacao_imagenet",
            "medias": medias.tolist(),
            "desvios": desvios.tolist(),
        }
    treino = DatasetDCEAumentado(treino_base, "treino")
    validacao = DatasetDCEAumentado(validacao_base, "validacao")

    gerador = torch.Generator().manual_seed(seed)
    opcoes = _opcoes_dataloader(tamanho_lote, num_workers)
    peso = calcular_pos_weight_treino(pacientes_treino)
    negativos = sum(p.pcr == 0 for p in pacientes_treino)
    positivos = sum(p.pcr == 1 for p in pacientes_treino)
    return {
        "treino": DataLoader(treino, shuffle=True, generator=gerador, **opcoes),
        "validacao": DataLoader(validacao, shuffle=False, **opcoes),
        "teste": DataLoader(teste_base, shuffle=False, **opcoes),
        "pos_weight": float(peso.item()),
        "negativos_treino": negativos,
        "positivos_treino": positivos,
        "metadados": {
            "dataset": "BreastDCEDL I-SPY1 MinCrop",
            "dataset_dir": str(pasta_dataset.resolve()),
            "csv": str(caminho_csv.resolve()),
            "fases": list(FASES) if fase == "tres" else [fase],
            "normalizacao": estatisticas_normalizacao,
            "corte": corte,
            "mascara_como_filtro": usar_mascara,
            "recorte_roi": recorte_roi,
            "augmentation": {
                "particoes": ["treino", "validacao"],
                "angulos_graus": [-20, -10, 0, 10, 20],
                "espelhamento": [False, True],
                "variacoes_por_paciente": 10,
                "teste_aumentado": False,
            },
            "particoes": {"treino": 0, "validacao": 2, "teste": 1},
            "pacientes": {"treino": 104, "validacao": 33, "teste": 35},
            "num_workers": num_workers,
            "cache_preprocessamento": "cortes em memoria por processo; variantes sob demanda",
        },
    }
