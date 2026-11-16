"""Integra os datasets das Issues #1/#2 ao treinamento da Issue #4.

Os volumes ficam fora do repositorio. Defina ``PAI_DATASET_DIR`` apontando para a
pasta que contem diretamente ``dce/`` e ``mask/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
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
        self.pacientes = dataset_base.pacientes
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.dataset_base)

    def __getitem__(self, indice: int) -> dict[str, Any]:
        if indice not in self._cache:
            self._cache[indice] = self.dataset_base[indice]
        return self._cache[indice]


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
    teste = DatasetDCE(pacientes_teste, fase=fase, estrategia=corte)
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
        "teste": DataLoader(teste, shuffle=False, **opcoes),
        "pos_weight": float(peso.item()),
        "negativos_treino": negativos,
        "positivos_treino": positivos,
        "metadados": {
            "dataset": "BreastDCEDL I-SPY1 MinCrop",
            "dataset_dir": str(pasta_dataset.resolve()),
            "csv": str(caminho_csv.resolve()),
            "fases": list(FASES) if fase == "tres" else [fase],
            "normalizacao": "percentis 1/99 conjuntos por amostra; clip [0,1]",
            "corte": corte,
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
            "cache_preprocessamento": "memoria por processo; sem alterar imagens",
        },
    }
