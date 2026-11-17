"""Regressoes para a integracao sem alterar os treinamentos antigos."""
import copy

import pytest
import torch
from torch.utils.data import Dataset

from modulos.fabrica_carregadores import DatasetNormalizacaoImageNet
from modulos.checkpoints_integrados_issue4 import (
    normalizacao_checkpoint, validar_configuracao,
)


class _DatasetSintetico(Dataset):
    pacientes = ["teste"]
    def __init__(self, canais):
        self.canais = canais
    def __len__(self):
        return 1
    def __getitem__(self, i):
        return {"imagem": torch.ones(self.canais, 8, 8) * 0.5,
                "mascara": torch.ones(1, 8, 8), "paciente": "teste"}


@pytest.mark.parametrize("canais,esperado", [
    (1, (0.5 - 0.449) / 0.226),
    (3, (0.5 - 0.485) / 0.229),
])
def test_normalizacao_imagenet_antes_augmentation(canais, esperado):
    dataset = DatasetNormalizacaoImageNet(_DatasetSintetico(canais))
    amostra = dataset[0]
    assert amostra["imagem"].shape == (canais, 8, 8)
    assert float(amostra["imagem"][0, 0, 0]) == pytest.approx(esperado, abs=1e-6)
    assert torch.equal(amostra["mascara"], torch.ones(1, 8, 8))
    assert "paciente" in amostra


def _checkpoint_minimo():
    return {
        "configuracao": {"arquitetura": "resnet50", "canais_entrada": 1,
                         "configuracao_temporal": "post_early", "estrategia_corte": "maior_tumor"},
        "preprocessamento": {
            "fases": ["post_early"], "corte": "maior_tumor",
            "normalizacao": {"tipo": "percentil_1_99_seguido_normalizacao_imagenet",
                              "medias": [0.449], "desvios": [0.226]},
            "mascara_como_filtro": False, "recorte_roi": False,
        },
        "selecao_validacao": {"limiar": 0.25, "metricas": {"f1": 0.57}},
    }


def test_validar_metadados_imagenet():
    ck = _checkpoint_minimo()
    validar_configuracao(ck)
    assert normalizacao_checkpoint(ck) == "imagenet"
    ck2 = copy.deepcopy(ck)
    ck2["preprocessamento"]["normalizacao"] = {"tipo": "percentil_1_99_conjunto"}
    validar_configuracao(ck2)
    assert normalizacao_checkpoint(ck2) == "percentil"


@pytest.mark.parametrize("mudanca", [
    ("configuracao", "canais_entrada", 3),
    ("preprocessamento", "representacao_entrada", "vizinhanca_2p5d"),
    ("preprocessamento", "recorte_roi", True),
    ("preprocessamento", "mascara_como_filtro", True),
])
def test_recusar_configuracao_incompativel(mudanca):
    ck = _checkpoint_minimo()
    ck[mudanca[0]][mudanca[1]] = mudanca[2]
    with pytest.raises(ValueError):
        validar_configuracao(ck)
