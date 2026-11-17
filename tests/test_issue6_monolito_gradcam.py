"""Testes de regressao do monolito; independentes do dataset real e de GUI aberta."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


CAMINHO = Path(__file__).resolve().parents[1] / "entrega_final_G10" / "trabalho_pai.py"
spec = importlib.util.spec_from_file_location("monolito_pai_teste", CAMINHO)
assert spec and spec.loader
pai = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = pai
spec.loader.exec_module(pai)


class RedeTeste(nn.Module):
    def __init__(self):
        super().__init__()
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.classificador = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 1),
        )
        with torch.no_grad():
            self.caracteristicas[0].weight.fill_(1)
            self.classificador[-1].weight.fill_(-1)
            self.classificador[-1].bias.fill_(-0.2)

    def forward(self, imagem):
        return self.classificador(self.caracteristicas(imagem))


def test_cam_previsao_negativa_explora_logit_negativo():
    modelo = RedeTeste().eval()
    imagem = torch.zeros(1, 1, 16, 16)
    imagem[:, :, 3:10, 4:12] = 1
    mapa, prob, dados = pai.gerar_gradcam(
        modelo, imagem, "cnn", classe_alvo="prevista", limiar=0.5
    )
    assert prob < 0.5
    assert dados["classe_prevista"] == dados["classe_explicada"] == 0
    assert mapa.shape == (16, 16)
    assert np.isfinite(mapa).all()
    assert mapa.max() > 0
    assert np.sum(mapa[3:10, 4:12]) > np.sum(mapa[:2, :2])
    # Hooks temporarios nao devem ficar registrados no modelo.
    assert len(modelo.caracteristicas[1]._forward_hooks) == 0


def test_cam_pcr1_nao_inventa_calor_se_gradiente_negativo():
    modelo = RedeTeste().eval()
    imagem = torch.ones(1, 1, 16, 16)
    mapa, _, dados = pai.gerar_gradcam(modelo, imagem, "cnn", classe_alvo="pcr1")
    assert dados["classe_explicada"] == 1
    assert dados["mapa_nulo"]
    assert not mapa.any()


def test_mapa_quantitativo_nao_usa_mascara_para_modificar_cam():
    mapa = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    mascara = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    referencia = mapa.copy()
    resumo = pai.resumir_relevancia_gradcam(
        mapa, mascara, {
            "classe_explicada": 1,
            "resolucao_nativa": (1, 1),
            "maximo_bruto": 2.5,
            "mapa_nulo": False,
        },
    )
    assert "50.0%" in resumo
    assert "25.0%" in resumo
    assert np.array_equal(mapa, referencia)


def test_tres_fases_e_mascara_compartilham_z(monkeypatch):
    mascara = np.zeros((3, 256, 256), dtype=np.float32)
    mascara[2, 11:15, 14:19] = 1
    dce = [
        np.full_like(mascara, 5),
        np.full_like(mascara, 20),
        np.full_like(mascara, 80),
    ]
    dce[0][2, 0, 0] = 7
    dce[1][2, 0, 0] = 25
    dce[2][2, 0, 0] = 85
    volumes = {"mask": mascara, "pre": dce[0], "early": dce[1], "late": dce[2]}
    def fake_read(caminho):
        return volumes[str(caminho)], np.eye(4)
    monkeypatch.setattr(pai, "ler_volume", fake_read)
    paciente = pai.Paciente("ISPY1_X", 1, (Path("pre"), Path("early"), Path("late")), Path("mask"))
    imagem, tumor, z = pai.preparar_paciente(
        paciente,
        {"configuracao_temporal": "tres", "estrategia_corte": "maior_tumor"},
        {"normalizacao": {"tipo": "percentil_1_99_conjunto"}},
    )
    assert z == 2
    assert imagem.shape == (3, 256, 256)
    assert tumor.shape == (256, 256)
    assert tumor.sum() == 20
    assert torch.all(imagem[0] <= imagem[1])
    assert torch.all(imagem[1] <= imagem[2])


def test_cam_rejeita_classe_invalida():
    with pytest.raises(ValueError, match="Classe-alvo"):
        pai.gerar_gradcam(RedeTeste(), torch.ones(1, 1, 8, 8), "cnn", classe_alvo="invalida")
