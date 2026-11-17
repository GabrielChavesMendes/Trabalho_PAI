import pytest
import torch
from torch import nn

from modulos.modulo_gradcam import gerar_gradcam, resolver_camada_alvo, sobrepor_gradcam
from modulos.modulo_redes import CNNSimples


def test_gradcam_cnn_produz_mapa_normalizado_e_probabilidade():
    torch.manual_seed(7)
    modelo = CNNSimples(canais_entrada=1)
    imagem = torch.rand(1, 1, 64, 64)

    resultado = gerar_gradcam(modelo, imagem, arquitetura="cnn")

    assert resultado.mapa.shape == (64, 64)
    assert torch.isfinite(resultado.mapa).all()
    assert 0.0 <= float(resultado.mapa.min()) <= 1.0
    assert 0.0 <= float(resultado.mapa.max()) <= 1.0
    assert 0.0 <= resultado.probabilidade_pcr <= 1.0


def test_gradcam_remove_hook_apos_execucao():
    modelo = CNNSimples(canais_entrada=1)
    camada = resolver_camada_alvo(modelo, "cnn")
    hooks_antes = len(camada._forward_hooks)

    gerar_gradcam(modelo, torch.rand(1, 1, 64, 64), camada_alvo=camada)

    assert len(camada._forward_hooks) == hooks_antes


def test_gradcam_rejeita_lote_com_mais_de_uma_imagem():
    modelo = CNNSimples(canais_entrada=1)
    with pytest.raises(ValueError, match="exatamente uma"):
        gerar_gradcam(modelo, torch.rand(2, 1, 64, 64), arquitetura="cnn")


def test_gradcam_rejeita_arquitetura_incompativel():
    with pytest.raises(ValueError, match="Nao foi possivel resolver"):
        resolver_camada_alvo(nn.Linear(4, 1), "desconhecida")


def test_sobreposicao_retorna_rgb_e_valida_dimensoes():
    imagem = torch.rand(32, 32)
    mapa = torch.rand(32, 32)
    saida = sobrepor_gradcam(imagem, mapa)
    assert saida.shape == (32, 32, 3)
    assert torch.isfinite(saida).all()

    with pytest.raises(ValueError, match="mesmo tamanho"):
        sobrepor_gradcam(imagem, torch.rand(16, 16))
