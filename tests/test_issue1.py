import numpy as np
import pytest
import torch
from types import SimpleNamespace

from modulos.modulo_augmentacao import (
    gerar_variacoes, listar_variacoes, transformar_par, validar_particao_aumentada,
)
from modulos.modulo_balanceamento import calcular_pos_weight_treino, contar_classes
from modulos.modulo_dataset_issue1 import DatasetDCEAumentado


def test_dez_variacoes_por_padrao():
    imagem = np.arange(2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32)
    mascara = np.zeros((1, 32, 32), dtype=np.float32)
    mascara[:, 8:18, 10:20] = 1
    variacoes = gerar_variacoes(imagem, mascara)
    assert len(variacoes) == 10
    assert len({v.nome for v, _, _ in variacoes}) == 10
    assert variacoes[0][1].shape == imagem.shape
    assert variacoes[0][2].shape == mascara.shape
    assert set(np.unique(variacoes[0][2])).issubset({0.0, 1.0})


def test_angulos_zero_original_e_espelhado():
    imagem = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    original, _ = transformar_par(imagem, None, 0, False)
    espelho, _ = transformar_par(imagem, None, 0, True)
    np.testing.assert_array_equal(original, imagem)
    np.testing.assert_array_equal(espelho, imagem[..., ::-1])


def test_mascara_e_imagem_transformadas_em_conjunto():
    imagem = np.zeros((1, 40, 40), dtype=np.float32)
    mascara = np.zeros((1, 40, 40), dtype=np.float32)
    imagem[:, 10:20, 10:20] = 1
    mascara[:, 10:20, 10:20] = 1
    img_t, mask_t = transformar_par(imagem, mascara, 20, True)
    assert img_t.shape == mask_t.shape
    assert set(np.unique(mask_t)).issubset({0.0, 1.0})
    # A geometria binária continua sobreposta após a mesma transformação.
    assert np.array_equal(img_t > 0.5, mask_t > 0.5)


@pytest.mark.parametrize("particao", ["teste", "validacao", "Test", ""])
def test_bloqueia_augmentacao_fora_treino(particao):
    with pytest.raises(ValueError):
        validar_particao_aumentada(particao)


def test_dataset_aumentado_multiplica_itens_e_preserva_metadados():
    class Base:
        def __len__(self): return 2
        def __getitem__(self, idx):
            return {
                "imagem": torch.ones((1, 16, 16)),
                "mascara": torch.ones((1, 16, 16)),
                "rotulo": torch.tensor(1.0),
                "paciente": f"P{idx}",
                "corte_z": 4,
            }
    ds = DatasetDCEAumentado(Base(), "treino")
    assert len(ds) == 20
    item = ds[0]
    assert item["paciente"] == "P0"
    assert item["rotulo"].item() == 1
    assert item["imagem"].shape == (1, 16, 16)
    assert item["mascara"].shape == (1, 16, 16)
    assert "augmentacao" in item


def test_peso_calculado_apenas_com_treino():
    treino = [SimpleNamespace(pcr=0, particao_codigo=0) for _ in range(6)]
    treino += [SimpleNamespace(pcr=1, particao_codigo=0) for _ in range(2)]
    assert contar_classes(treino) == {0: 6, 1: 2}
    assert calcular_pos_weight_treino(treino).item() == 3
    misto = treino + [SimpleNamespace(pcr=1, particao_codigo=1)]
    with pytest.raises(ValueError, match="somente"):
        calcular_pos_weight_treino(misto)
