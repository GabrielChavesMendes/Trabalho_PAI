import numpy as np
import pytest
import torch
from types import SimpleNamespace

from modulos.modulo_augmentacao import gerar_variacoes, transformar_par
from modulos.modulo_balanceamento import calcular_pos_weight_treino, contar_classes
from modulos.modulo_dataset_issue1 import DatasetDCEAumentado


def test_dez_variacoes_por_padrao():
    imagem = np.arange(2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32)
    mascara = np.zeros((1, 32, 32), dtype=np.float32)
    mascara[:, 8:18, 10:20] = 1
    variacoes = gerar_variacoes(imagem, mascara)
    assert len(variacoes) == 10
    assert len({variacao.nome for variacao, _, _ in variacoes}) == 10
    assert variacoes[0][1].shape == imagem.shape
    assert variacoes[0][2].shape == mascara.shape
    assert set(np.unique(variacoes[0][2])).issubset({0.0, 1.0})


def test_rotacao_zero_original_e_espelhado():
    imagem = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    original, _ = transformar_par(imagem, None, 0, False)
    espelhada, _ = transformar_par(imagem, None, 0, True)
    np.testing.assert_array_equal(original, imagem)
    np.testing.assert_array_equal(espelhada, imagem[..., ::-1])


def test_imagem_e_mascara_mantem_alinhamento():
    imagem = np.zeros((1, 40, 40), dtype=np.float32)
    mascara = np.zeros((1, 40, 40), dtype=np.float32)
    imagem[:, 10:20, 10:20] = 1
    mascara[:, 10:20, 10:20] = 1
    imagem_saida, mascara_saida = transformar_par(imagem, mascara, 20, True)
    assert imagem_saida.shape == mascara_saida.shape
    assert set(np.unique(mascara_saida)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(imagem_saida > 0.5, mascara_saida > 0.5)


class DatasetBaseFalso:
    def __init__(self, codigo_particao):
        self.pacientes = [SimpleNamespace(particao_codigo=codigo_particao) for _ in range(2)]

    def __len__(self):
        return len(self.pacientes)

    def __getitem__(self, indice):
        return {
            "imagem": torch.ones((1, 16, 16)),
            "mascara": torch.ones((1, 16, 16)),
            "rotulo": torch.tensor(1.0),
            "paciente": f"P{indice}",
            "corte_z": 4,
        }


@pytest.mark.parametrize(("particao", "codigo"), [("treino", 0), ("validacao", 2)])
def test_aumenta_treino_e_validacao(particao, codigo):
    dataset = DatasetDCEAumentado(DatasetBaseFalso(codigo), particao)
    assert len(dataset) == 20
    amostra = dataset[0]
    assert amostra["paciente"] == "P0"
    assert amostra["rotulo"].item() == 1
    assert amostra["imagem"].shape == (1, 16, 16)
    assert amostra["mascara"].shape == (1, 16, 16)
    assert "augmentacao" in amostra


def test_bloqueia_teste_e_particao_incorreta():
    with pytest.raises(ValueError, match="somente"):
        DatasetDCEAumentado(DatasetBaseFalso(1), "teste")
    with pytest.raises(ValueError, match="exclusivamente"):
        DatasetDCEAumentado(DatasetBaseFalso(1), "treino")
    with pytest.raises(ValueError, match="exclusivamente"):
        DatasetDCEAumentado(DatasetBaseFalso(0), "validacao")


def test_peso_calculado_apenas_com_treino():
    treino = [SimpleNamespace(pcr=0, particao_codigo=0) for _ in range(6)]
    treino += [SimpleNamespace(pcr=1, particao_codigo=0) for _ in range(2)]
    assert contar_classes(treino) == {0: 6, 1: 2}
    assert calcular_pos_weight_treino(treino).item() == 3
    misto = treino + [SimpleNamespace(pcr=1, particao_codigo=1)]
    with pytest.raises(ValueError, match="somente"):
        calcular_pos_weight_treino(misto)


def test_dataset_vazio_e_rejeitado():
    dataset = DatasetBaseFalso(0)
    dataset.pacientes = []
    with pytest.raises(ValueError, match="precisa conter pacientes"):
        DatasetDCEAumentado(dataset, "treino")
