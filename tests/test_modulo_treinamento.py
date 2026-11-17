"""Testes da Issue #4 sem depender de NIfTI nem do modulo da Issue #2."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from modulos.modulo_redes import CNNSimples, configurar_fine_tuning, criar_modelo
from modulos import fabrica_carregadores
from modulos.fabrica_carregadores import DatasetMapasRealce, DatasetRecorteROI
from modulos.modulo_treinamento import (
    ConfiguracaoTreinamento,
    RegistroPredicao,
    agregar_por_paciente,
    calcular_auc_roc,
    calcular_pos_weight_treino,
    calcular_metricas,
    carregar_checkpoint,
    configurar_reprodutibilidade,
    registrar_validacao_checkpoint,
    selecionar_limiar_validacao,
    treinar_experimento,
)


class DatasetSintetico(Dataset):
    def __init__(self, quantidade: int = 8):
        gerador = torch.Generator().manual_seed(7)
        self.imagens = torch.rand(quantidade, 1, 32, 32, generator=gerador)
        self.rotulos = torch.tensor([i % 2 for i in range(quantidade)]).float()

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, indice):
        return self.imagens[indice], self.rotulos[indice], f"P{indice:03d}"


def test_metricas_sao_calculadas_por_paciente():
    registros = [
        RegistroPredicao("P1", 1, 0.8),
        RegistroPredicao("P1", 1, 0.6),
        RegistroPredicao("P2", 0, 0.1),
        RegistroPredicao("P3", 1, 0.2),
        RegistroPredicao("P4", 0, 0.9),
    ]
    agregados = agregar_por_paciente(registros)
    assert len(agregados) == 4
    assert next(r for r in agregados if r.paciente == "P1").probabilidade == pytest.approx(0.7)

    metricas = calcular_metricas(registros, limiar=0.5)
    assert metricas["tp"] == 1
    assert metricas["tn"] == 1
    assert metricas["fp"] == 1
    assert metricas["fn"] == 1
    assert metricas["acuracia"] == pytest.approx(0.5)
    assert metricas["f1"] == pytest.approx(0.5)


def test_pos_weight_conta_pacientes_e_nao_variacoes():
    lotes = [
        (
            torch.rand(4, 1, 8, 8),
            torch.tensor([0, 0, 0, 1]),
            ["N1", "N1", "N2", "P1"],
        )
    ]
    peso, negativos, positivos = calcular_pos_weight_treino(lotes)
    assert (peso, negativos, positivos) == (2.0, 2, 1)


def test_fabrica_integra_particoes_sem_carregar_volumes(tmp_path, monkeypatch):
    monkeypatch.setenv("PAI_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("PAI_NUM_WORKERS", "0")
    monkeypatch.setattr(fabrica_carregadores, "verificar_arquivos", lambda manifesto: [])
    carregadores = fabrica_carregadores.criar_carregadores(
        configuracao_temporal="post_early",
        estrategia_corte="central",
        tamanho_lote=8,
        seed=42,
    )
    assert len(carregadores["treino"].dataset) == 1040
    assert len(carregadores["validacao"].dataset) == 330
    assert len(carregadores["teste"].dataset) == 35
    assert carregadores["pos_weight"] == pytest.approx(3.0)
    assert carregadores["metadados"]["fases"] == ["post_early"]
    assert carregadores["metadados"]["particoes"] == {
        "treino": 0,
        "validacao": 2,
        "teste": 1,
    }


def test_rejeita_rotulos_conflitantes_do_mesmo_paciente():
    with pytest.raises(ValueError, match="conflitantes"):
        agregar_por_paciente(
            [RegistroPredicao("P1", 0, 0.2), RegistroPredicao("P1", 1, 0.8)]
        )


def test_limiar_e_selecionado_apenas_das_predicoes_fornecidas():
    registros = [
        RegistroPredicao("P1", 1, 0.6),
        RegistroPredicao("P2", 1, 0.7),
        RegistroPredicao("P3", 0, 0.1),
        RegistroPredicao("P4", 0, 0.4),
    ]
    limiar, metricas = selecionar_limiar_validacao(registros, [0.25, 0.5, 0.75])
    assert limiar == 0.5
    assert metricas["f1"] == pytest.approx(1.0)


def test_auc_e_acuracia_balanceada_nao_premiam_classe_unica():
    registros = [
        RegistroPredicao("P1", 1, 0.8),
        RegistroPredicao("P2", 1, 0.4),
        RegistroPredicao("N1", 0, 0.6),
        RegistroPredicao("N2", 0, 0.1),
    ]
    assert calcular_auc_roc(registros) == pytest.approx(0.75)
    limiar, metricas = selecionar_limiar_validacao(
        registros, [0.05, 0.35, 0.7], criterio="acuracia_balanceada"
    )
    assert limiar == pytest.approx(0.35)
    assert metricas["acuracia_balanceada"] == pytest.approx(0.75)
    assert metricas["auc_roc"] == pytest.approx(0.75)


def test_fine_tuning_resnet_libera_layer4_e_preserva_batchnorm():
    modelo = criar_modelo("resnet50", 3, pre_treinado=False)
    resumo = configurar_fine_tuning(modelo, "resnet50", "layer4")
    assert resumo["parametros_treinaveis"] > modelo.fc.weight.numel()
    assert modelo.layer4[-1].conv3.weight.requires_grad
    assert modelo.fc.weight.requires_grad
    assert not modelo.layer4[-1].bn3.weight.requires_grad
    assert not modelo.layer3[-1].conv3.weight.requires_grad


def test_seed_antes_do_modelo_reproduz_inicializacao_da_cabeca():
    configurar_reprodutibilidade(42)
    primeiro = criar_modelo("resnet50", 3, pre_treinado=False)
    configurar_reprodutibilidade(42)
    segundo = criar_modelo("resnet50", 3, pre_treinado=False)
    assert torch.equal(primeiro.fc.weight, segundo.fc.weight)
    assert torch.equal(primeiro.fc.bias, segundo.fc.bias)


def test_recorte_roi_preserva_formato_e_amplia_tumor():
    class DatasetROI(Dataset):
        pacientes = [object()]

        def __len__(self):
            return 1

        def __getitem__(self, indice):
            imagem = torch.zeros(1, 256, 256)
            mascara = torch.zeros(1, 256, 256)
            imagem[:, 100:120, 130:150] = 1
            mascara[:, 100:120, 130:150] = 1
            return {"imagem": imagem, "mascara": mascara}

    original = DatasetROI()[0]
    recortado = DatasetRecorteROI(DatasetROI())[0]
    assert recortado["imagem"].shape == original["imagem"].shape
    assert recortado["mascara"].shape == original["mascara"].shape
    assert recortado["mascara"].sum() > original["mascara"].sum()
    assert recortado["recorte_roi"]["lado"] == 64


def test_mapas_realce_explicitam_subtracoes_temporais():
    class DatasetTresFases(Dataset):
        pacientes = [object()]

        def __len__(self):
            return 1

        def __getitem__(self, indice):
            pre = torch.full((2, 2), 0.2)
            early = torch.full((2, 2), 0.6)
            late = torch.full((2, 2), 0.4)
            return {"imagem": torch.stack((pre, early, late))}

    imagem = DatasetMapasRealce(DatasetTresFases())[0]["imagem"]
    assert torch.allclose(imagem[0], torch.full((2, 2), 0.6))
    assert torch.allclose(imagem[1], torch.full((2, 2), 0.7))
    assert torch.allclose(imagem[2], torch.full((2, 2), 0.6))


def test_treino_salva_checkpoint_recarregavel(tmp_path: Path):
    carregador = DataLoader(DatasetSintetico(), batch_size=4, shuffle=False)
    modelo = CNNSimples(canais_entrada=1)
    configuracao = ConfiguracaoTreinamento(
        arquitetura="cnn",
        canais_entrada=1,
        configuracao_temporal="post_early",
        estrategia_corte="central",
        epocas=2,
        tamanho_lote=4,
        paciencia=2,
    )
    checkpoint = tmp_path / "cnn_post-early_central.pt"
    historico = treinar_experimento(
        modelo,
        carregador,
        carregador,
        configuracao,
        checkpoint,
        metadados_preprocessamento={
            "fases": ["post_early"],
            "normalizacao": "sintetica_apenas_para_teste",
            "corte": "central_sintetico",
            "augmentation": "nenhum_sintetico",
            "particoes": {"treino": 0, "validacao": 2, "teste": 1},
        },
        dispositivo="cpu",
    )

    assert len(historico) == 2
    assert checkpoint.is_file()
    recarregado = CNNSimples(canais_entrada=1)
    dados = carregar_checkpoint(checkpoint, recarregado)
    assert dados["configuracao"]["pos_weight"] == 3.0
    assert dados["preprocessamento"]["particoes"]["treino"] == 0
    assert dados["epoca"] in (1, 2)
    for nome, parametro in recarregado.state_dict().items():
        assert torch.equal(parametro, dados["model_state_dict"][nome])

    registrar_validacao_checkpoint(
        checkpoint,
        0.4,
        {"f1": 0.75, "limiar": 0.4, "numero_pacientes": 8},
    )
    atualizado = carregar_checkpoint(checkpoint, recarregado)
    assert atualizado["selecao_validacao"]["limiar"] == pytest.approx(0.4)
    assert atualizado["selecao_validacao"]["metricas"]["f1"] == pytest.approx(0.75)
