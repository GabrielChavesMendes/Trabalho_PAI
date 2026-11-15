"""Testes de partições e preparação de imagens da Issue #2."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from modulos.modulo_dataset import (
    DatasetDCE,
    carregar_amostra,
    carregar_metadados,
    criar_manifesto,
    normalizar_canais,
    selecionar_corte,
    selecionar_particao,
    verificar_arquivos,
)

CSV = Path(__file__).resolve().parents[1] / "dataset" / "BreastDCEDL_SPY1_min_crop_metadata.csv"


def test_csv_oficial_e_particoes():
    tabela = carregar_metadados(CSV)
    assert len(tabela) == 172
    assert tabela["pCR"].value_counts().to_dict() == {0.0: 123, 1.0: 49}
    assert tabela["test"].value_counts().to_dict() == {0.0: 104, 1.0: 35, 2.0: 33}
    manifesto = criar_manifesto(tabela, "dataset/BreastDCEDL_ISPY1_min_crop")
    treino = selecionar_particao(manifesto, "treino")
    teste = selecionar_particao(manifesto, "teste")
    validacao = selecionar_particao(manifesto, "validacao")
    assert (len(treino), len(teste), len(validacao)) == (104, 35, 33)
    conjuntos = [set(p.identificador for p in grupo) for grupo in (treino, teste, validacao)]
    assert not conjuntos[0] & conjuntos[1]
    assert not conjuntos[0] & conjuntos[2]
    assert not conjuntos[1] & conjuntos[2]
    with pytest.raises(ValueError, match="Partição inválida"):
        selecionar_particao(manifesto, "aleatoria")


def test_fases_nao_consecutivas():
    tabela = carregar_metadados(CSV)
    manifesto = criar_manifesto(tabela, "dataset/BreastDCEDL_ISPY1_min_crop")
    assert len({p.identificador for p in manifesto}) == 172
    caso = next(p for p in manifesto if p.identificador == "ISPY1_1001")
    assert [p.name for p in caso.caminhos_dce] == [
        f"ISPY1_1001_spy1_vis1_dce_aqc_{i}.nii.gz" for i in (0, 1, 2)
    ]
    assert caso.caminho_mascara.name == "ISPY1_1001_spy1_vis1_mask.nii.gz"
    nao_consecutivo = tabela[tabela.post_late != 2].iloc[0]
    paciente = next(p for p in manifesto if p.identificador == nao_consecutivo.pid)
    assert paciente.caminhos_dce[2].name.endswith(f"_dce_aqc_{int(nao_consecutivo.post_late)}.nii.gz")


def test_arquivos_ausentes(tmp_path):
    manifesto = criar_manifesto(carregar_metadados(CSV).iloc[:1], tmp_path)
    erros = verificar_arquivos(manifesto)
    assert len(erros) == 4
    assert "arquivo pre ausente" in erros[0]


def test_rejeita_pid_duplicado(tmp_path):
    tabela = pd.read_csv(CSV)
    tabela.loc[1, "pid"] = tabela.loc[0, "pid"]
    caminho = tmp_path / "duplicado.csv"
    tabela.to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="duplicados"):
        carregar_metadados(caminho)


def test_estrategias_de_corte():
    mascara = np.zeros((5, 256, 256), dtype=np.float32)
    mascara[1, :2, :2] = 1
    mascara[4, :5, :5] = 1
    assert selecionar_corte(mascara, "central") == 2
    assert selecionar_corte(mascara, "maior_tumor") == 4
    with pytest.raises(ValueError, match="não possui"):
        selecionar_corte(np.zeros_like(mascara), "maior_tumor")


def test_normalizacao_compartilhada():
    arr = np.stack((np.linspace(0, 10, 64).reshape(8, 8),
                    np.linspace(100, 110, 64).reshape(8, 8)), axis=0)
    result = normalizar_canais(arr)
    assert result.shape == (2, 8, 8)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    assert result[1].mean() > result[0].mean()
    assert not normalizar_canais(np.full((1, 8, 8), 7)).any()


def _criar_paciente_sintetico(tmp_path, monkeypatch, affine_incorreta=False):
    tabela = carregar_metadados(CSV).iloc[:1]
    p = criar_manifesto(tabela, tmp_path)[0]
    mask = np.zeros((5, 256, 256), dtype=np.float32)
    mask[1, :2, :2] = 1
    mask[4, 1:6, 1:6] = 1
    dados = [np.arange(5 * 256 * 256, dtype=np.float32).reshape(5, 256, 256) + i * 100 for i in range(3)]
    arquivos = {}
    for caminho, dado in zip(p.caminhos_dce, dados):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.touch()
        arquivos[str(caminho)] = (dado, np.eye(4))
    p.caminho_mascara.parent.mkdir(parents=True, exist_ok=True)
    p.caminho_mascara.touch()
    affine_mask = np.eye(4)
    if affine_incorreta:
        affine_mask[0, 3] = 5
    arquivos[str(p.caminho_mascara)] = (mask, affine_mask)
    class ImagemFake:
        def __init__(self, arr, affine):
            self.dataobj, self.affine = arr, affine
    modulo = SimpleNamespace(load=lambda caminho: ImagemFake(*arquivos[caminho]))
    monkeypatch.setitem(sys.modules, "nibabel", modulo)
    return p


def test_uma_fase_e_tres_fases_com_mascara_sintetica(tmp_path, monkeypatch):
    paciente = _criar_paciente_sintetico(tmp_path, monkeypatch)
    imagem, mask, z = carregar_amostra(paciente, "tres", "maior_tumor")
    assert tuple(imagem.shape) == (3, 256, 256)
    assert tuple(mask.shape) == (1, 256, 256)
    assert z == 4 and int(mask.sum()) == 25
    assert imagem.dtype == torch.float32
    imagem_um, _, z_central = carregar_amostra(paciente, "post_early", "central")
    assert tuple(imagem_um.shape) == (1, 256, 256)
    assert z_central == 2
    dataset = DatasetDCE([paciente], fase="tres", estrategia="maior_tumor")
    item = dataset[0]
    assert item["paciente"] == paciente.identificador
    assert tuple(item["imagem"].shape) == (3, 256, 256)
    assert item["rotulo"].item() == paciente.pcr


def test_geometria_incompativel_rejeitada(tmp_path, monkeypatch):
    paciente = _criar_paciente_sintetico(tmp_path, monkeypatch, affine_incorreta=True)
    with pytest.raises(ValueError, match="não estão alinhados"):
        carregar_amostra(paciente, "tres", "central")


def test_min_crop_rejeita_eixo_trocado(tmp_path, monkeypatch):
    """Regression: [Z,H,W] is valid; [H,W,Z] must fail explicitly."""
    paciente = _criar_paciente_sintetico(tmp_path, monkeypatch)
    import nibabel as nib
    original = nib.load

    def trocar_eixos(caminho):
        imagem = original(caminho)
        return SimpleNamespace(
            dataobj=np.moveaxis(imagem.dataobj, 0, 2),
            affine=imagem.affine,
        )

    monkeypatch.setattr(nib, "load", trocar_eixos)
    with pytest.raises(ValueError, match="Esperado volume MinCrop"):
        carregar_amostra(paciente, "tres", "central")
