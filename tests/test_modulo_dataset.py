"""Testes iniciais da Issue #2: metadados e caminhos, sem baixar os volumes."""
from pathlib import Path

import pandas as pd
import pytest

from modulos.modulo_dataset import carregar_metadados, criar_manifesto, verificar_arquivos

CSV = Path(__file__).resolve().parents[1] / "dados" / "BreastDCEDL_SPY1_min_crop_metadata.csv"


def test_csv_oficial():
    tabela = carregar_metadados(CSV)
    assert len(tabela) == 172
    assert tabela["pCR"].value_counts().to_dict() == {0.0: 123, 1.0: 49}
    assert tabela["test"].value_counts().to_dict() == {0.0: 104, 1.0: 35, 2.0: 33}


def test_caminhos_de_fases_nao_consecutivas():
    tabela = carregar_metadados(CSV)
    manifesto = criar_manifesto(tabela, "dados/BreastDCEDL_ISPY1_min_crop")
    assert len(manifesto) == 172
    assert len({p.identificador for p in manifesto}) == 172
    caso = next(p for p in manifesto if p.identificador == "ISPY1_1001")
    assert [p.name for p in caso.caminhos_dce] == [
        f"ISPY1_1001_spy1_vis1_dce_aqc_{i}.nii.gz" for i in (0, 1, 2)
    ]
    assert caso.caminho_mascara.name == "ISPY1_1001_spy1_vis1_mask.nii.gz"
    nao_consecutivo = tabela[tabela.post_late != 2].iloc[0]
    paciente = next(p for p in manifesto if p.identificador == nao_consecutivo.pid)
    assert paciente.caminhos_dce[2].name.endswith(
        f"_dce_aqc_{int(nao_consecutivo.post_late)}.nii.gz"
    )


def test_verificacao_ausentes(tmp_path):
    tabela = carregar_metadados(CSV)
    manifesto = criar_manifesto(tabela.iloc[:1], tmp_path)
    erros = verificar_arquivos(manifesto)
    assert len(erros) == 4
    assert "arquivo pre ausente" in erros[0]


def test_rejeita_paciente_duplicado(tmp_path):
    tabela = pd.read_csv(CSV)
    tabela.loc[1, "pid"] = tabela.loc[0, "pid"]
    caminho = tmp_path / "duplicado.csv"
    tabela.to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="duplicados"):
        carregar_metadados(caminho)
