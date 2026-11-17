"""Regressoes para ajustes de hiperparametros e avaliacao final da Issue #4."""
import pytest

from modulos.modulo_treinamento import (
    RegistroPredicao, calcular_metricas, selecionar_limiar_validacao,
)
from scripts.avaliar_grade_teste_issue4 import configuracoes
from scripts.ajustar_issue4 import CANDIDATOS


def test_auc_perfeita_e_balanced_accuracy():
    registros = [
        RegistroPredicao("P1", 1, 0.9),
        RegistroPredicao("P2", 1, 0.8),
        RegistroPredicao("N1", 0, 0.2),
        RegistroPredicao("N2", 0, 0.1),
    ]
    met = calcular_metricas(registros, 0.5)
    assert met["auc_roc"] == pytest.approx(1.0)
    assert met["acuracia_balanceada"] == pytest.approx(1.0)
    assert met["f1"] == pytest.approx(1.0)


def test_metricas_mostraram_colapso_em_uma_classe():
    registros = [
        RegistroPredicao("P1", 1, 0.6),
        RegistroPredicao("P2", 1, 0.7),
        RegistroPredicao("N1", 0, 0.8),
        RegistroPredicao("N2", 0, 0.9),
    ]
    met = calcular_metricas(registros, 0.5)
    assert met["sensibilidade"] == pytest.approx(1.0)
    assert met["especificidade"] == pytest.approx(0.0)
    assert met["acuracia_balanceada"] == pytest.approx(0.5)
    assert met["auc_roc"] == pytest.approx(0.0)


def test_selecao_de_limiar_equilibra_as_classes():
    registros = [
        RegistroPredicao("P1", 1, 0.9),
        RegistroPredicao("P2", 1, 0.6),
        RegistroPredicao("N1", 0, 0.55),
        RegistroPredicao("N2", 0, 0.2),
    ]
    limiar, met = selecionar_limiar_validacao(
        registros, candidatos=[0.4, 0.6, 0.8], criterio="acuracia_balanceada"
    )
    assert limiar == pytest.approx(0.6)
    assert met["acuracia_balanceada"] == pytest.approx(1.0)


def test_grade_tem_12_configuracoes_unicas():
    nomes = [nome for _, _, _, nome in configuracoes(42)]
    assert len(nomes) == len(set(nomes)) == 12
    assert len(CANDIDATOS) == 4
