import math
import sys

import pytest

from modulos.modulo_treinamento import RegistroPredicao, selecionar_limiar_validacao
from scripts.refinar_limiares_validacao_issue4 import candidatos_por_escores, main


def test_limiar_discrimina_escores_estreitos_que_grade_fixa_perde():
    registros = [RegistroPredicao('N1', 0, .498477), RegistroPredicao('N2', 0, .498478),
                 RegistroPredicao('P1', 1, .498480), RegistroPredicao('P2', 1, .498481)]
    _, grade_fixa = selecionar_limiar_validacao(registros, criterio='acuracia_balanceada')
    candidatos = candidatos_por_escores(p.probabilidade for p in registros)
    limiar, refinado = selecionar_limiar_validacao(registros, candidatos, 'acuracia_balanceada')
    assert grade_fixa['acuracia_balanceada'] == .5
    assert refinado['acuracia_balanceada'] == refinado['f1'] == 1
    assert .498478 < limiar < .498480
    assert refinado['auc_roc'] == grade_fixa['auc_roc'] == 1


def test_escores_saturados_e_repetidos_mantem_limiares_validos():
    candidatos = candidatos_por_escores([0, 0, 1, 1])
    assert candidatos
    assert all(math.isfinite(p) and 0 < p < 1 for p in candidatos)
    assert candidatos_por_escores([.5] * 33)


@pytest.mark.parametrize('escores', [[], [float('nan')], [float('inf')], [-.1], [1.1]])
def test_rejeita_escores_invalidos(escores):
    with pytest.raises(ValueError):
        candidatos_por_escores(escores)


def test_recusa_sobrescrever_experimentos(tmp_path, monkeypatch):
    origem = tmp_path / 'origem'
    origem.mkdir()
    protegido = origem / 'checkpoint_existente.pt'
    protegido.write_bytes(b'experimento-anterior')
    monkeypatch.setattr(sys, 'argv', [
        'refinar_limiares_validacao_issue4.py', '--origem-resultados', str(origem),
        '--origem-checkpoints', str(origem), '--saida', str(origem),
        '--checkpoints', str(tmp_path / 'novos_checkpoints'),
    ])
    with pytest.raises(FileExistsError):
        main()
    assert protegido.read_bytes() == b'experimento-anterior'
    assert not (tmp_path / 'novos_checkpoints').exists()
