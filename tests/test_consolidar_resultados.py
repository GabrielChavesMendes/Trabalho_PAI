"""A epoca divulgada deve corresponder aos pesos realmente selecionados."""
import csv
import json
import sys

import torch

from scripts.consolidar_resultados import main


def test_consolidacao_usa_epoca_do_checkpoint_auc(tmp_path, monkeypatch):
    nome = 'cnn__post-early__central__seed-42'
    resultados = tmp_path / 'resultados'
    checkpoints = tmp_path / 'checkpoints'
    resultados.mkdir()
    checkpoints.mkdir()
    metricas = dict.fromkeys((
        'sensibilidade', 'especificidade', 'precisao', 'acuracia', 'f1',
        'acuracia_balanceada', 'auc_roc', 'loss', 'limiar',
    ), 0.5)
    metricas.update(auc_roc=0.8, loss=0.6, tn=2, fp=1, fn=1, tp=2, numero_pacientes=6)
    (resultados / f'{nome}__validation.json').write_text(
        json.dumps({'metricas': metricas}), encoding='utf-8'
    )
    with (resultados / f'{nome}__history.csv').open('w', newline='', encoding='utf-8') as arq:
        writer = csv.DictWriter(arq, fieldnames=('epoca', 'loss_validacao', 'tempo_segundos'))
        writer.writeheader()
        writer.writerows([
            {'epoca': 1, 'loss_validacao': 0.3, 'tempo_segundos': 2},
            {'epoca': 2, 'loss_validacao': 0.6, 'tempo_segundos': 3},
        ])
    torch.save({'epoca': 2, 'configuracao': {'criterio_checkpoint': 'auc_roc'}},
               checkpoints / f'{nome}.pt')
    saida = resultados / 'comparacao_validacao.csv'
    monkeypatch.setattr(sys, 'argv', [
        'consolidar_resultados.py', '--resultados', str(resultados),
        '--checkpoints', str(checkpoints), '--saida', str(saida), '--criterio', 'auc_roc',
    ])
    assert main() == 0
    with saida.open(encoding='utf-8', newline='') as arq:
        linha = next(csv.DictReader(arq))
    assert linha['melhor_epoca'] == '2'
    assert linha['epoca_menor_loss_validacao'] == '1'
    assert linha['criterio_checkpoint'] == 'auc_roc'
    manifesto = json.loads((resultados / 'modelo_selecionado.json').read_text(encoding='utf-8'))
    assert manifesto['configuracao']['melhor_epoca'] == 2
    assert manifesto['configuracao']['criterio_checkpoint'] == 'auc_roc'
