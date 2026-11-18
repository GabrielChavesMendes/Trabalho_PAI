"""Estudo suplementar de limiares, sem treino ou acesso a imagens de teste.

Conserva todos os artefatos de origem. Novos checkpoints recebem os mesmos
pesos e a decisão de limiar tomada exclusivamente nos 33 pacientes de validação.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.modulo_treinamento import (
    RegistroPredicao, salvar_matriz_confusao,
    salvar_resultado_json, selecionar_limiar_validacao,
)


def candidatos_por_escores(probabilidades):
    """Representa todas as separações entre escores distintos, sem usar rótulos."""
    valores = sorted(set(float(p) for p in probabilidades))
    if not valores or any(not math.isfinite(p) or not 0 <= p <= 1 for p in valores):
        raise ValueError('Probabilidades devem ser finitas, não vazias e estar em [0,1]')
    candidatos = [0.5, valores[0] / 2, (1 + valores[-1]) / 2]
    candidatos.extend(a + (b - a) / 2 for a, b in zip(valores, valores[1:]))
    return sorted(set(c for c in candidatos if 0 < c < 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem-resultados', type=Path, required=True)
    parser.add_argument('--origem-checkpoints', type=Path, required=True)
    parser.add_argument('--saida', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for destino in (args.saida, args.checkpoints):
        if destino.exists():
            raise FileExistsError(f'Escolha uma pasta nova: {destino}')
        for origem in (args.origem_resultados, args.origem_checkpoints):
            if destino.resolve().is_relative_to(origem.resolve()) or origem.resolve().is_relative_to(destino.resolve()):
                raise ValueError('Origem e destino devem ser pastas independentes')
    nomes = [f"{a}__{f.replace('_','-')}__{c.replace('_','-')}__seed-{args.seed}"
             for c in ('central', 'maior_tumor') for f in ('post_early', 'tres')
             for a in ('cnn', 'mobilenetv2', 'resnet50')]
    tabela = pd.read_csv(RAIZ / 'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    ids = set(tabela.loc[tabela.test.eq(2), 'pid'])
    rotulos = dict(zip(tabela.loc[tabela.test.eq(2), 'pid'], tabela.loc[tabela.test.eq(2), 'pCR']))
    preparados = []
    for nome in nomes:
        origem = args.origem_resultados / (nome + '__validation.json')
        dados = json.loads(origem.read_text(encoding='utf-8'))
        ps = dados['predicoes_por_paciente']
        if len(ps) != 33 or {p['paciente'] for p in ps} != ids:
            raise ValueError(f'IDs não correspondem à validação: {nome}')
        if any(p['rotulo'] != rotulos[p['paciente']] for p in ps):
            raise ValueError(f'Rótulos divergentes do CSV de validação: {nome}')
        registros = [RegistroPredicao(p['paciente'], p['rotulo'], p['probabilidade']) for p in ps]
        candidatos = candidatos_por_escores(p.probabilidade for p in registros)
        limiar, metricas = selecionar_limiar_validacao(
            registros, candidatos=candidatos, criterio='acuracia_balanceada')
        if not math.isclose(metricas['auc_roc'], dados['metricas']['auc_roc'], abs_tol=1e-12):
            raise ValueError('AUC não pode mudar por alteração de limiar')
        for pasta, sufixo in ((args.origem_checkpoints, '.pt'),
                              (args.origem_resultados, '__history.csv'),
                              (args.origem_resultados, '__loss.png')):
            if not (pasta / (nome + sufixo)).is_file():
                raise FileNotFoundError(pasta / (nome + sufixo))
        preparados.append((nome, origem, dados, registros, candidatos, limiar, metricas))
    args.saida.mkdir(parents=True)
    args.checkpoints.mkdir(parents=True)
    linhas = []
    for nome, origem, dados, registros, candidatos, limiar, metricas in preparados:
        cp_origem = args.origem_checkpoints / (nome + '.pt')
        cp_destino = args.checkpoints / (nome + '.pt')
        metricas = {**metricas, 'loss': dados['metricas']['loss']}
        anterior = torch.load(cp_origem, map_location='cpu', weights_only=False)
        if anterior.get('formato_checkpoint') != 1:
            raise ValueError(f'Formato de checkpoint inesperado: {cp_origem}')
        novo = dict(anterior)
        novo['selecao_validacao'] = {
            'limiar': float(limiar),
            'metricas': {**metricas, 'criterio_selecao_limiar': 'acuracia_balanceada'},
            'metodo_busca': 'pontos medios entre probabilidades unicas de validacao; extremos internos e 0.5',
            'numero_candidatos': len(candidatos),
            'exploratorio_apos_inspecao_validacao': True,
            'origem_checkpoint_sha256': hashlib.sha256(cp_origem.read_bytes()).hexdigest(),
        }
        torch.save(novo, cp_destino)
        del novo
        novo = torch.load(cp_destino, map_location='cpu', weights_only=False)
        pesos_iguais = (anterior['model_state_dict'].keys() == novo['model_state_dict'].keys()
            and all(torch.equal(v, novo['model_state_dict'][k]) for k, v in anterior['model_state_dict'].items()))
        if not pesos_iguais:
            raise ValueError(f'Pesos divergentes: {nome}')
        del anterior, novo
        destino = args.saida / (nome + '__validation.json')
        salvar_resultado_json(destino, metricas, registros)
        exportado = json.loads(destino.read_text(encoding='utf-8'))
        exportado['estudo_limiar'] = {
            'metodo': 'pontos medios entre probabilidades unicas de validacao; extremos internos e 0.5',
            'numero_candidatos': len(candidatos), 'criterio': 'acuracia_balanceada',
            'desempate': 'especificidade, sensibilidade, proximidade de 0.5',
            'prob_min': min(p.probabilidade for p in registros),
            'prob_max': max(p.probabilidade for p in registros),
            'origem_validacao': str(origem),
            'origem_checkpoint': str(cp_origem),
            'origem_checkpoint_sha256': hashlib.sha256(cp_origem.read_bytes()).hexdigest(),
            'pesos_iguais_tensor_a_tensor': pesos_iguais,
            'exploratorio_apos_inspecao_validacao': True, 'teste_utilizado': False,
        }
        destino.write_text(json.dumps(exportado, ensure_ascii=False, indent=2), encoding='utf-8')
        salvar_matriz_confusao(args.saida / (nome + '__validation__confusion-matrix.png'), metricas)
        for sufixo in ('__history.csv', '__loss.png'):
            shutil.copy2(args.origem_resultados / (nome + sufixo), args.saida / (nome + sufixo))
        linha = {'experimento': nome, 'auc_roc': metricas['auc_roc'], 'candidatos': len(candidatos)}
        for k in ('limiar', 'acuracia_balanceada', 'f1', 'sensibilidade', 'especificidade', 'tn', 'fp', 'fn', 'tp'):
            linha[k + '_original'] = dados['metricas'][k]
            linha[k + '_refinado'] = metricas[k]
        linha['pesos_iguais_tensor_a_tensor'] = pesos_iguais
        linhas.append(linha)
        print('Limiar de validação registrado; pesos preservados:', nome, flush=True)
    with (args.saida / 'comparacao_limiares.csv').open('x', encoding='utf-8', newline='') as arq:
        writer = csv.DictWriter(arq, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


if __name__ == '__main__':
    main()
