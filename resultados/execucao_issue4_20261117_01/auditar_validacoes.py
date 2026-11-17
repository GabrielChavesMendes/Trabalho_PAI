"""Auditoria desta execução: somente JSONs de validação e recarga em CPU.

Não cria carregadores, não abre NIfTI e não executa treinamento ou teste.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from modulos.modulo_redes import criar_modelo
from modulos.modulo_treinamento import RegistroPredicao, calcular_metricas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultados', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--quantidade', type=int, required=True)
    parser.add_argument('--saida', type=Path, required=True)
    args = parser.parse_args()
    if args.saida.exists() or args.saida.with_suffix('.csv').exists():
        raise FileExistsError('Auditoria já existe; escolha uma saída nova.')
    arquivos = sorted(args.resultados.rglob('*__validation.json'))
    if len(arquivos) != args.quantidade:
        raise ValueError(f'Esperados {args.quantidade} JSONs, encontrados {len(arquivos)}')
    tabela = pd.read_csv(RAIZ / 'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    ids = set(tabela.loc[tabela.test.eq(2), 'pid'])
    rotulos = dict(zip(tabela.loc[tabela.test.eq(2), 'pid'], tabela.loc[tabela.test.eq(2), 'pCR']))
    linhas = []
    for arquivo in arquivos:
        nome = arquivo.name.removesuffix('__validation.json')
        pasta_cp = args.checkpoints or arquivo.parent.parent / 'checkpoints'
        caminho = pasta_cp / (nome + '.pt')
        bruto = torch.load(caminho, map_location='cpu', weights_only=False)
        conf = bruto['configuracao']
        nome_config = (f"{conf['arquitetura']}__"
                       f"{conf['configuracao_temporal'].replace('_', '-')}__"
                       f"{conf['estrategia_corte'].replace('_', '-')}__seed-{conf['seed']}")
        if nome != nome_config:
            raise ValueError(f'Nome e configuração do checkpoint divergentes: {caminho}')
        if conf['canais_entrada'] != (3 if conf['configuracao_temporal'] == 'tres' else 1):
            raise ValueError(f'Número de canais divergente: {caminho}')
        modelo = criar_modelo(conf['arquitetura'], conf['canais_entrada'], pre_treinado=False)
        modelo.load_state_dict(bruto['model_state_dict'], strict=True)
        dados = json.loads(arquivo.read_text(encoding='utf-8'))
        predicoes = dados['predicoes_por_paciente']
        if len(predicoes) != 33 or {p['paciente'] for p in predicoes} != ids:
            raise ValueError(f'Pacientes divergentes da validação: {arquivo}')
        if any(p['rotulo'] != rotulos[p['paciente']] for p in predicoes):
            raise ValueError(f'Rótulos divergentes da validação: {arquivo}')
        registros = [RegistroPredicao(p['paciente'], p['rotulo'], p['probabilidade']) for p in predicoes]
        recalculadas = calcular_metricas(registros, dados['metricas']['limiar'])
        for metrica, valor in recalculadas.items():
            if not math.isclose(valor, dados['metricas'][metrica], abs_tol=1e-12):
                raise ValueError(f'Métrica divergente {metrica}: {arquivo}')
        selecao = bruto['selecao_validacao']
        if selecao['limiar'] != dados['metricas']['limiar']:
            raise ValueError(f'Limiar do checkpoint divergente: {caminho}')
        for metrica, valor in dados['metricas'].items():
            if not math.isclose(valor, selecao['metricas'][metrica], abs_tol=1e-12):
                raise ValueError(f'Métrica divergente no checkpoint {metrica}: {caminho}')
        with (arquivo.parent / (nome + '__history.csv')).open(encoding='utf-8', newline='') as arq:
            historico = list(csv.DictReader(arq))
        metadados = bruto['preprocessamento']
        if metadados['particoes'] != {'treino': 0, 'validacao': 2, 'teste': 1}:
            raise ValueError('Mapeamento de partições divergente')
        if metadados['augmentation']['teste_aumentado']:
            raise ValueError('Teste não deve ter augmentation')
        linhas.append({
            'experimento': nome,
            'candidato': arquivo.parent.parent.name if not args.checkpoints else nome,
            **conf,
            **dados['metricas'],
            'fracao_previstos_positivos': (dados['metricas']['tp'] + dados['metricas']['fp']) / 33,
            'epoca_escolhida': bruto['epoca'],
            'epocas_executadas': len(historico),
            'tempo_treinamento_segundos': sum(float(h['tempo_segundos']) for h in historico),
            'checkpoint': str(caminho),
            'checkpoint_sha256': hashlib.sha256(caminho.read_bytes()).hexdigest(),
            'validation_sha256': hashlib.sha256(arquivo.read_bytes()).hexdigest(),
            'recarga_cpu_estrita': True,
        })
        print('OK recarga sem treino e métricas de validação:', nome, flush=True)
        del modelo, bruto
    args.saida.write_text(json.dumps(linhas, ensure_ascii=False, indent=2), encoding='utf-8')
    with args.saida.with_suffix('.csv').open('x', encoding='utf-8', newline='') as arq:
        writer = csv.DictWriter(arq, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


if __name__ == '__main__':
    main()
