"""Avaliação final por manifesto congelado; requer revisão explícita antes de executar."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.rodada_final_issue4 import carregar_modelo_registrado, criar_carregador
from modulos.modulo_treinamento import (
    avaliar_modelo, escolher_dispositivo, configurar_reprodutibilidade,
    salvar_resultado_json, salvar_matriz_confusao,
)


def resolver(caminho):
    p = Path(caminho)
    return p if p.is_absolute() else RAIZ / p


def validar_manifesto(manifesto):
    if manifesto.get('teste_usado_para_selecao') is not False:
        raise ValueError('Manifesto deve declarar seleção exclusivamente por validação')
    entradas = manifesto['modelos']
    ids = [m['id'] for m in entradas]
    if len(set(ids)) != len(ids):
        raise ValueError('IDs de modelos duplicados')
    originais = [m for m in entradas if m['grupo']=='originais']
    novos = [m for m in entradas if m['grupo']=='novos']
    esperadas = {(a,f,c) for a in ('cnn','resnet50','mobilenetv2')
                 for f in ('post_early','tres') for c in ('central','maior_tumor')}
    obtidas = {(m['configuracao']['arquitetura'],m['configuracao']['configuracao_temporal'],
                m['configuracao']['estrategia_corte']) for m in originais}
    if len(originais)!=12 or obtidas!=esperadas or len(novos)>3 or len(entradas)!=12+len(novos):
        raise ValueError('Necessárias 12 configurações originais e no máximo três novos candidatos')
    for m in entradas:
        cp = resolver(m['checkpoint'])
        if hashlib.sha256(cp.read_bytes()).hexdigest()!=m['checkpoint_sha256']:
            raise ValueError(f'Checkpoint alterado após congelamento: {cp}')
        validacao = resolver(m['validacao'])
        if hashlib.sha256(validacao.read_bytes()).hexdigest()!=m['validacao_sha256']:
            raise ValueError(f'Validação alterada após congelamento: {validacao}')
        modelo, bruto, norm = carregar_modelo_registrado(cp)
        if bruto['configuracao']!=m['configuracao'] or bruto['preprocessamento']!=m['preprocessamento']:
            raise ValueError('Configuração/preprocessamento não coincide com o manifesto')
        if bruto['epoca']!=m['epoca'] or bruto['selecao_validacao']['limiar']!=m['limiar']:
            raise ValueError('Época/limiar divergem do congelamento')
        del modelo, bruto
    csv = resolver('dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    if hashlib.sha256(csv.read_bytes()).hexdigest()!=manifesto['csv_sha256']:
        raise ValueError('CSV alterado desde o congelamento')
    return entradas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifesto',type=Path,required=True)
    parser.add_argument('--saida',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--confirmar-validacao-fechada',action='store_true')
    args = parser.parse_args()
    if not args.confirmar_validacao_fechada:
        parser.error('Teste bloqueado: revisar resultados e só depois confirmar validação fechada.')
    if args.saida.exists():
        raise FileExistsError('Saída existente: não repetir/sobrescrever avaliação final.')
    manifesto = json.loads(args.manifesto.read_text(encoding='utf-8'))
    tarefas = validar_manifesto(manifesto)  # todos os pesos recarregados ANTES de abrir teste
    args.saida.mkdir(parents=True)
    (args.saida/'manifesto_utilizado.json').write_text(json.dumps(manifesto,ensure_ascii=False,indent=2),encoding='utf-8')
    device = escolher_dispositivo(args.device)
    linhas = []
    for i, m in enumerate(tarefas,1):
        configurar_reprodutibilidade(int(m['configuracao']['seed']))
        modelo, bruto, norm = carregar_modelo_registrado(resolver(m['checkpoint']),device)
        loader = criar_carregador('teste',bruto['configuracao'],norm,permitir_teste=True)
        criterio = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([m['configuracao']['pos_weight']],device=device))
        inicio = time.perf_counter()
        loss, metricas, ps = avaliar_modelo(modelo,loader,criterio,device,m['limiar'])
        if device.type=='cuda': torch.cuda.synchronize(device)
        if len(ps)!=35:
            raise ValueError('Teste deve conter exatamente 35 pacientes, sem aumento')
        metricas = {**metricas,'loss':loss,'tempo_teste_segundos':time.perf_counter()-inicio}
        pasta = args.saida/m['grupo']
        pasta.mkdir(exist_ok=True)
        salvar_resultado_json(pasta/(m['id']+'__test.json'),metricas,ps)
        salvar_matriz_confusao(pasta/(m['id']+'__test__confusion-matrix.png'),metricas)
        with (pasta/(m['id']+'__test__examples.csv')).open('x',encoding='utf-8',newline='') as f:
            w = csv.DictWriter(f,fieldnames=('paciente','rotulo','probabilidade','previsto','correto'))
            w.writeheader()
            for p in ps:
                previsto = int(p.probabilidade>=m['limiar'])
                w.writerow(dict(paciente=p.paciente,rotulo=p.rotulo,probabilidade=p.probabilidade,
                                previsto=previsto,correto=previsto==p.rotulo))
        linhas.append({'grupo':m['grupo'],'modelo':m['id'],**metricas})
        print(f'[{i}/{len(tarefas)}] Avaliação registrada: {m["id"]}',flush=True)
        del modelo, bruto, loader
        if device.type=='cuda': torch.cuda.empty_cache()
    for grupo in ('originais','novos'):
        grupo_linhas = [l for l in linhas if l['grupo']==grupo]
        if grupo_linhas:
            with (args.saida/f'comparacao_teste_{grupo}.csv').open('x',encoding='utf-8',newline='') as f:
                w = csv.DictWriter(f,fieldnames=list(grupo_linhas[0]))
                w.writeheader(); w.writerows(grupo_linhas)
    print('Teste final concluído. Seleção e hiperparâmetros continuam os definidos na validação.')


if __name__ == '__main__':
    main()
