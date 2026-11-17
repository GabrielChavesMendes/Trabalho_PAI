"""Análises descritivas exclusivamente sobre predições de validação fornecidas."""
from __future__ import annotations

import numpy as np

from modulos.modulo_treinamento import agregar_por_paciente, calcular_metricas


def distribuicao_probabilidades(registros):
    ps = agregar_por_paciente(registros)
    valores = np.array([p.probabilidade for p in ps],dtype=float)
    if not len(valores) or not np.isfinite(valores).all():
        raise ValueError('Probabilidades ausentes ou não finitas')
    def resumo(x):
        return {'min':float(x.min()),'max':float(x.max()),'amplitude':float(np.ptp(x)),
                'media':float(x.mean()),'std':float(x.std()),
                'q25':float(np.quantile(x,.25)),'mediana':float(np.median(x)),
                'q75':float(np.quantile(x,.75)),'iqr':float(np.quantile(x,.75)-np.quantile(x,.25))}
    saida = {'todos':resumo(valores)}
    for classe in (0,1):
        x = np.array([p.probabilidade for p in ps if p.rotulo==classe])
        if len(x): saida[f'pcr_{classe}']=resumo(x)
    return saida


def estabilidade_limiar(registros, limiar):
    ps = agregar_por_paciente(registros)
    base = [int(p.probabilidade>=limiar) for p in ps]
    linhas = []
    for delta in (-.01,-.005,-.001,0.,.001,.005,.01):
        t = float(np.clip(limiar+delta,1e-12,1-1e-12))
        metricas = calcular_metricas(ps,t)
        linhas.append({'deslocamento':delta,**metricas,
                       'classes_alteradas':sum(int(p.probabilidade>=t)!=a for p,a in zip(ps,base))})
    return {'avaliacoes':linhas,
            'pior_acuracia_balanceada_mais_menos_001':min(
                d['acuracia_balanceada'] for d in linhas if abs(d['deslocamento'])==.01),
            'menor_distancia_ao_limiar':min(abs(p.probabilidade-limiar) for p in ps)}


def diferenca_auc_bootstrap(referencia, candidato, n=2000, seed=2026):
    a = agregar_por_paciente(referencia)
    b = agregar_por_paciente(candidato)
    if [(p.paciente,p.rotulo) for p in a]!=[(p.paciente,p.rotulo) for p in b]:
        raise ValueError('Bootstrap pareado exige os mesmos pacientes e rótulos')
    y = np.array([p.rotulo for p in a])
    pos, neg = np.flatnonzero(y==1),np.flatnonzero(y==0)
    if not len(pos) or not len(neg) or n<1:
        raise ValueError('Bootstrap requer duas classes e n positivo')
    pa,pb = np.array([p.probabilidade for p in a]),np.array([p.probabilidade for p in b])
    def auc(v, ip, inn):
        positivos,negativos = v[ip,None],v[inn][None,:]
        return np.mean(positivos>negativos)+.5*np.mean(positivos==negativos)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        ip = rng.choice(pos,size=len(pos),replace=True)
        inn = rng.choice(neg,size=len(neg),replace=True)
        deltas.append(auc(pb,ip,inn)-auc(pa,ip,inn))
    return {'delta_auc':float(auc(pb,pos,neg)-auc(pa,pos,neg)),
            'ic95_percentil_inferior':float(np.quantile(deltas,.025)),
            'ic95_percentil_superior':float(np.quantile(deltas,.975)),
            'fracao_reamostras_delta_positivo':float(np.mean(np.array(deltas)>0)),
            'reamostragens':n,'seed':seed,'estratificado':True,'pareado_por_paciente':True,
            'limite':'Descritivo/exploratório; época/modelo/limiar escolhidos na mesma validação.'}


def recomendar_candidato(metricas, estabilidade, criterios):
    testes = {
        'auc_superior':metricas['auc_roc']>criterios['auc_maior_que']+1e-12,
        'balanced_nao_inferior':metricas['acuracia_balanceada']>=criterios['balanced_min']-1e-12,
        'sensibilidade_nao_inferior':metricas['sensibilidade']>=criterios['sens_min']-1e-12,
        'ambas_classes':0<metricas['tp']+metricas['fp']<metricas['numero_pacientes'],
        'robustez_limiar':estabilidade['pior_acuracia_balanceada_mais_menos_001']>
                         criterios['pior_balanced_mais_menos_001_maior_que']+1e-12,
    }
    return {'recomendado':all(testes.values()),'criterios':testes}
