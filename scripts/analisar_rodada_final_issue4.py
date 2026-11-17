"""Compara exclusivamente validação e prepara manifesto pendente de revisão."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from modulos.rodada_final_issue4 import carregar_modelo_registrado
from modulos.modulo_treinamento import RegistroPredicao, calcular_metricas, selecionar_limiar_validacao
from modulos.analise_validacao_issue4 import (
    distribuicao_probabilidades, estabilidade_limiar, diferenca_auc_bootstrap, recomendar_candidato,
)
from scripts.refinar_limiares_validacao_issue4 import candidatos_por_escores


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ler(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def registros(p):
    dados = ler(p)
    ps = [RegistroPredicao(d['paciente'],d['rotulo'],d['probabilidade']) for d in dados['predicoes_por_paciente']]
    tabela = pd.read_csv(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    verdade = dict(zip(tabela.loc[tabela.test.eq(2),'pid'],tabela.loc[tabela.test.eq(2),'pCR']))
    if len(ps)!=33 or {p.paciente:p.rotulo for p in ps}!=verdade:
        raise ValueError(f'IDs/rótulos não correspondem à validação: {p}')
    return ps


def entrada_manifesto(id, grupo, cp, validacao):
    modelo, bruto, norm = carregar_modelo_registrado(cp)
    del modelo
    return {'id':id,'grupo':grupo,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),
            'validacao':str(validacao),'validacao_sha256':sha(validacao),
            'configuracao':bruto['configuracao'],'preprocessamento':bruto['preprocessamento'],
            'epoca':bruto['epoca'],'limiar':bruto['selecao_validacao']['limiar'],
            'normalizacao_adicional':norm,'recarga_estrita_cpu':True}


def analisar(id, tipo, cp, validacao, historico):
    ps = registros(validacao)
    m = ler(validacao)['metricas']
    recalculadas = calcular_metricas(ps,m['limiar'])
    for k,v in recalculadas.items():
        if not np.isclose(v,m[k],rtol=0,atol=1e-12):
            raise ValueError(f'Métrica divergente: {id}/{k}')
    entrada = entrada_manifesto(id,tipo,cp,validacao)
    modelo, bruto, _ = carregar_modelo_registrado(cp)
    del modelo
    if entrada['limiar']!=m['limiar']:
        raise ValueError(f'Limiar não corresponde ao checkpoint: {id}')
    hist = pd.read_csv(historico)
    conf = bruto['configuracao']
    return {'id':id,'tipo':tipo,**m,'epoca':entrada['epoca'],'epocas_executadas':len(hist),
            'tempo_treinamento_segundos':float(hist.tempo_segundos.sum()),
            'configuracao':conf,'preprocessamento':bruto['preprocessamento'],
            'loss_treino_ultima':float(hist.iloc[-1].loss_treino),
            'loss_validacao_ultima':float(hist.iloc[-1].loss_validacao),
            'distribuicao':distribuicao_probabilidades(ps),
            'estabilidade':estabilidade_limiar(ps,m['limiar']),
            'entrada_manifesto':entrada,'checkpoint':str(cp),'validacao':str(validacao),
            'historico':str(historico)},ps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    args = parser.parse_args()
    base = args.base
    destino = base/'comparacao_validacao.json'
    if destino.exists() or (base/'manifesto_teste_pendente.json').exists():
        raise FileExistsError('Análise existente: escolher nova saída, não sobrescrever')
    plano = ler(base/'plano_antes_treinamento.json')
    antiga = RAIZ/'resultados/execucao_issue4_20261117_01'
    auditoria = ler(antiga/'auditoria_limiares.json')
    linhas, por_id, entradas = [],{},[]
    for d in auditoria:
        nome = d['experimento']
        cp = Path(d['checkpoint'])
        validacao = antiga/'grade_limiares_validacao'/(nome+'__validation.json')
        hist = antiga/'grade_limiares_validacao'/(nome+'__history.csv')
        linha,ps = analisar(nome,'originais',cp,validacao,hist)
        # Preservar/validar também o checkpoint originalmente treinado, anterior ao refinamento.
        original = RAIZ/'checkpoints/execucao_issue4_20261117_01/grade'/(nome+'.pt')
        m1,b1,_ = carregar_modelo_registrado(original)
        m2,b2,_ = carregar_modelo_registrado(cp)
        if not all(torch.equal(v,b2['model_state_dict'][k]) for k,v in b1['model_state_dict'].items()):
            raise ValueError('Pesos originais e cópia com limiar não coincidem')
        del m1,m2,b1,b2
        linha['entrada_manifesto']['checkpoint_original'] = str(original)
        linha['entrada_manifesto']['checkpoint_original_sha256'] = sha(original)
        linhas.append(linha); por_id[nome]=ps; entradas.append(linha['entrada_manifesto'])
    selecionado = ler(antiga/'grade_limiares_validacao/modelo_selecionado.json')['experimento']
    referencia = por_id[selecionado]
    novos = []
    for caso in plano['casos']:
        d = ler(base/caso['id']/'concluido.json')
        linha,ps = analisar(caso['id'],'novos',Path(d['checkpoint']),Path(d['validacao']),Path(d['historico']))
        linha['metricas_treino'] = d['metricas_treino']
        linha['bootstrap_auc_vs_melhor_anterior'] = diferenca_auc_bootstrap(
            referencia,ps,n=plano['bootstrap']['n'],seed=plano['bootstrap']['seed'])
        linha['decisao'] = recomendar_candidato(linha,linha['estabilidade'],plano['criterios_recomendacao'])
        linhas.append(linha); por_id[caso['id']]=ps; novos.append(linha)
        if linha['decisao']['recomendado']: entradas.append(linha['entrada_manifesto'])
    # Controle pareado de e3: mesma ResNet50 congelada, fases/corte/LR/peso.
    arquivo_frozen = antiga/'ajustes/peso_3_congelada/resultados/resnet50__tres__maior-tumor__seed-42__validation.json'
    frozen_ps = registros(arquivo_frozen)
    t, fm = selecionar_limiar_validacao(frozen_ps,candidatos_por_escores(p.probabilidade for p in frozen_ps),
                                       'acuracia_balanceada')
    controle = {'origem':str(arquivo_frozen),'metricas_limiar_comum':fm,
                'auc_original':ler(arquivo_frozen)['metricas']['auc_roc'],
                'loss_original':ler(arquivo_frozen)['metricas']['loss'],
                'bootstrap_e3_vs_congelada':diferenca_auc_bootstrap(
                    frozen_ps,por_id['e3_resnet_imagenet'],n=2000,seed=2026)}
    linhas_csv = []
    campos = ['auc_roc','acuracia_balanceada','sensibilidade','especificidade','precisao','f1','acuracia','loss',
              'limiar','tn','fp','fn','tp','epoca','epocas_executadas','tempo_treinamento_segundos']
    for d in linhas:
        x = {'id':d['id'],'tipo':d['tipo'],**{k:d[k] for k in campos}}
        x.update({f'prob_{k}':v for k,v in d['distribuicao']['todos'].items()})
        x['pior_balanced_limiar_mais_menos_001'] = d['estabilidade']['pior_acuracia_balanceada_mais_menos_001']
        x['recomendado_novo'] = d.get('decisao',{}).get('recomendado',False)
        linhas_csv.append(x)
    pd.DataFrame(linhas_csv).to_csv(base/'comparacao_validacao.csv',index=False)
    resultado = {'referencia_principal':selecionado,'modelos':linhas,'controle_resnet_congelada':controle,
                 'novos_recomendados':[d['id'] for d in novos if d['decisao']['recomendado']],
                 'teste_avaliado':False,'limite':'Resultados exploratórios na mesma validação reutilizada.'}
    destino.write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf-8')
    manifesto = {'estado':'aguardando_revisao_do_usuario','teste_usado_para_selecao':False,
                 'revisao_obrigatoria_antes_do_teste':True,
                 'csv_sha256':sha(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv'),
                 'modelos':entradas,'modelo_escolhido_pela_validacao':selecionado,
                 'novos_recomendados':resultado['novos_recomendados'],
                 'protocolo_teste':{'pacientes':35,'codigo_particao':1,'aumento':False,
                                    'limiar':'fixado previamente na validacao','selecionar_vencedor_no_teste':False},
                 'observacao':'12 pesos originais com os limiares já refinados; versões brutas também preservadas. '
                              'Novos entram somente se aprovados pelos critérios prévios de validação.'}
    (base/'manifesto_teste_pendente.json').write_text(json.dumps(manifesto,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Modelos comparados:',len(linhas),'Novos recomendados:',resultado['novos_recomendados'])
    for d in novos:
        print(d['id'],{k:d[k] for k in campos},'escores',d['distribuicao']['todos'],
              'estabilidade ±0.01',d['estabilidade']['pior_acuracia_balanceada_mais_menos_001'],
              'decisao',d['decisao'])


if __name__=='__main__':
    main()
