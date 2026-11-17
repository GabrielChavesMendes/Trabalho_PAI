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
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.rodada_final_issue4 import carregar_modelo_registrado, criar_carregador
from modulos.modulo_treinamento import (
    avaliar_modelo, escolher_dispositivo, configurar_reprodutibilidade,
    salvar_resultado_json, salvar_matriz_confusao, RegistroPredicao, calcular_metricas,
)
from modulos.grade_final_issue4 import familia, selecionar_representantes
from modulos.analise_validacao_issue4 import estabilidade_limiar


def resolver(caminho):
    p = Path(caminho)
    return p if p.is_absolute() else RAIZ / p


def validar_coorte_27(manifesto):
    entradas = manifesto['modelos']
    esperadas = {(a,f,c) for a in ('cnn','resnet50','mobilenetv2')
                 for f in ('post_early','tres') for c in ('central','maior_tumor')}
    contagens = {g: sum(m['grupo']==g for m in entradas) for g in 'ABC'}
    if len(entradas)!=27 or contagens!={'A':12,'B':3,'C':12}:
        raise ValueError('Coorte definitiva exige exatamente A=12, B=3, C=12')
    b_ids = {m['id'] for m in entradas if m['grupo']=='B'}
    if b_ids != {'e1_cnn_lr_menor','e2_cnn_leaky','e3_resnet_imagenet'}:
        raise ValueError('Os três experimentos intermediários devem ser preservados sem substituição')
    if len({str(resolver(m['checkpoint']).resolve()) for m in entradas}) != 27:
        raise ValueError('Caminhos de checkpoints duplicados')
    for grupo in ('A', 'C'):
        chaves = {(familia(m['configuracao']['arquitetura']),m['configuracao']['configuracao_temporal'],
                   m['configuracao']['estrategia_corte']) for m in entradas if m['grupo']==grupo}
        if chaves != esperadas:
            raise ValueError('Cada grade A/C precisa cobrir as 12 combinações')
    reps = manifesto['representantes']
    if len(reps)!=12 or len(set(reps))!=12 or not set(reps).issubset({m['id'] for m in entradas}):
        raise ValueError('Necessários exatamente 12 representantes únicos e existentes')
    chaves = {(familia(m['configuracao']['arquitetura']),m['configuracao']['configuracao_temporal'],
               m['configuracao']['estrategia_corte']) for m in entradas if m['id'] in reps}
    if chaves != esperadas:
        raise ValueError('Representantes devem cobrir as 12 combinações uma única vez')
    protocolo = manifesto['protocolo_teste']
    if (protocolo['pacientes']!=35 or protocolo['codigo_particao']!=1 or protocolo['aumento'] is not False
            or protocolo['selecionar_vencedor_no_teste'] is not False):
        raise ValueError('Protocolo final deve manter 35 pacientes, sem aumento ou seleção pelo teste')
    for prefixo in ('selecao', 'plano', 'recarga', 'revisao_validacao'):
        caminho = resolver(manifesto[prefixo+'_arquivo'])
        if hashlib.sha256(caminho.read_bytes()).hexdigest()!=manifesto[prefixo+'_sha256']:
            raise ValueError(f'{prefixo} alterado desde congelamento')
    for caminho, digest in manifesto['fontes_avaliacao_sha256'].items():
        if hashlib.sha256(resolver(caminho).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Fonte de avaliação mudou: {caminho}')
    recargas = json.loads(resolver(manifesto['recarga_arquivo']).read_text(encoding='utf-8'))['modelos']
    if len(recargas)!=27 or {r['id'] for r in recargas}!={m['id'] for m in entradas}:
        raise ValueError('Recargas independentes incompletas')
    if any(r['classes_divergentes'] or r['diferenca_max_prob']>1e-6 or r['pacientes']!=33 for r in recargas):
        raise ValueError('Recargas independentes divergentes')
    selecao = json.loads(resolver(manifesto['selecao_arquivo']).read_text(encoding='utf-8'))
    if (selecao['representantes']!=reps or selecao['melhores_por_arquitetura']!=manifesto['melhores_por_arquitetura']
            or selecao['melhor_geral']!=manifesto['modelo_escolhido_pela_validacao'] or selecao['teste_usado']):
        raise ValueError('Decisões de validação divergentes')


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
    if manifesto.get('versao_protocolo') == 2:
        validar_coorte_27(manifesto)
    elif len(originais)!=12 or obtidas!=esperadas or len(novos)>3 or len(entradas)!=12+len(novos):
        raise ValueError('Necessárias 12 configurações originais e no máximo três novos candidatos')
    linhas_validacao = []
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
        if norm != m.get('normalizacao_adicional', norm):
            raise ValueError('Normalização adicional diverge do manifesto')
        if manifesto.get('versao_protocolo') == 2:
            dados = json.loads(validacao.read_text(encoding='utf-8'))
            ps = [RegistroPredicao(p['paciente'],p['rotulo'],p['probabilidade']) for p in dados['predicoes_por_paciente']]
            tabela_val = pd.read_csv(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
            verdade_val = dict(zip(tabela_val.loc[tabela_val.test.eq(2),'pid'],tabela_val.loc[tabela_val.test.eq(2),'pCR']))
            if len(ps)!=33 or {p.paciente:p.rotulo for p in ps}!=verdade_val:
                raise ValueError('Validação deve corresponder aos 33 pacientes oficiais')
            conf = m['configuracao']
            fases = ['pre','post_early','post_late'] if conf['configuracao_temporal']=='tres' else ['post_early']
            if (m['preprocessamento']['fases']!=fases or conf['canais_entrada']!=len(fases)
                    or m['preprocessamento']['corte']!=conf['estrategia_corte']
                    or m['preprocessamento']['augmentation']['teste_aumentado'] is not False):
                raise ValueError('Pré-processamento incompatível com configuração/protocolo')
            metricas = calcular_metricas(ps,m['limiar'])
            for k,v in metricas.items():
                if abs(v-dados['metricas'][k])>1e-12:
                    raise ValueError('Métricas de validação não correspondem às predições')
            linhas_validacao.append(dict(id=m['id'],configuracao=m['configuracao'],**dados['metricas'],
                                          estabilidade=estabilidade_limiar(ps,m['limiar'])))
        del modelo, bruto
    csv = resolver('dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    if hashlib.sha256(csv.read_bytes()).hexdigest()!=manifesto['csv_sha256']:
        raise ValueError('CSV alterado desde o congelamento')
    if manifesto.get('versao_protocolo') == 2:
        reps, melhores, geral = selecionar_representantes(linhas_validacao)
        if (reps!=manifesto['representantes'] or melhores!=manifesto['melhores_por_arquitetura']
                or geral!=manifesto['modelo_escolhido_pela_validacao']):
            raise ValueError('Seleção não corresponde ao critério prévio de validação')
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
    from datetime import datetime, timezone
    (args.saida/'inicio_avaliacao.json').write_text(json.dumps({
        'iniciado_em_utc':datetime.now(timezone.utc).isoformat(),
        'manifesto_sha256':hashlib.sha256(args.manifesto.read_bytes()).hexdigest(),
        'selecoes_congeladas':manifesto.get('representantes',[]),
        'hiperparametros_ou_limiares_ajustados_no_teste':False},indent=2),encoding='utf-8')
    device = escolher_dispositivo(args.device)
    tabela = pd.read_csv(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    verdade = dict(zip(tabela.loc[tabela.test.eq(1),'pid'],tabela.loc[tabela.test.eq(1),'pCR']))
    if len(verdade)!=35:
        raise ValueError('Partição de teste oficial inválida')
    linhas = []
    carregadores = {}  # Cache só de imagens do teste, criado após todo preflight.
    for i, m in enumerate(tarefas,1):
        configurar_reprodutibilidade(int(m['configuracao']['seed']))
        modelo, bruto, norm = carregar_modelo_registrado(resolver(m['checkpoint']),device)
        conf = bruto['configuracao']
        chave = (conf['configuracao_temporal'],conf['estrategia_corte'],norm,conf['tamanho_lote'],conf['seed'])
        if chave not in carregadores:
            carregadores[chave] = criar_carregador('teste',conf,norm,permitir_teste=True)
        loader = carregadores[chave]
        criterio = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([m['configuracao']['pos_weight']],device=device))
        inicio = time.perf_counter()
        loss, metricas, ps = avaliar_modelo(modelo,loader,criterio,device,m['limiar'])
        if device.type=='cuda': torch.cuda.synchronize(device)
        if len(ps)!=35 or len(loader.dataset)!=35 or {p.paciente:p.rotulo for p in ps}!=verdade:
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
    for grupo in sorted({l['grupo'] for l in linhas}):
        grupo_linhas = [l for l in linhas if l['grupo']==grupo]
        if grupo_linhas:
            with (args.saida/f'comparacao_teste_{grupo}.csv').open('x',encoding='utf-8',newline='') as f:
                w = csv.DictWriter(f,fieldnames=list(grupo_linhas[0]))
                w.writeheader(); w.writerows(grupo_linhas)
    (args.saida/'concluido.json').write_text(json.dumps({
        'terminado_em_utc':datetime.now(timezone.utc).isoformat(), 'modelos':len(linhas),
        'pacientes_por_modelo':35, 'selecoes_alteradas':False,
        'manifesto_sha256':hashlib.sha256(args.manifesto.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print('Teste final concluído. Seleção e hiperparâmetros continuam os definidos na validação.')


if __name__ == '__main__':
    main()
