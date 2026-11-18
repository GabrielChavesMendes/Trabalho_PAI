"""Audita cálculos e gera tabelas/figuras descritivas após o único teste final."""
import argparse
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score, f1_score)
from scripts.consolidar_coorte import ler, gravar, tabela
from scripts.preparar_grade_final import sha
from modulos.grade_experimental import familia

METRICAS = ['sensibilidade','especificidade','precisao','acuracia','f1','acuracia_balanceada','auc_roc','loss']


def auditar_predicoes(arquivo, tabela_csv, codigo, limiar):
    d = ler(arquivo)
    ps = d['predicoes_por_paciente']
    verdade = dict(zip(tabela_csv.loc[tabela_csv.test.eq(codigo),'pid'],tabela_csv.loc[tabela_csv.test.eq(codigo),'pCR']))
    if len(ps)!=len(verdade) or {p['paciente']:p['rotulo'] for p in ps}!=verdade:
        raise ValueError('Pacientes/rótulos divergentes ou duplicados')
    y = np.array([p['rotulo'] for p in ps])
    prob = np.array([p['probabilidade'] for p in ps])
    previsto = (prob>=limiar).astype(int)
    if not np.array_equal(previsto,[p['previsto'] for p in ps]):
        raise ValueError('Classe armazenada diverge do limiar congelado')
    tn,fp,fn,tp = confusion_matrix(y,previsto,labels=[0,1]).ravel()
    esperado = dict(sensibilidade=recall_score(y,previsto,zero_division=0),
                    especificidade=tn/(tn+fp),precisao=precision_score(y,previsto,zero_division=0),
                    acuracia=accuracy_score(y,previsto),f1=f1_score(y,previsto,zero_division=0),
                    acuracia_balanceada=balanced_accuracy_score(y,previsto),auc_roc=roc_auc_score(y,prob),
                    tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),numero_pacientes=len(ps),limiar=limiar)
    for k,v in esperado.items():
        if not np.isclose(v,d['metricas'][k],atol=1e-12,rtol=0):
            raise ValueError(f'Métrica não confere com sklearn: {arquivo}/{k}')
    return d, esperado


def ranking(df, principal):
    outras = {'acuracia_balanceada':['f1','auc_roc'],'auc_roc':['acuracia_balanceada','f1'],
              'f1':['acuracia_balanceada','auc_roc']}[principal]
    r = df.sort_values([principal,*outras,'id'],ascending=[False,False,False,True]).copy()
    r.insert(0,'posicao',np.arange(1,len(r)+1))
    return r


def topologia(pasta):
    from matplotlib.patches import FancyBboxPatch
    fig, axs = plt.subplots(1,3,figsize=(12,8))
    redes = [
        ('CNN + GroupNorm (C)', ['Entrada: 1 ou 3 × 256 × 256',
          'Conv 3×3: 16 + GN(4) + ReLU','MaxPool 2×2',
          'Conv 3×3: 32 + GN(4) + ReLU','MaxPool 2×2',
          'Conv 3×3: 64 + GN(4) + ReLU','MaxPool 2×2',
          'Conv 3×3: 128 + GN(4) + ReLU','Global average pooling → 128',
          'Linear 128→32 + ReLU + Dropout .3','Linear 32→1 (logit)']),
        ('ResNet50 (C)', ['Entrada: 1 ou 3 × 256 × 256',
          'Pesos iniciais ImageNet','Conv inicial adaptada a 1/3 canais',
          'Stem + layer1–3: congelados','layer4: convoluções treináveis',
          'LR layer4: 1e-5','Todas BN: affine e stats congelados',
          'Global average pooling','FC 2048→1: treinável, LR 1e-4',
          'Saída: logit']),
        ('MobileNetV2 (C)', ['Entrada: 1 ou 3 × 256 × 256',
          'Pesos iniciais ImageNet','Conv inicial adaptada a 1/3 canais',
          'features[0:17]: congeladas','features[17]: último residual',
          'features[18]: projeção final','LR features[17:19]: 1e-5',
          'Todas BN: affine e stats congelados','Global average pooling',
          'Dropout .2 + Linear 1280→1','LR classificador: 1e-4'])]
    for ax,(nome,blocos) in zip(axs,redes):
        ax.set(xlim=(0,1),ylim=(0,12),title=nome); ax.axis('off')
        for i,texto in enumerate(blocos):
            y=11-i
            ax.add_patch(FancyBboxPatch((.03,y),.94,.65,boxstyle='round,pad=.02',
                                       facecolor='#e6eff6',edgecolor='#47647d'))
            ax.text(.5,y+.325,texto,ha='center',va='center',fontsize=8)
            if i<len(blocos)-1: ax.annotate('',xy=(.5,y-.27),xytext=(.5,y),arrowprops={'arrowstyle':'->'})
    fig.suptitle('Última grade: intervenções e partes treináveis; sigmoid apenas na avaliação')
    fig.tight_layout(); fig.savefig(pasta/'topologia_intervencoes.png'); plt.close(fig)


def figura_limiares_legivel(base):
    """Revisão editorial: conserva a primeira figura e exporta PNG/PDF legíveis."""
    pasta=base/'artigo/figuras_revisadas'
    if pasta.exists(): raise FileExistsError('Figura revisada anterior preservada')
    pasta.mkdir()
    linhas=ler(base/'comparacao_validacao_27.json')['modelos']
    selecao=ler(base/'selecao_validacao_congelada.json')
    plt.rcParams.update({'font.size':9,'figure.dpi':150})
    fig,axs=plt.subplots(3,2,figsize=(10,9))
    for i,(arq,id) in enumerate(selecao['melhores_por_arquitetura'].items()):
        d=next(x for x in linhas if x['id']==id)
        ps=ler(RAIZ/d['validacao'])['predicoes_por_paciente']
        v=np.array([p['probabilidade'] for p in ps])
        bins=np.histogram_bin_edges(v,bins=10)
        for classe in (0,1):
            axs[i,0].hist([p['probabilidade'] for p in ps if p['rotulo']==classe],bins=bins,alpha=.5,label=f'pCR={classe}')
        axs[i,0].axvline(d['limiar'],ls=':',color='black')
        ticks=np.linspace(v.min(),v.max(),3)
        axs[i,0].set_xticks(ticks,[f'{t:.8f}' if np.ptp(v)<.001 else f'{t:.3f}' for t in ticks])
        axs[i,0].set(title=f'{arq}: validação',xlabel='Probabilidade',ylabel='Pacientes')
        axs[i,0].legend()
        est=d['estabilidade']['avaliacoes']
        axs[i,1].plot([e['deslocamento'] for e in est],[e['acuracia_balanceada'] for e in est],marker='o')
        axs[i,1].set(title='Perturbação do limiar de validação',xlabel='Delta de limiar',ylabel='Ac. balanceada',ylim=(0,1))
        axs[i,1].set_xticks([-.01,0,.01],['−0,01','0','+0,01'])
    fig.tight_layout()
    fig.savefig(pasta/'probabilidades_e_limiares.png')
    fig.savefig(pasta/'probabilidades_e_limiares.pdf')
    plt.close(fig)


def figura_ranking_legivel(base):
    pasta=base/'artigo/figuras_revisadas'
    pasta.mkdir(exist_ok=True)
    if (pasta/'ranking_teste_27.png').exists(): raise FileExistsError('Ranking revisado preservado')
    df=pd.read_csv(base/'artigo/ranking_teste_27_acuracia_balanceada.csv')
    rotulos=[]
    for _,d in df.iterrows():
        fase='1F' if d.fases=='post_early' else '3F'
        corte='C' if d.corte=='central' else 'T'
        rotulos.append(f'{d.codigo} {d.arquitetura} {fase} {corte}'+(' *' if d.representante else '')
                       +(' [V]' if d.vencedor_validacao else ''))
    plt.rcParams.update({'font.size':9,'figure.dpi':150})
    fig,ax=plt.subplots(figsize=(10,10))
    y=np.arange(len(df))
    ax.barh(y-.16,df.acuracia_balanceada,height=.3,label='Ac. balanceada')
    ax.barh(y+.16,df.auc_roc,height=.3,label='AUC')
    ax.set(yticks=y,yticklabels=rotulos,xlim=(0,1),title='Teste: 27 modelos (* representante; [V] escolhido na validação)')
    ax.invert_yaxis(); ax.legend(); ax.axvline(.5,color='gray',ls=':',lw=1)
    fig.tight_layout(); fig.savefig(pasta/'ranking_teste_27.png'); fig.savefig(pasta/'ranking_teste_27.pdf'); plt.close(fig)


def graficos(base, linhas, df):
    pasta = base/'artigo/figuras'
    pasta.mkdir(parents=True)
    topologia(pasta)
    plt.rcParams.update({'font.size':9, 'figure.dpi':150})
    nomes = {d['id']: f"{d['grupo']} {familia(d['configuracao']['arquitetura'])} "
             f"{'1F' if d['configuracao']['configuracao_temporal']=='post_early' else '3F'} "
             f"{'C' if d['configuracao']['estrategia_corte']=='central' else 'T'}" for d in linhas}
    r = ranking(df,'acuracia_balanceada')
    fig, ax = plt.subplots(figsize=(10,10))
    y = np.arange(len(r))
    ax.barh(y-.16,r.acuracia_balanceada,height=.3,label='Ac. balanceada')
    ax.barh(y+.16,r.auc_roc,height=.3,label='AUC')
    ax.set(yticks=y,yticklabels=[nomes[x]+(' *' if bool(rep) else '') for x,rep in zip(r.id,r.representante)],
           xlim=(0,1),title='Teste: 27 checkpoints (* representantes fixados na validação)')
    ax.invert_yaxis(); ax.legend(); ax.axvline(.5,color='gray',ls=':',lw=1)
    fig.tight_layout(); fig.savefig(pasta/'ranking_teste_27.png'); plt.close(fig)
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    for metrica, ax in zip(('auc_roc','acuracia_balanceada'),axs):
        for g in 'ABC':
            sub = df.loc[df.grupo.eq(g)]
            ax.scatter(sub[metrica+'_validacao'],sub[metrica],label=g)
        ax.plot([0,1],[0,1],ls=':',color='gray')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Validação',ylabel='Teste',title=metrica)
        ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'validacao_versus_teste.png'); plt.close(fig)
    fig, axs = plt.subplots(3,4,figsize=(13,8))
    for ax,d in zip(axs.flat,[d for d in linhas if d['grupo']=='C']):
        h = pd.read_csv(RAIZ/d['historico'])
        ax.plot(h.epoca,h.loss_treino,label='Treino')
        ax.plot(h.epoca,h.loss_validacao,label='Validação')
        ax.axvline(d['epoca'],color='gray',ls=':')
        ax.set(title=nomes[d['id']],xlabel='Época',ylabel='BCE ponderada'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pasta/'convergencia_grade_C.png'); plt.close(fig)
    fig, axs = plt.subplots(3,4,figsize=(13,8))
    for ax,d in zip(axs.flat,[d for d in linhas if d['grupo']=='C']):
        h = pd.read_csv(RAIZ/d['historico'])
        ax.plot(h.epoca,h.auc_roc,label='AUC validação')
        ax.axvline(d['epoca'],color='gray',ls=':')
        ax.axhline(.5,color='gray',ls='--',lw=.7)
        ax.set(title=nomes[d['id']],xlabel='Época',ylabel='AUC por paciente',ylim=(0,1))
    fig.tight_layout(); fig.savefig(pasta/'auc_por_epoca_C.png'); plt.close(fig)
    # Curvas dos três melhores pela validação, comparador A e nova intervenção C.
    selecao = ler(base/'selecao_validacao_congelada.json')
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5,4))
    for arq,id_modelo in selecao['melhores_por_arquitetura'].items():
        d = next(x for x in linhas if x['id']==id_modelo)
        ps = ler(base/'teste_final'/d['grupo']/(id_modelo+'__test.json'))['predicoes_por_paciente']
        fpr,tpr,_ = roc_curve([p['rotulo'] for p in ps],[p['probabilidade'] for p in ps])
        auc = float(df.loc[df.id.eq(id_modelo),'auc_roc'].iloc[0])
        ax.plot(fpr,tpr,label=f'{arq}: AUC {auc:.3f}')
    ax.plot([0,1],[0,1],color='gray',ls=':')
    ax.set(xlabel='1 - Especificidade',ylabel='Sensibilidade',xlim=(0,1),ylim=(0,1),
           title='Teste: três checkpoints fixados pela validação'); ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'roc_melhores_por_validacao.png'); plt.close(fig)
    fig, axs = plt.subplots(3,2,figsize=(10,10))
    for i, (arq, id_modelo) in enumerate(selecao['melhores_por_arquitetura'].items()):
        d = next(x for x in linhas if x['id']==id_modelo)
        ps = ler(RAIZ/d['validacao'])['predicoes_por_paciente']
        for classe in (0,1):
            v = [p['probabilidade'] for p in ps if p['rotulo']==classe]
            axs[i,0].hist(v,bins=10,alpha=.5,label=f'pCR={classe}')
        axs[i,0].axvline(d['limiar'],ls=':',color='black')
        axs[i,0].ticklabel_format(axis='x',style='plain',useOffset=False)
        axs[i,0].set(title=f'{arq}: escores de validação',xlabel='Probabilidade',ylabel='Pacientes')
        axs[i,0].legend()
        est = d['estabilidade']['avaliacoes']
        axs[i,1].plot([e['deslocamento'] for e in est],[e['acuracia_balanceada'] for e in est],marker='o')
        axs[i,1].set(title='Limiar fixado na validação: perturbações',xlabel='Delta de limiar',ylabel='Ac. balanceada',ylim=(0,1))
    fig.tight_layout(); fig.savefig(pasta/'probabilidades_e_limiares.png'); plt.close(fig)
    reps = df.loc[df.representante]
    fig, axs = plt.subplots(3,4,figsize=(11,8))
    for ax, (_,d) in zip(axs.flat,reps.iterrows()):
        matriz = np.array([[d.tn,d.fp],[d.fn,d.tp]],dtype=int)
        ax.imshow(matriz,cmap='Blues',vmin=0,vmax=35)
        for i in range(2):
            for j in range(2): ax.text(j,i,str(matriz[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predito',ylabel='Real',title=nomes[d.id])
    fig.tight_layout(); fig.savefig(pasta/'matrizes_representantes.png'); plt.close(fig)
    for eixo in ('grupo','arquitetura','fases','corte'):
        agrupado = reps.groupby(eixo)[METRICAS].mean()
        ax = agrupado[['sensibilidade','especificidade','precisao','acuracia','f1']].plot.bar(figsize=(10,4),ylim=(0,1))
        ax.set(title=f'Teste: médias descritivas dos representantes por {eixo}',ylabel='Métrica')
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(pasta/f'comparacao_{eixo}.png'); plt.close()
    # Sonda de ativação: mesma classe/corte/fases, oito pacientes de treino, zero passos.
    diag = ler(base/'recarga_independente/diagnostico_ativacoes_cnn.json')
    rows = []
    for d in diag:
        relus = [(k,v) for k,v in d['ativacoes'].items() if k.startswith('caracteristicas.') and len(v['shape'])==4]
        ultima = relus[-1][1]
        rows.append(dict(id=d['id'],grupo=d['grupo'],**{k:ultima[k] for k in
                    ('fracao_zeros','fracao_canais_totalmente_zero','variacao_entre_pacientes')},
                    fracao_neuronios_cabeca_zero=d['ativacoes']['classificador.1']['fracao_canais_totalmente_zero'],
                    variacao_cabeca=d['ativacoes']['classificador.1']['variacao_entre_pacientes'],
                    amplitude_probabilidades=d['amplitude_probabilidades']))
    pd.DataFrame(rows).to_csv(base/'artigo/diagnostico_ativacoes_cnn.csv',index=False)
    fig, axs = plt.subplots(2,1,figsize=(10,7))
    for ax,chave,titulo in zip(axs,('fracao_canais_totalmente_zero','fracao_neuronios_cabeca_zero'),
                              ('Última ReLU convolucional','ReLU da cabeça (32 neurônios)')):
        ax.bar(np.arange(len(rows)),[d[chave] for d in rows])
        ax.set(xticks=np.arange(len(rows)),xticklabels=[nomes[d['id']] for d in rows],ylim=(0,1),
               ylabel='Fração inteiramente zero',title=titulo+' — oito pacientes de treino')
        ax.tick_params(axis='x',rotation=30)
    fig.tight_layout(); fig.savefig(pasta/'ativacoes_cnn.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(np.arange(len(linhas)),[d['tempo_treinamento_segundos']/60 for d in linhas])
    ax.set(xticks=np.arange(len(linhas)),xticklabels=[nomes[d['id']] for d in linhas],
           ylabel='Minutos',title='Tempo efetivo de treino (inclui validação por época)')
    plt.xticks(rotation=75,ha='right'); fig.tight_layout(); fig.savefig(pasta/'tempos_treinamento.png'); plt.close(fig)
    # Imagens reais do vencedor escolhido ANTES do teste. Não executa inferência.
    from modulos.modulo_dataset import carregar_metadados, criar_manifesto, DatasetDCE
    vencedor = next(d for d in linhas if d['id']==selecao['melhor_geral'])
    preds = ler(base/'teste_final'/vencedor['grupo']/(vencedor['id']+'__test.json'))['predicoes_por_paciente']
    exemplos = []
    for nome, real, predito in [('TN',0,0),('TP',1,1),('FP',0,1),('FN',1,0)]:
        candidatos = sorted([p for p in preds if p['rotulo']==real and p['previsto']==predito],key=lambda p:p['paciente'])
        if candidatos: exemplos.append(dict(tipo=nome,**candidatos[0]))
    gravar(base/'artigo/exemplos_vencedor.json',exemplos)
    metadata = carregar_metadados(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    metadata = metadata.loc[metadata.test.eq(1)]
    pacientes = criar_manifesto(metadata,RAIZ/'dataset/BreastDCEDL_ISPY1_min_crop')
    conf = vencedor['configuracao']
    fig, axs = plt.subplots(1,4,figsize=(12,3))
    for ax in axs: ax.axis('off')
    for ax, exemplo in zip(axs,exemplos):
        paciente = next(p for p in pacientes if p.identificador==exemplo['paciente'])
        amostra = DatasetDCE([paciente],fase=conf['configuracao_temporal'],estrategia=conf['estrategia_corte'])[0]
        canal = 1 if conf['configuracao_temporal']=='tres' else 0
        ax.imshow(amostra['imagem'][canal],cmap='gray',vmin=0,vmax=1)
        ax.set_title(f"{exemplo['tipo']} {exemplo['paciente']}\np={exemplo['probabilidade']:.6g}",fontsize=9)
    fig.tight_layout(); fig.savefig(pasta/'exemplos_vencedor_teste.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--somente-figura-limiares',action='store_true')
    p.add_argument('--somente-ranking-legivel',action='store_true')
    a = p.parse_args(); base = a.base
    if a.somente_figura_limiares:
        figura_limiares_legivel(base)
        return
    if a.somente_ranking_legivel:
        figura_ranking_legivel(base)
        return
    destino = base/'artigo'
    if destino.exists(): raise FileExistsError('Relatórios anteriores preservados')
    manifesto = ler(base/'manifesto_final_27.json')
    termino = ler(base/'teste_final/concluido.json')
    if termino['modelos']!=27 or termino['manifesto_sha256']!=sha(base/'manifesto_final_27.json'):
        raise ValueError('Avaliação não corresponde ao manifesto congelado')
    linhas = ler(base/'comparacao_validacao_27.json')['modelos']
    selecao = ler(base/'selecao_validacao_congelada.json')
    if sha(base/'selecao_validacao_congelada.json')!=manifesto['selecao_sha256']:
        raise ValueError('Seleção foi alterada após congelamento')
    csv = pd.read_csv(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    ids = [set(csv.loc[csv.test.eq(k),'pid']) for k in (0,2,1)]
    if [len(s) for s in ids]!=[104,33,35] or any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Vazamento/partições incorretas')
    resultado, auditoria = [], []
    for d in linhas:
        auditar_predicoes(RAIZ/d['validacao'],csv,2,d['limiar'])
        arquivo = base/'teste_final'/d['grupo']/(d['id']+'__test.json')
        teste, esperado = auditar_predicoes(arquivo,csv,1,d['limiar'])
        treino, _ = auditar_predicoes(base/'recarga_independente'/(d['id']+'__treino.json'),csv,0,d['limiar'])
        r = tabela([d]).iloc[0].to_dict()
        for k in METRICAS:
            r[k+'_validacao'] = r[k]
            r[k] = teste['metricas'][k]
            r[k+'_treino'] = treino['metricas'][k]
        for k in ('tn','fp','fn','tp'):
            r[k+'_validacao'] = r[k]; r[k] = teste['metricas'][k]
        r['representante'] = d['id'] in selecao['representantes']
        r['vencedor_validacao'] = d['id']==selecao['melhor_geral']
        r['delta_auc_teste_validacao'] = r['auc_roc']-r['auc_roc_validacao']
        r['delta_ba_teste_validacao'] = r['acuracia_balanceada']-r['acuracia_balanceada_validacao']
        resultado.append(r)
        auditoria.append(dict(id=d['id'],validacao_sklearn=True,teste_sklearn=True,treino_sklearn=True,
                              pacientes_teste=esperado['numero_pacientes']))
    destino.mkdir()
    df = pd.DataFrame(resultado)
    correlacoes = {}
    for metrica in ('auc_roc','acuracia_balanceada','f1'):
        ordem_val = df.sort_values([metrica+'_validacao','id'],ascending=[False,True]).id.tolist()
        ordem_teste = df.sort_values([metrica,'id'],ascending=[False,True]).id.tolist()
        df['posicao_'+metrica+'_validacao'] = df.id.map({id:i+1 for i,id in enumerate(ordem_val)})
        df['posicao_'+metrica+'_teste'] = df.id.map({id:i+1 for i,id in enumerate(ordem_teste)})
        df['mudanca_posicao_'+metrica] = df['posicao_'+metrica+'_teste']-df['posicao_'+metrica+'_validacao']
        correlacoes[metrica] = float(df[metrica+'_validacao'].corr(df[metrica],method='spearman'))
    df.to_csv(destino/'comparacao_validacao_teste_27.csv',index=False)
    df.loc[df.representante].to_csv(destino/'representantes_validacao_teste_12.csv',index=False)
    df.loc[df.id.isin(selecao['melhores_por_arquitetura'].values())].to_csv(destino/'melhores_por_arquitetura_validacao.csv',index=False)
    for nome, sub in [('27',df),('12',df.loc[df.representante])]:
        for metrica in ('acuracia_balanceada','auc_roc','f1'):
            ranking(sub,metrica).to_csv(destino/f'ranking_teste_{nome}_{metrica}.csv',index=False)
    for eixo in ('grupo','arquitetura','fases','corte'):
        # Estratificação completa dos grupos e comparação restrita aos representantes.
        df.groupby(eixo)[METRICAS].agg(['mean','min','max','count']).to_csv(destino/f'resumo_todos_por_{eixo}.csv')
        df.loc[df.representante].groupby(eixo)[METRICAS].agg(['mean','min','max','count']).to_csv(destino/f'resumo_representantes_por_{eixo}.csv')
    pareadas = []
    for _, c in df.loc[df.grupo.eq('C')].iterrows():
        original = df.loc[df.grupo.eq('A') & df.arquitetura.eq(c.arquitetura) & df.fases.eq(c.fases) & df.corte.eq(c.corte)].iloc[0]
        pareadas.append(dict(id=c.id,comparador=original.id,arquitetura=c.arquitetura,fases=c.fases,corte=c.corte,
                             **{f'delta_{k}{s}':c[k+s]-original[k+s] for k in METRICAS for s in ('','_validacao')}))
    pd.DataFrame(pareadas).to_csv(destino/'comparacao_pareada_C_vs_A_validacao_teste.csv',index=False)
    # Contrastes dentro de cada grade completa: mantém intervenção e arquitetura.
    # A média dos representantes sozinha mistura grupos escolhidos pela validação.
    for fator, primeiro, segundo in [('fases','post_early','tres'),('corte','central','maior_tumor')]:
        contrastes = []
        for grupo in ('A','C'):
            for arq in ('cnn','resnet50','mobilenetv2'):
                sub = df.loc[df.grupo.eq(grupo)&df.arquitetura.eq(arq)]
                um = sub.loc[sub[fator].eq(primeiro)]
                dois = sub.loc[sub[fator].eq(segundo)]
                contrastes.append(dict(grupo=grupo,arquitetura=arq,contraste=f'{segundo} - {primeiro}',
                    **{f'delta_media_{k}{s}':float(dois[k+s].mean()-um[k+s].mean())
                       for k in METRICAS for s in ('','_validacao')}))
        pd.DataFrame(contrastes).to_csv(destino/f'contrastes_{fator}_dentro_grade.csv',index=False)
    destaques = dict(vencedor_validacao=selecao['melhor_geral'],
                     melhores_por_arquitetura_validacao=selecao['melhores_por_arquitetura'],
                     maior_ba_observada_teste=ranking(df,'acuracia_balanceada').iloc[0].id,
                     maior_auc_observada_teste=ranking(df,'auc_roc').iloc[0].id,
                     maior_f1_observado_teste=ranking(df,'f1').iloc[0].id,
                     maior_auc_observada_treino=df.sort_values(['auc_roc_treino','id'],ascending=[False,True]).iloc[0].id,
                     empates_maior_auc_treino=df.loc[df.auc_roc_treino.eq(df.auc_roc_treino.max()),'id'].tolist(),
                     empates_maior_ba_teste=df.loc[df.acuracia_balanceada.eq(df.acuracia_balanceada.max()),'id'].tolist(),
                     empates_maior_auc_teste=df.loc[df.auc_roc.eq(df.auc_roc.max()),'id'].tolist(),
                     empates_maior_f1_teste=df.loc[df.f1.eq(df.f1.max()),'id'].tolist(),
                     selecoes_alteradas=False)
    gravar(destino/'destaques.json',destaques)
    gravar(destino/'correlacoes_validacao_teste.json',dict(spearman=correlacoes,
           observacao='Descritivo; modelos e pacientes compartilhados, sem teste de superioridade'))
    gravar(destino/'auditoria_metricas.json',dict(modelos=auditoria,coorte=27,representantes=12,
                                                particoes_disjuntas=True,conferido_com_sklearn=True,
                                                selecao_sha256=sha(base/'selecao_validacao_congelada.json'),
                                                manifesto_sha256=sha(base/'manifesto_final_27.json')))
    # Tabela compacta reutilizável pelo relatório SBC, sem depender de pacote LaTeX novo.
    colunas = ['arquitetura','fases','corte','grupo','sensibilidade','especificidade','precisao','acuracia','f1','acuracia_balanceada','auc_roc']
    with (destino/'representantes_teste.tex').open('x',encoding='utf-8') as f:
        f.write(df.loc[df.representante,colunas].to_latex(index=False,float_format='%.3f',escape=True))
    graficos(base,linhas,df)
    print(json.dumps(destaques,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
