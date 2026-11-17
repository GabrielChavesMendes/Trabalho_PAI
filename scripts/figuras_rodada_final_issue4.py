"""Figuras da rodada curta usando somente artefatos de treino/validação."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--pasta-figuras',default='figuras')
    args=parser.parse_args()
    base=args.base
    pasta=base/args.pasta_figuras
    if pasta.exists(): raise FileExistsError(pasta)
    pasta.mkdir()
    analise=json.loads((base/'comparacao_validacao.json').read_text(encoding='utf-8'))
    todos=analise['modelos']
    modelo_base=next(d for d in todos if d['id']==analise['referencia_principal'])
    linhas=[modelo_base]+[d for d in todos if d['tipo']=='novos']
    nomes=['CNN anterior','CNN LR menor','CNN LeakyReLU','ResNet50 padronizada']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,d,nome in zip(axes.flat,linhas,nomes):
        h=pd.read_csv(d['historico'])
        ax.plot(h.epoca,h.loss_treino,label='Treino')
        ax.plot(h.epoca,h.loss_validacao,label='Validação')
        ax.axvline(d['epoca'],color='black',ls=':',label='Época escolhida por AUC')
        ax.set(title=f"{nome}; AUC val={d['auc_roc']:.3f}",xlabel='Época completa',ylabel='BCE ponderada (peso 3)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Convergência: mesmo protocolo, seed 42, sem avaliação do teste')
    fig.tight_layout(); fig.savefig(pasta/'convergencia.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    rng=np.random.default_rng(2026)
    for i,(d,nome) in enumerate(zip(linhas,nomes)):
        dados=json.loads(Path(d['validacao']).read_text(encoding='utf-8'))['predicoes_por_paciente']
        x=np.array([p['probabilidade'] for p in dados]); y=np.array([p['rotulo'] for p in dados])
        jitter=rng.uniform(-.07,.07,len(x))
        for col in (0,1):
            ax=axes[i,col]
            for classe,cor in ((0,'royalblue'),(1,'darkorange')):
                ix=y==classe
                ax.scatter(x[ix],y[ix]+jitter[ix],color=cor,label=f'pCR={classe}',s=24)
            ax.axvline(d['limiar'],color='black',ls='--',label='Limiar de validação')
            ax.set(yticks=[0,1],yticklabels=['pCR=0','pCR=1'],ylim=(-.25,1.25),xlabel='Probabilidade média por paciente')
            ax.set_title(f"{nome}: {'escala 0–1' if col==0 else 'zoom; amplitude='+format(np.ptp(x),'.3g')}")
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            if col==0:
                ax.set_xlim(0,1)
                ax.xaxis.set_major_locator(FixedLocator([0,.5,1]))
            else:
                margem=max(np.ptp(x)*.12,1e-7)
                ax.set_xlim(x.min()-margem,x.max()+margem)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis='x',labelsize=8)
            ax.grid(alpha=.2)
            if i==0 and col==0: ax.legend(fontsize=8)
    fig.suptitle('Dispersão dos escores na validação; a coluna de zoom usa escalas diferentes')
    fig.tight_layout(); fig.savefig(pasta/'distribuicao_probabilidades.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for d,nome in zip(linhas,nomes):
        es=d['estabilidade']['avaliacoes']
        ax.plot([e['deslocamento'] for e in es],[e['acuracia_balanceada'] for e in es],marker='o',label=nome)
    ax.axhline(.5,color='gray',ls=':')
    ax.set(xlabel='Deslocamento do limiar escolhido (sem retunar)',ylabel='Acurácia balanceada de validação',
           ylim=(0,1),title='Sensibilidade da decisão ao limiar; mesmos pesos/escores')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(pasta/'estabilidade_limiar.png',dpi=160); plt.close(fig)
    print('Três figuras de comparação geradas em',pasta)


if __name__=='__main__': main()
