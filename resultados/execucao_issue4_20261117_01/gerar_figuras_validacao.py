"""Figuras de validação desta execução; não abre o conjunto de teste."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent


def curvas(linhas, destino, colunas):
    if destino.exists():
        raise FileExistsError(destino)
    n = len(linhas)
    fig, axes = plt.subplots((n + colunas - 1) // colunas, colunas,
                             figsize=(5 * colunas, 3.5 * ((n + colunas - 1) // colunas)),
                             squeeze=False)
    for ax, linha in zip(axes.flat, linhas):
        if n == 4:
            caminho = BASE / 'ajustes' / linha['candidato'] / 'resultados'
            titulo = linha['candidato']
        else:
            caminho = BASE / 'grade'
            titulo = linha['experimento'].removesuffix('__seed-42')
        h = pd.read_csv(caminho / (linha['experimento'] + '__history.csv'))
        ax.plot(h.epoca, h.loss_treino, label='Treino')
        ax.plot(h.epoca, h.loss_validacao, label='Validação')
        ax.axvline(linha['epoca_escolhida'], color='black', ls=':', label='Checkpoint por AUC')
        ax.set(title=f"{titulo}\nAUC val={linha['auc_roc']:.3f}; {linha['tempo_treinamento_segundos']:.0f} s",
               xlabel='Época completa', ylabel='BCE ponderada')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle('Validação por paciente (n=33); modelos selecionados na mesma validação', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(destino, dpi=160)
    plt.close(fig)


ajustes = json.loads((BASE / 'auditoria_ajustes.json').read_text(encoding='utf-8'))
curvas(ajustes, BASE / 'curvas_quatro_ajustes.png', 2)
grade = json.loads((BASE / 'auditoria_grade.json').read_text(encoding='utf-8'))
curvas(grade, BASE / 'curvas_doze_modelos.png', 3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, corte in zip(axes, ('central', 'maior_tumor')):
    for arquitetura in ('cnn', 'mobilenetv2', 'resnet50'):
        pares = [next(d for d in grade if d['arquitetura'] == arquitetura
                      and d['estrategia_corte'] == corte and d['configuracao_temporal'] == fase)
                 for fase in ('post_early', 'tres')]
        ax.plot([0, 1], [p['auc_roc'] for p in pares], marker='o', label=arquitetura)
    ax.axhline(.5, color='gray', ls='--', label='Escore constante (AUC 0,5)')
    ax.set(xticks=[0, 1], xticklabels=['post_early', 'Três fases'], ylim=(0, 1),
           title=corte, ylabel='AUC de validação por paciente')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('Comparações pareadas na validação (33 pacientes; uma seed)')
fig.tight_layout()
destino = BASE / 'comparacao_fases_cortes_validacao.png'
if destino.exists():
    raise FileExistsError(destino)
fig.savefig(destino, dpi=160)
plt.close(fig)
print('Três figuras de validação geradas.')

arquivo_refinados = BASE / 'auditoria_limiares.json'
if arquivo_refinados.exists():
    refinados = json.loads(arquivo_refinados.read_text(encoding='utf-8'))
    por_nome = {d['experimento']: d for d in refinados}
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = list(range(len(grade)))
    ax.bar([x - .18 for x in xs], [d['acuracia_balanceada'] for d in grade],
           width=.36, label='Grade de limiares original', color='gray')
    ax.bar([x + .18 for x in xs], [por_nome[d['experimento']]['acuracia_balanceada'] for d in grade],
           width=.36, label='Pontos médios de escores de validação', color='teal')
    ax.axhline(.5, color='black', ls=':')
    ax.set(xticks=xs, xticklabels=[d['experimento'].removesuffix('__seed-42') for d in grade],
           ylabel='Acurácia balanceada de validação', ylim=(0, 1),
           title='Mesmos pesos e AUC; muda somente o limiar\nEstudo exploratório nos mesmos 33 pacientes')
    plt.setp(ax.get_xticklabels(), rotation=65, ha='right', fontsize=8)
    ax.legend()
    fig.tight_layout()
    destino = BASE / 'comparacao_limiares_validacao.png'
    if destino.exists():
        raise FileExistsError(destino)
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    selecao = json.loads((BASE / 'grade_limiares_validacao/modelo_selecionado.json').read_text(encoding='utf-8'))
    dados = json.loads((BASE / 'grade_limiares_validacao' / (selecao['experimento'] + '__validation.json')).read_text(encoding='utf-8'))
    ps = sorted(dados['predicoes_por_paciente'], key=lambda p: p['probabilidade'])
    fig, ax = plt.subplots(figsize=(10, 4))
    for classe, cor in ((0, 'royalblue'), (1, 'darkorange')):
        ax.scatter([i for i, p in enumerate(ps) if p['rotulo'] == classe],
                   [p['probabilidade'] for p in ps if p['rotulo'] == classe],
                   label=f'pCR={classe}', color=cor)
    ax.axhline(dados['metricas']['limiar'], color='black', ls='--', label='Limiar selecionado na validação')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.set(xlabel='Pacientes ordenados pelo escore', ylabel='Probabilidade média das 10 variantes',
           title=f"{selecao['experimento']}\nAUC={dados['metricas']['auc_roc']:.3f}; escala vertical ampliada")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    destino = BASE / 'escores_validacao_modelo_escolhido.png'
    if destino.exists():
        raise FileExistsError(destino)
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    print('Duas figuras suplementares de limiar/escores geradas.')
