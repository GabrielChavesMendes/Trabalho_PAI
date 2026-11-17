"""Publica tabelas reais já auditadas em Markdown, sem treinar ou avaliar redes."""
import argparse
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import pandas as pd
from scripts.consolidar_coorte_issue4 import ler


def markdown(df, colunas):
    cabecalho = '| ' + ' | '.join(colunas) + ' |\n'
    cabecalho += '|'+'|'.join('---' for _ in colunas)+'|\n'
    for _,d in df.iterrows():
        valores=[]
        for k in colunas:
            v=d[k]
            valores.append(f'{v:.4f}' if isinstance(v,float) else str(v))
        cabecalho += '| '+' | '.join(valores)+' |\n'
    return cabecalho+'\n'


def revisao_validacao(base):
    from datetime import datetime, timezone
    val=pd.read_csv(base/'comparacao_validacao_27.csv')
    dados=ler(base/'comparacao_validacao_27.json')
    selecao=ler(base/'selecao_validacao_congelada.json')
    por_id={d['id']:d for d in dados['modelos']}
    partes=['# Revisão dos 27 — decisões anteriores ao teste\n\n',
            f'Registrado em UTC: {datetime.now(timezone.utc).isoformat()}. '
            '**O teste ainda não foi executado.** O usuário autorizou avaliação automática '
            'somente após terminar a recarga e aprovar o preflight.\n\n',
            'A=12 originais; B=3 intermediários; C=12 novos. '
            'Cada métrica é calculada em 33 pacientes; média das dez probabilidades de cada paciente. '
            'Os 12 originais não incluem as cópias de limiar como novos treinamentos.\n\n',
            '## Validação completa\n\n',
            markdown(val,['codigo','arquitetura','fases','corte','auc_roc','sensibilidade','especificidade',
                          'precisao','acuracia','f1','acuracia_balanceada','loss']),
            '## Doze representantes\n\n',
            markdown(val.loc[val.id.isin(selecao['representantes'])],
                     ['codigo','arquitetura','fases','corte','auc_roc','acuracia_balanceada','sensibilidade','especificidade']),
            'AUC primária; desempates BA, robustez ±.01, sensibilidade, F1, loss e ID, '
            'conforme plano anterior. Oito representantes C, quatro A, zero B. '
            'Algumas combinações têm discriminação baixa: representante significa melhor disponível '
            'pelo critério, não modelo clínico adequado.\n\n']
    for a,id in selecao['melhores_por_arquitetura'].items():
        partes.append(f'- Melhor {a}: **{por_id[id]["codigo"]}**, AUC {por_id[id]["auc_roc"]:.4f}.\n')
    partes.append(f'\nVencedor geral: **{por_id[selecao["melhor_geral"]]["codigo"]}**. '
                  'A CNN original permanece à frente por AUC. Os 27 serão testados, '
                  'sem substituir essas escolhas depois.\n\n')
    partes.append('## Comparação controlada C versus A\n\n')
    comparacoes=[]
    for d in dados['modelos']:
        if d['grupo']=='C':
            b=por_id[d['comparador']]; bs=d['bootstrap_vs_A']
            comparacoes.append(dict(codigo=d['codigo'],comparador=b['codigo'],auc_A=b['auc_roc'],auc_C=d['auc_roc'],
                                   delta_auc=d['auc_roc']-b['auc_roc'],ba_A=b['acuracia_balanceada'],ba_C=d['acuracia_balanceada'],
                                   ic95_delta_inf=bs['ic95_percentil_inferior'],ic95_delta_sup=bs['ic95_percentil_superior']))
    partes.append(markdown(pd.DataFrame(comparacoes),['codigo','comparador','auc_A','auc_C','delta_auc','ba_A','ba_C','ic95_delta_inf','ic95_delta_sup']))
    partes.append('Bootstrap pareado estratificado por paciente, 2.000 reamostragens, seed 2026. '
                  'Intervalos exploratórios: não corrigem seleção de época/modelo nem comparações múltiplas. '
                  'Os aumentos positivos de AUC não excluíram zero nesses intervalos; '
                  'não há evidência convincente de superioridade. C02 piorou numericamente; '
                  'o intervalo negativo isolado também não é uma conclusão confirmatória.\n\n'
                  'CNN: GroupNorm não superou a referência e não resolveu universalmente o colapso. '
                  'ResNet: BN congelada não eliminou sobreajuste; loss elevada. '
                  'Mobile: adaptação parcial favoreceu AUC, principalmente C11, mas continua '
                  'abaixo da melhor CNN e com grande diferença treino/validação.\n\n'
                  'Estabilidade das probabilidades, tempos, época e limiares completos estão no CSV/JSON. '
                  'A avaliação final só começará após a recarga independente dos 27.\n\n')
    for d in dados['modelos']:
        partes.append(f'- **{d["codigo"]}** `{d["id"]}`: limiar `{d["limiar"]!r}`, época {d["epoca"]}; '
                      f'std escores `{d["distribuicao"]["todos"]["std"]!r}`, '
                      f'pior BA ±.01 `{d["estabilidade"]["pior_acuracia_balanceada_mais_menos_001"]!r}`.\n')
    with (base/'REVISAO_VALIDACAO_27.md').open('x',encoding='utf-8') as f: f.write(''.join(partes))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--somente-validacao',action='store_true')
    a=p.parse_args(); base=a.base
    if a.somente_validacao:
        revisao_validacao(base)
        return
    destino=base/'RELATORIO_FINAL.md'
    if destino.exists(): raise FileExistsError('Relatório existente preservado')
    val=pd.read_csv(base/'comparacao_validacao_27.csv')
    df=pd.read_csv(base/'artigo/comparacao_validacao_teste_27.csv')
    rank=pd.read_csv(base/'artigo/ranking_teste_27_acuracia_balanceada.csv')
    rank12=pd.read_csv(base/'artigo/ranking_teste_12_acuracia_balanceada.csv')
    dados=ler(base/'comparacao_validacao_27.json')
    selecao=ler(base/'selecao_validacao_congelada.json')
    destaques=ler(base/'artigo/destaques.json')
    partes=['# Issue #4 — resultados finais da coorte de 27 checkpoints\n\n',
            'Resultados reais: A=12 originais, B=3 intermediários, C=12 novos. '
            '12 representantes fixados antes do teste; quatro por arquitetura. '
            'Teste: mesmos 35 pacientes, sem augmentation, limiares de validação inalterados. '
            'Rankings do teste são descritivos e não mudam os selecionados.\n\n',
            'CNN inclui as variantes ReLU, LeakyReLU e GroupNorm. '
            'post_early=uma fase; tres=pre/post_early/post_late; central=corte Z//2; '
            'maior_tumor=maior área da máscara. A loss é BCE ponderada com pos_weight=3. '
            'No treino/validação inclui as dez variantes; métricas usam média por paciente.\n\n',
            '## Validação dos 27\n\n',
            markdown(val,['codigo','arquitetura','fases','corte','auc_roc','sensibilidade','especificidade',
                          'precisao','acuracia','f1','acuracia_balanceada','loss','epoca','tempo_treinamento_segundos']),
            'A tabela CSV acrescenta intervenção, limiar com precisão completa, confusão e estabilidade.\n\n',
            '## Representantes e escolhas anteriores ao teste\n\n',
            markdown(df.loc[df.representante],['codigo','arquitetura','fases','corte','auc_roc_validacao',
                     'acuracia_balanceada_validacao','auc_roc','acuracia_balanceada']),
            'AUC de validação é primária; desempates previamente registrados no plano. '
            'Diferenças numéricas pequenas não comprovam superioridade estatística.\n\n']
    por_id={d['id']:d for d in dados['modelos']}
    for arq,id in selecao['melhores_por_arquitetura'].items():
        partes.append(f'- Melhor {arq} pela validação: **{por_id[id]["codigo"]}**, `{id}`.\n')
    partes.append(f'\nVencedor geral pela validação: **{por_id[selecao["melhor_geral"]]["codigo"]}**, '
                  f'`{selecao["melhor_geral"]}`.\n\n')
    for titulo, campo in [('Maior BA observada no teste','maior_ba_observada_teste'),
                          ('Maior AUC observada no teste','maior_auc_observada_teste'),
                          ('Maior F1 observado no teste','maior_f1_observado_teste'),
                          ('Maior AUC observada no treino, no checkpoint já escolhido','maior_auc_observada_treino')]:
        partes.append(f'- {titulo}: **{por_id[destaques[campo]]["codigo"]}**, `{destaques[campo]}`.\n')
    partes.append('\n## Ranking A — todos os 27 no teste\n\n')
    colunas=['posicao','codigo','grupo','arquitetura','fases','corte','sensibilidade','especificidade','precisao',
             'acuracia','f1','acuracia_balanceada','auc_roc','loss']
    partes.append(markdown(rank,colunas))
    partes.append('## Ranking B — os 12 representantes no teste\n\n')
    partes.append(markdown(rank12,colunas))
    partes.append('Ordem: BA decrescente, F1 decrescente, AUC decrescente, ID. '
                  'Rankings complementares por AUC e F1 estão em `artigo/`.\n\n')
    partes.append('## Correspondência dos códigos, pesos e resultados\n\n')
    for d in dados['modelos']:
        partes.append(f'- **{d["codigo"]}** `{d["id"]}`: peso local `'+d['checkpoint']+'`; '
                      f'limiar `{d["limiar"]!r}`; época {d["epoca"]}; '
                      f'{d["epocas_executadas"]} épocas completas; intervenção: {d["intervencao"]}.\n')
    partes.append('\n## Artefatos para o artigo\n\n')
    for figura in ['topologia_intervencoes','convergencia_grade_C','auc_por_epoca_C',
                   'ranking_teste_27','validacao_versus_teste','roc_melhores_por_validacao','probabilidades_e_limiares',
                   'ativacoes_cnn','matrizes_representantes','comparacao_arquitetura','comparacao_fases',
                   'comparacao_corte','tempos_treinamento','exemplos_vencedor_teste']:
        partes.append(f'![{figura}](artigo/figuras/{figura}.png)\n\n')
    partes.append('## Limites e integração\n\n'
                  'Dataset pequeno; validação reutilizada; dez aumentos não são dez pacientes. '
                  'Uma seed e uma partição não demonstram robustez externa. Consultas históricas ao '
                  'teste em outra branch impedem chamar o teste de completamente cego; não são '
                  'atribuídas a esta execução. Não houve novos ajustes após a avaliação final.\n\n'
                  'Preservar todos os pesos localmente. Para apresentação usar vencedor da validação, '
                  'destacar melhor por arquitetura e os 12 representantes. Não enviar os 27 pesos '
                  'no ZIP de 10 MB. Integração em Python único, interface/Grad-CAM e artigo SBC '
                  'dependem de revisão dos integrantes. Nenhum merge ou fechamento foi realizado.\n\n'
                  'Uso de IAG: Codex/OpenAI auxiliou código, planejamento, execução, análise e redação. '
                  'Os integrantes devem revisar e validar todas essas partes antes de entrega. '
                  'Prompt e finalidades estão em `docs/PROMPT_GRADE_DEFINITIVA_ISSUE_4_20261117.md`.\n')
    with destino.open('x',encoding='utf-8') as f: f.write(''.join(partes))
    print(destino)


if __name__=='__main__': main()
