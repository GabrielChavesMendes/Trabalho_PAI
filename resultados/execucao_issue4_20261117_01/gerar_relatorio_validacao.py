"""Consolida artefatos já calculados de validação para revisão humana."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
destino = BASE / 'REVISAO_ANTES_TESTE.md'
if destino.exists():
    raise FileExistsError(destino)
ler = lambda nome: json.loads((BASE / nome).read_text(encoding='utf-8'))
ajustes = ler('auditoria_ajustes.json')
grade = ler('auditoria_grade.json')
refinados = ler('auditoria_limiares.json')
selecao = ler('grade_limiares_validacao/modelo_selecionado.json')
recarga = ler('recarga_validacao/relatorio.json')
original_por_nome = {d['experimento']: d for d in grade}
escolhido = next(d for d in refinados if d['experimento'] == selecao['experimento'])
linhas = [
    '# Issue #4 — revisão anterior ao teste final',
    '',
    '**Etapas de ajuste e validação concluídas. Avaliação final no teste não executada.**',
    'Nenhum merge na develop ou fechamento da issue foi realizado.',
    '',
    'CUDA corrigido na `.venv`: PyTorch 2.9.1+cu126 / torchvision 0.24.1+cu126; GTX 1650 SUPER.',
    '36 testes automatizados passaram, além dos smoke tests das três redes na GPU.',
    'Partições: 104 treino, 33 validação, 35 teste; aumento ×10 somente em treino/validação.',
    'As cinco métricas, AUC e acurácia balanceada são calculadas por paciente após média das probabilidades.',
    'Loss é a média da BCE ponderada nas amostras aumentadas; os 330 exemplos de validação não são 330 pacientes independentes.',
    '',
    '## Ajustes iniciais (ResNet50, três fases, maior tumor)',
    '',
    '| Candidato | AUC | Ac. bal. | F1 | Sens. | Espec. | Loss | Época/total | Tempo (s) |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for d in sorted(ajustes, key=lambda d: -d['auc_roc']):
    linhas.append(f"| {d['candidato']} | {d['auc_roc']:.4f} | {d['acuracia_balanceada']:.4f} | {d['f1']:.4f} | {d['sensibilidade']:.4f} | {d['especificidade']:.4f} | {d['loss']:.4f} | {d['epoca_escolhida']}/{d['epocas_executadas']} | {d['tempo_treinamento_segundos']:.1f} |")
linhas += [
    '',
    'Escolha anterior à grade: peso 3,0 e fine-tuning de layer4 na ResNet50 pela maior AUC.',
    'A escolha não elimina o forte sobreajuste observado nem prova melhora nas demais combinações.',
    '',
    '## Grade de 12 configurações e estudo de limiares',
    '',
    'Os pesos, épocas, AUCs e tempos são os mesmos nas duas versões. A versão suplementar muda apenas o limiar,',
    'usando pontos médios entre escores distintos de validação, extremos internos e 0,5. Critério: acurácia balanceada;',
    'desempate: especificidade, sensibilidade e proximidade de 0,5. A versão original com passo 0,05 foi preservada.',
    'Este refinamento foi motivado por inspeção da validação durante a grade; é exploratório e foi aplicado uniformemente às 12 configurações.',
    '',
    '| Configuração | AUC | Ac. bal. original | Ac. bal. refinada | F1 original | F1 refinado | Positivos previstos/33 |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for d in sorted(refinados, key=lambda d: -d['auc_roc']):
    a = original_por_nome[d['experimento']]
    nome = d['experimento'].removesuffix('__seed-42')
    linhas.append(f"| {nome} | {d['auc_roc']:.4f} | {a['acuracia_balanceada']:.4f} | {d['acuracia_balanceada']:.4f} | {a['f1']:.4f} | {d['f1']:.4f} | {d['tp']+d['fp']} |")
linhas += ['', '| Configuração | Sens. | Espec. | Precisão | Acurácia | Loss | Época/total | Tempo (s) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for d in sorted(refinados, key=lambda d: -d['auc_roc']):
    linhas.append(f"| {d['experimento'].removesuffix('__seed-42')} | {d['sensibilidade']:.4f} | {d['especificidade']:.4f} | {d['precisao']:.4f} | {d['acuracia']:.4f} | {d['loss']:.4f} | {d['epoca_escolhida']}/{d['epocas_executadas']} | {d['tempo_treinamento_segundos']:.1f} |")
linhas += [
    '',
    'Tabela completa com limiares sem arredondamento, hiperparâmetros, TN/FP/FN/TP e hashes: [auditoria_limiares.csv](auditoria_limiares.csv).',
    'Comparação antes/depois: [comparacao_limiares.csv](grade_limiares_validacao/comparacao_limiares.csv).',
    '',
    '## Configuração escolhida exclusivamente pela validação',
    '',
    f"**{escolhido['experimento']}**; época {escolhido['epoca_escolhida']}; limiar completo **{escolhido['limiar']!r}**.",
    f"AUC={escolhido['auc_roc']:.4f}; ac. balanceada={escolhido['acuracia_balanceada']:.4f}; F1={escolhido['f1']:.4f}; sensibilidade={escolhido['sensibilidade']:.4f}; especificidade={escolhido['especificidade']:.4f}.",
    f"Precisão={escolhido['precisao']:.4f}; acurácia={escolhido['acuracia']:.4f}; TN={escolhido['tn']}, FP={escolhido['fp']}, FN={escolhido['fn']}, TP={escolhido['tp']}.",
    f"Checkpoint: `{escolhido['checkpoint']}`. SHA-256: `{escolhido['checkpoint_sha256']}`.",
    f"Adam; LR={escolhido['taxa_aprendizado']}; weight decay={escolhido['decaimento_pesos']}; peso positivo={escolhido['pos_weight']}; batch={escolhido['tamanho_lote']}; seed={escolhido['seed']}; máximo de {escolhido['epocas']} épocas, paciência {escolhido['paciencia']}.",
    'Época por AUC de validação (empate por loss); modelo por AUC (desempates: ac. balanceada, sensibilidade, loss e nome).',
    'Normalização: percentis 1/99 conjuntos por amostra, clip [0,1]; corte e fases conforme o nome da configuração.',
    '',
    '## Efeito das três fases na validação',
    '',
    '| Rede | Corte | AUC uma fase | AUC três fases | ΔAUC (três − uma) |',
    '|---|---|---:|---:|---:|',
]
for a in ('cnn', 'mobilenetv2', 'resnet50'):
    for c in ('central', 'maior_tumor'):
        pares = [next(d for d in grade if d['arquitetura']==a and d['estrategia_corte']==c and d['configuracao_temporal']==f)
                 for f in ('post_early', 'tres')]
        linhas.append(f"| {a} | {c} | {pares[0]['auc_roc']:.4f} | {pares[1]['auc_roc']:.4f} | {pares[1]['auc_roc']-pares[0]['auc_roc']:+.4f} |")
linhas += ['', '## Efeito do corte na validação', '',
           '| Rede | Fases | AUC central | AUC maior tumor | ΔAUC (maior − central) |',
           '|---|---|---:|---:|---:|']
for a in ('cnn', 'mobilenetv2', 'resnet50'):
    for f in ('post_early', 'tres'):
        pares = [next(d for d in grade if d['arquitetura']==a and d['estrategia_corte']==c and d['configuracao_temporal']==f)
                 for c in ('central', 'maior_tumor')]
        linhas.append(f"| {a} | {f} | {pares[0]['auc_roc']:.4f} | {pares[1]['auc_roc']:.4f} | {pares[1]['auc_roc']-pares[0]['auc_roc']:+.4f} |")
linhas += [
    '',
    '## Limitações e verificações',
    '',
    '- Os mesmos 33 pacientes servem para escolher época, limiar e configuração. Métricas de validação são otimistas; uma seed e 12 configurações não sustentam alegação de superioridade robusta.',
    '- No estudo inicial o fine-tuning reduziu fortemente a loss de treino e aumentou a de validação: sobreajuste. A CNN de fase única manteve loss de treino perto de 1,04: aprendizagem discriminativa limitada.',
    '- Na CNN de fase única/maior tumor, escores entre 0,4984775841 e 0,4984809875 tornam o limiar numericamente sensível. O ganho de F1 vem da escolha do limiar na validação, sem melhora dos pesos ou AUC. Não arredondar o limiar de aplicação.',
    '- Teste não aumentado e validação com média das 10 variantes seguem o protocolo; as distribuições de escore podem diferir. Não adaptar o limiar depois de consultar o teste.',
    '- Losses de candidatos com pesos de classe diferentes não são diretamente comparáveis. Tempos incluem preparação por época e execução num Windows com outros aplicativos, não um benchmark isolado.',
    '- O grupo já consultou os resultados originais do teste antes desta revisão. Portanto, o teste não é completamente cego; nenhuma nova métrica de teste orientou esta execução.',
    '- 12 checkpoints originais e 12 cópias com limiares refinados foram recarregados estritamente em CPU. Cada cópia preserva todos os tensores de pesos da origem; os quatro candidatos também foram auditados.',
    f"- Recarga com inferência na validação: maior diferença de probabilidade={recarga['maior_diferenca_absoluta_probabilidade']!r}; classificações divergentes={recarga['classificacoes_divergentes']}. Essa verificação na mesma GPU não prova estabilidade entre ambientes.",
    '- Os artefatos anteriores à sessão tiveram seus hashes conferidos. Dados e pesos não devem entrar no ZIP principal de entrega.',
    '',
    '## Figuras',
    '',
    '![Curvas dos quatro ajustes](curvas_quatro_ajustes.png)',
    '![Curvas das 12 configurações](curvas_doze_modelos.png)',
    '![Comparações de fases e cortes](comparacao_fases_cortes_validacao.png)',
    '![Efeito dos limiares](comparacao_limiares_validacao.png)',
    '![Escores do modelo escolhido](escores_validacao_modelo_escolhido.png)',
    '',
    '## Próxima etapa depende da revisão do usuário',
    '',
    'A avaliação final dos 12 modelos nos 35 pacientes de teste aguarda revisão destes resultados e da configuração. Depois de autorizada, executar uma única vez e não usar métricas do teste para novo ajuste. Nenhum merge ou fechamento da issue deve preceder a validação final.',
    '',
    'O protocolo, diagnóstico do Windows, decisões e prompt de IAG estão em `docs/EXECUCAO_ISSUE_4_20261117.md`.',
]
destino.write_text('\n'.join(linhas) + '\n', encoding='utf-8')
print(destino)
