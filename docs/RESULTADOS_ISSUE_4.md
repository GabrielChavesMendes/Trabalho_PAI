# Resultados experimentais — Issue #4

> **Registro histórico:** este arquivo descreve os experimentos iniciais de 15/11/2026 e não representa os resultados finais da Issue #4. Para a coorte mais recente (27 checkpoints, seleção de 12 representantes, avaliação final e figuras para o artigo), consultar [`GRADE_DEFINITIVA_ISSUE_4_20261117.md`](GRADE_DEFINITIVA_ISSUE_4_20261117.md) e [`RELATORIO_FINAL.md`](../resultados/grade_definitiva_issue4_20261117_01/RELATORIO_FINAL.md). Os resultados experimentais de outras branches devem permanecer identificados separadamente.

Execucao local em 15/11/2026 com o BreastDCEDL I-SPY1 MinCrop real: 104
pacientes de treino, 33 de validacao e 35 de teste. O teste foi consultado uma
unica vez, somente depois da escolha formal do checkpoint pela validacao.

## Protocolo executado

- seed 42; NVIDIA GeForce GTX 1660 Ti; PyTorch 2.9.1+cu126;
- fase isolada `post_early` versus `pre + post_early + post_late`;
- corte central versus maior area tumoral;
- dez variacoes por paciente no treino e na validacao: cinco rotacoes, com e sem
  espelhamento; teste sem augmentation;
- normalizacao conjunta por amostra pelos percentis 1/99;
- `BCEWithLogitsLoss(pos_weight=3.0)`, obtido apenas dos 78 negativos e 26
  positivos do treino;
- Adam, `weight_decay=1e-4`, batch 64, no maximo 30 epocas e early stopping de
  cinco epocas pela menor loss de validacao;
- learning rate `1e-3` na CNN e `1e-4` nas cabecas da ResNet50/MobileNetV2;
- pesos ImageNet e backbone congelado nas redes transferidas; a cabeca
  totalmente conectada foi treinada;
- metricas calculadas por paciente depois da media das probabilidades das dez
  variantes; limiar escolhido na validacao por maior F1, com desempate por
  sensibilidade, proximidade de 0,5 e, entre configuracoes, menor loss.

O tempo acumulado dentro das epocas foi 1.847,3 s (30,8 min). O primeiro passe
de cada configuracao inclui a leitura dos NIfTI; os cortes foram mantidos em
cache apenas em memoria durante aquele processo.

| Arquitetura | Tempo total das 4 configuracoes | Media por configuracao |
|---|---:|---:|
| CNN | 714,6 s | 178,7 s |
| MobileNetV2 | 481,5 s | 120,4 s |
| ResNet50 | 651,2 s | 162,8 s |

A MobileNetV2 teve a menor media de tempo. Os melhores checkpoints das redes
transferidas apareceram em geral entre as epocas 1 e 4, enquanto os da CNN
ficaram entre as epocas 5 e 10, embora isso nao tenha produzido boa separacao
das classes.

## Comparacao na validacao

| Modelo | Fases | Corte | Melhor epoca | Loss | Limiar | Sens. | Esp. | Precisao | Acuracia | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | tres | maior tumor | 2 | 1,1539 | 0,45 | 1,000 | 0,045 | 0,344 | 0,364 | 0,512 |
| ResNet50 | post early | central | 2 | 1,1580 | 0,45 | 1,000 | 0,045 | 0,344 | 0,364 | 0,512 |
| ResNet50 | post early | maior tumor | 2 | 1,1624 | 0,45 | 1,000 | 0,045 | 0,344 | 0,364 | 0,512 |
| MobileNetV2 | tres | central | 2 | 1,1726 | 0,45 | 1,000 | 0,045 | 0,344 | 0,364 | 0,512 |
| MobileNetV2 | post early | central | 1 | 1,1387 | 0,45 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| CNN | tres | maior tumor | 5 | 1,1422 | 0,50 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| MobileNetV2 | tres | maior tumor | 1 | 1,1467 | 0,45 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| CNN | tres | central | 5 | 1,1484 | 0,50 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| ResNet50 | tres | central | 2 | 1,1494 | 0,45 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| CNN | post early | maior tumor | 9 | 1,1541 | 0,45 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| CNN | post early | central | 10 | 1,1543 | 0,50 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |
| MobileNetV2 | post early | maior tumor | 4 | 1,1922 | 0,40 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 |

A tabela completa, incluindo tempos, numero de epocas e matrizes `TN/FP/FN/TP`,
esta em `resultados/comparacao_validacao.csv`.

## Selecao e teste reservado

Pela regra fixada, o checkpoint selecionado foi
`resnet50__tres__maior-tumor__seed-42.pt`, melhor epoca 2. Ele foi recarregado em
outro processo, sem novo treino, com limiar 0,45:

| Conjunto | Loss | Sens. | Esp. | Precisao | Acuracia | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teste | 1,1600 | 1,000 | 0,000 | 0,343 | 0,343 | 0,511 | 0 | 23 | 0 | 12 |

Os 12 pacientes pCR=1 foram classificados como positivos (por exemplo,
`ISPY1_1163`, probabilidade 0,525), mas os 23 pacientes pCR=0 tambem foram
classificados como positivos (por exemplo, `ISPY1_1166`, probabilidade 0,536).
Todos os casos, probabilidades e tipos de erro estao no CSV de exemplos do teste.

## Interpretacao e limitacoes

O vencedor formal nao e um classificador discriminativo: o limiar que maximiza
F1 favoreceu prever praticamente todos os pacientes como pCR=1 na validacao e
todos no teste. Portanto, F1 perto de 0,51 nao deve ser interpretado como bom
desempenho. A especificidade zero e a acuracia proxima da prevalencia positiva
confirmam o colapso.

As tres fases produziram menor loss que uma fase em cinco das seis comparacoes
pareadas de modelo/corte, mas isso nao se converteu em separacao confiavel das
classes. A estrategia de corte teve efeito misto. Nas redes transferidas, a loss
de treino continuou caindo enquanto a de validacao piorou cedo; isso caracteriza
sobreajuste. Na CNN, perdas proximas ao baseline ponderado tambem indicam
subajuste em varias configuracoes.

Esses resultados respondem a grade solicitada com dados reais, mas nao sustentam
uma alegacao de utilidade preditiva. Trabalhos futuros devem ser definidos sem
reutilizar este teste para ajuste: mais seeds, validacao cruzada, fine-tuning
parcial controlado, outro criterio de selecao de limiar que penalize o colapso e
analise de calibracao.

O batch, as taxas de aprendizado e a regularizacao foram definidos antes da
grade; somente melhor epoca, configuracao e limiar foram selecionados com a
validacao. Nenhum hiperparametro foi alterado depois da avaliacao do teste.
