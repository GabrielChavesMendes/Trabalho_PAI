# Resultados experimentais — Issue #4

Execucao final em 17/11/2026 com o BreastDCEDL I-SPY1 MinCrop real, 104
pacientes de treino, 33 de validacao e 35 de teste. O teste ja havia sido
consultado no baseline de 15/11/2026; os novos ajustes foram escolhidos somente
com treino e validacao.

## Baseline e diagnostico

O baseline selecionava `resnet50__tres__maior-tumor__seed-42.pt` por F1. No
teste ele previu todos os 35 pacientes como positivos: acuracia 0,343,
sensibilidade 1, especificidade 0 e F1 0,511. Sua AUC de validacao, recalculada
das predicoes salvas, foi 0,479. O melhor AUC do baseline era 0,574 na
MobileNetV2 `post_early`/central.

Nao foram encontrados rotulos trocados, vazamento, falha nos pesos ImageNet ou
ausencia de atualizacao dos parametros. O principal problema foi a combinacao
de separacao fraca, `pos_weight=3`, selecao por F1 e sobreajuste precoce. Tambem
foi corrigida a aplicacao tardia da seed e alinhado o criterio do checkpoint ao
ranking por AUC.

## Ajustes controlados

| Variante ResNet50, tres fases/maior tumor | AUC val. | BAcc val. | F1 val. |
|---|---:|---:|---:|
| `layer4`, `pos_weight=1` | **0,607** | **0,659** | **0,560** |
| `layer4`, `pos_weight=1,73` | 0,587 | 0,591 | 0,421 |
| congelada, `pos_weight=1,73` | 0,574 | 0,545 | 0,333 |
| `layer4`, z-score do treino | 0,558 | 0,545 | 0,513 |
| `layer4`, `pos_weight=3` | 0,554 | 0,614 | 0,444 |
| congelada, `pos_weight=3` | 0,554 | 0,545 | 0,381 |
| congelada, `pos_weight=1` | 0,545 | 0,545 | 0,513 |
| `layer4`, LR menor | 0,545 | 0,523 | 0,500 |

Liberar a `layer4` com `pos_weight=1` produziu a melhor separacao dessa familia.
Z-score, mascara como filtro, somente o ultimo bloco e regularizacao adicional
nao melhoraram a AUC. A loss de treino da melhor ResNet caiu a 0,107 enquanto a
loss de validacao subiu a 1,390 na epoca selecionada, evidencia de sobreajuste.

## Grade final — validacao

| Modelo | Fases | Corte | Ep. | AUC | BAcc | Sens. | Esp. | Prec. | Acc. | F1 | Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | post early | maior tumor | 1 | **0,669** | 0,523 | 0,545 | 0,500 | 0,353 | 0,515 | 0,429 | 0,867 |
| ResNet50 | post early | maior tumor | 1 | 0,640 | 0,614 | 0,545 | 0,682 | 0,462 | 0,636 | 0,500 | 0,645 |
| CNN | tres | maior tumor | 1 | 0,628 | 0,500 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 | 0,873 |
| ResNet50 | tres | maior tumor | 5 | 0,607 | **0,659** | 0,636 | 0,682 | 0,500 | 0,667 | 0,560 | 1,390 |
| CNN | post early | central | 1 | 0,603 | 0,591 | 0,545 | 0,636 | 0,429 | 0,606 | 0,480 | 0,868 |
| CNN | tres | central | 1 | 0,591 | 0,500 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 | 0,873 |
| MobileNetV2 | tres | maior tumor | 11 | 0,533 | 0,545 | 0,636 | 0,455 | 0,368 | 0,515 | 0,467 | 0,891 |
| MobileNetV2 | tres | central | 4 | 0,529 | 0,500 | 1,000 | 0,000 | 0,333 | 0,333 | 0,500 | 0,882 |
| MobileNetV2 | post early | maior tumor | 8 | 0,512 | 0,523 | 0,727 | 0,318 | 0,348 | 0,455 | 0,471 | 0,889 |
| MobileNetV2 | post early | central | 9 | 0,500 | 0,523 | 0,727 | 0,318 | 0,348 | 0,455 | 0,471 | 0,890 |
| ResNet50 | tres | central | 1 | 0,450 | 0,523 | 0,091 | 0,955 | 0,500 | 0,667 | 0,154 | 0,684 |
| ResNet50 | post early | central | 1 | 0,421 | 0,523 | 0,909 | 0,136 | 0,345 | 0,394 | 0,500 | 0,678 |

O melhor modelo por AUC de validacao e
`cnn__post-early__maior-tumor__seed-42__final-v2.pt`. O alvo desejado de AUC
0,70 nao foi atingido: o melhor valor medido foi 0,669. A melhor alternativa sem
mascara e `cnn__post-early__central__seed-42__final-v2.pt`, AUC 0,603.

## Grade final — teste

Os limiares abaixo vieram da validacao e nao foram recalibrados no teste.

| Modelo | Fases | Corte | AUC | BAcc | Sens. | Esp. | Prec. | Acc. | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | post early | maior tumor | 0,533 | 0,536 | 0,333 | 0,739 | 0,400 | 0,600 | 0,364 |
| ResNet50 | post early | maior tumor | 0,543 | 0,531 | 0,583 | 0,478 | 0,368 | 0,514 | 0,452 |
| CNN | tres | maior tumor | 0,540 | 0,500 | 1,000 | 0,000 | 0,343 | 0,343 | 0,511 |
| ResNet50 | tres | maior tumor | 0,634 | 0,596 | 0,583 | 0,609 | 0,438 | 0,600 | 0,500 |
| CNN | post early | central | 0,591 | 0,516 | 0,250 | 0,783 | 0,375 | 0,600 | 0,300 |
| CNN | tres | central | 0,576 | 0,500 | 1,000 | 0,000 | 0,343 | 0,343 | 0,511 |
| MobileNetV2 | tres | maior tumor | 0,475 | 0,531 | 0,583 | 0,478 | 0,368 | 0,514 | 0,452 |
| MobileNetV2 | tres | central | 0,399 | 0,500 | 1,000 | 0,000 | 0,343 | 0,343 | 0,511 |
| MobileNetV2 | post early | maior tumor | 0,594 | 0,486 | 0,667 | 0,304 | 0,333 | 0,429 | 0,444 |
| MobileNetV2 | post early | central | 0,591 | 0,567 | 0,917 | 0,217 | 0,379 | 0,457 | 0,537 |
| ResNet50 | tres | central | 0,732 | 0,603 | 0,250 | 0,957 | 0,750 | 0,714 | 0,375 |
| ResNet50 | post early | central | 0,464 | 0,482 | 0,833 | 0,130 | 0,333 | 0,371 | 0,476 |

A ResNet50 de tres fases/central teve AUC 0,732 no teste, mas somente 0,450 na
validacao. Ela nao pode substituir o modelo selecionado: promove-la apos olhar o
teste seria vazamento. O vencedor formal caiu de AUC 0,669 na validacao para
0,533 no teste, indicando generalizacao limitada e alta incerteza pelo tamanho
reduzido das particoes.

## Comparacoes e erros

Na validacao, uma fase superou tres fases em 4 das 6 comparacoes pareadas; maior
tumor superou o corte central em 5 das 6. Isso nao foi uniforme no teste. Varios
modelos de tres fases ainda colapsaram para todos positivos, mostrando que o
novo ranking evita selecionar esse comportamento, mas nao elimina subajuste.

Os CSVs `*__test__examples.csv` registram paciente, rotulo, probabilidade,
predicao e tipo de erro. No modelo selecionado houve 17 verdadeiros negativos,
6 falsos positivos, 8 falsos negativos e 4 verdadeiros positivos. As matrizes de
confusao e curvas de loss estao em `resultados/finais_issue4/`.

## Tempos, artefatos e conclusao

A grade final consumiu aproximadamente 1.411 s de tempo medido dentro das
epocas. Cada configuracao exportou historico CSV, curva de convergencia, JSON de
validacao/teste, matriz de confusao e predicoes por paciente. As tabelas
maquina-legiveis completas sao `comparacao_validacao.csv` e
`comparacao_teste.csv`.

Os 12 checkpoints existem em `checkpoints/finais_issue4/` e permanecem fora do
Git. Para a tarefa "Modelo do Trabalho", enviar separadamente
`cnn__post-early__maior-tumor__seed-42__final-v2.pt`. Se a nova base nao tiver
mascaras, usar a alternativa central documentada, sem alegar equivalencia de
desempenho.

Os ajustes corrigiram a metodologia e elevaram o melhor AUC de validacao de
0,574 no baseline para 0,669, mas nao sustentam uma alegacao de desempenho
clinico ou de AUC >= 0,70. Estudos futuros devem usar mais seeds ou validacao
cruzada sem reutilizar o teste ja conhecido para ajuste.
