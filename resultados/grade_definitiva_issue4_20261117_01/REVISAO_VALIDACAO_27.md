# Revisão dos 27 — decisões anteriores ao teste

Registrado em UTC: 2026-11-17T16:21:35.577269+00:00. **O teste ainda não foi executado.** O usuário autorizou avaliação automática somente após terminar a recarga e aprovar o preflight.

A=12 originais; B=3 intermediários; C=12 novos. Cada métrica é calculada em 33 pacientes; média das dez probabilidades de cada paciente. Os 12 originais não incluem as cópias de limiar como novos treinamentos.

## Validação completa

| codigo | arquitetura | fases | corte | auc_roc | sensibilidade | especificidade | precisao | acuracia | f1 | acuracia_balanceada | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A01 | cnn | post_early | central | 0.4298 | 0.0909 | 0.9545 | 0.5000 | 0.6667 | 0.1538 | 0.5227 | 1.1595 |
| A02 | cnn | post_early | maior_tumor | 0.6818 | 0.5455 | 0.9091 | 0.7500 | 0.7879 | 0.6316 | 0.7273 | 1.1563 |
| A03 | cnn | tres | central | 0.5950 | 0.8182 | 0.5000 | 0.4500 | 0.6061 | 0.5806 | 0.6591 | 1.1770 |
| A04 | cnn | tres | maior_tumor | 0.6281 | 0.9091 | 0.4091 | 0.4348 | 0.5758 | 0.5882 | 0.6591 | 1.1882 |
| A05 | mobilenetv2 | post_early | central | 0.4959 | 0.2727 | 0.9545 | 0.7500 | 0.7273 | 0.4000 | 0.6136 | 1.2036 |
| A06 | mobilenetv2 | post_early | maior_tumor | 0.4752 | 0.9091 | 0.2273 | 0.3704 | 0.4545 | 0.5263 | 0.5682 | 1.1983 |
| A07 | mobilenetv2 | tres | central | 0.5207 | 0.2727 | 0.9545 | 0.7500 | 0.7273 | 0.4000 | 0.6136 | 1.2330 |
| A08 | mobilenetv2 | tres | maior_tumor | 0.4917 | 0.4545 | 0.6818 | 0.4167 | 0.6061 | 0.4348 | 0.5682 | 1.2630 |
| A09 | resnet50 | post_early | central | 0.3223 | 1.0000 | 0.0455 | 0.3438 | 0.3636 | 0.5116 | 0.5227 | 3.1194 |
| A10 | resnet50 | post_early | maior_tumor | 0.5248 | 0.2727 | 0.9545 | 0.7500 | 0.7273 | 0.4000 | 0.6136 | 2.3017 |
| A11 | resnet50 | tres | central | 0.4050 | 0.2727 | 0.9091 | 0.6000 | 0.6970 | 0.3750 | 0.5909 | 3.2192 |
| A12 | resnet50 | tres | maior_tumor | 0.6033 | 0.4545 | 0.8636 | 0.6250 | 0.7273 | 0.5263 | 0.6591 | 1.9822 |
| B01 | cnn | post_early | maior_tumor | 0.6694 | 0.7273 | 0.6818 | 0.5333 | 0.6970 | 0.6154 | 0.7045 | 1.1612 |
| B02 | cnn | post_early | maior_tumor | 0.5702 | 0.7273 | 0.5000 | 0.4211 | 0.5758 | 0.5333 | 0.6136 | 1.1598 |
| B03 | resnet50 | tres | maior_tumor | 0.5785 | 0.2727 | 1.0000 | 1.0000 | 0.7576 | 0.4286 | 0.6364 | 1.1432 |
| C01 | cnn | post_early | central | 0.5372 | 0.2727 | 0.9545 | 0.7500 | 0.7273 | 0.4000 | 0.6136 | 1.1579 |
| C02 | cnn | post_early | maior_tumor | 0.5331 | 0.6364 | 0.6364 | 0.4667 | 0.6364 | 0.5385 | 0.6364 | 1.1683 |
| C03 | cnn | tres | central | 0.5909 | 0.9091 | 0.4091 | 0.4348 | 0.5758 | 0.5882 | 0.6591 | 1.2958 |
| C04 | cnn | tres | maior_tumor | 0.5455 | 0.0909 | 1.0000 | 1.0000 | 0.6970 | 0.1667 | 0.5455 | 1.1590 |
| C05 | resnet50 | post_early | central | 0.3471 | 0.0909 | 0.9545 | 0.5000 | 0.6667 | 0.1538 | 0.5227 | 2.3729 |
| C06 | resnet50 | post_early | maior_tumor | 0.5785 | 0.3636 | 1.0000 | 1.0000 | 0.7879 | 0.5333 | 0.6818 | 4.5556 |
| C07 | resnet50 | tres | central | 0.4463 | 0.5455 | 0.5000 | 0.3529 | 0.5152 | 0.4286 | 0.5227 | 7.6089 |
| C08 | resnet50 | tres | maior_tumor | 0.5496 | 0.3636 | 0.8636 | 0.5714 | 0.6970 | 0.4444 | 0.6136 | 1.7532 |
| C09 | mobilenetv2 | post_early | central | 0.5372 | 0.9091 | 0.2727 | 0.3846 | 0.4848 | 0.5405 | 0.5909 | 2.0535 |
| C10 | mobilenetv2 | post_early | maior_tumor | 0.4959 | 1.0000 | 0.1364 | 0.3667 | 0.4242 | 0.5366 | 0.5682 | 1.5984 |
| C11 | mobilenetv2 | tres | central | 0.6488 | 0.8182 | 0.5000 | 0.4500 | 0.6061 | 0.5806 | 0.6591 | 2.4516 |
| C12 | mobilenetv2 | tres | maior_tumor | 0.5661 | 0.4545 | 0.7727 | 0.5000 | 0.6667 | 0.4762 | 0.6136 | 1.5717 |

## Doze representantes

| codigo | arquitetura | fases | corte | auc_roc | acuracia_balanceada | sensibilidade | especificidade |
|---|---|---|---|---|---|---|---|
| A02 | cnn | post_early | maior_tumor | 0.6818 | 0.7273 | 0.5455 | 0.9091 |
| A03 | cnn | tres | central | 0.5950 | 0.6591 | 0.8182 | 0.5000 |
| A04 | cnn | tres | maior_tumor | 0.6281 | 0.6591 | 0.9091 | 0.4091 |
| A12 | resnet50 | tres | maior_tumor | 0.6033 | 0.6591 | 0.4545 | 0.8636 |
| C01 | cnn | post_early | central | 0.5372 | 0.6136 | 0.2727 | 0.9545 |
| C05 | resnet50 | post_early | central | 0.3471 | 0.5227 | 0.0909 | 0.9545 |
| C06 | resnet50 | post_early | maior_tumor | 0.5785 | 0.6818 | 0.3636 | 1.0000 |
| C07 | resnet50 | tres | central | 0.4463 | 0.5227 | 0.5455 | 0.5000 |
| C09 | mobilenetv2 | post_early | central | 0.5372 | 0.5909 | 0.9091 | 0.2727 |
| C10 | mobilenetv2 | post_early | maior_tumor | 0.4959 | 0.5682 | 1.0000 | 0.1364 |
| C11 | mobilenetv2 | tres | central | 0.6488 | 0.6591 | 0.8182 | 0.5000 |
| C12 | mobilenetv2 | tres | maior_tumor | 0.5661 | 0.6136 | 0.4545 | 0.7727 |

AUC primária; desempates BA, robustez ±.01, sensibilidade, F1, loss e ID, conforme plano anterior. Oito representantes C, quatro A, zero B. Algumas combinações têm discriminação baixa: representante significa melhor disponível pelo critério, não modelo clínico adequado.

- Melhor cnn: **A02**, AUC 0.6818.
- Melhor resnet50: **A12**, AUC 0.6033.
- Melhor mobilenetv2: **C11**, AUC 0.6488.

Vencedor geral: **A02**. A CNN original permanece à frente por AUC. Os 27 serão testados, sem substituir essas escolhas depois.

## Comparação controlada C versus A

| codigo | comparador | auc_A | auc_C | delta_auc | ba_A | ba_C | ic95_delta_inf | ic95_delta_sup |
|---|---|---|---|---|---|---|---|---|
| C01 | A01 | 0.4298 | 0.5372 | 0.1074 | 0.5227 | 0.6136 | -0.2191 | 0.4091 |
| C02 | A02 | 0.6818 | 0.5331 | -0.1488 | 0.7273 | 0.6364 | -0.2975 | -0.0041 |
| C03 | A03 | 0.5950 | 0.5909 | -0.0041 | 0.6591 | 0.6591 | -0.2686 | 0.2603 |
| C04 | A04 | 0.6281 | 0.5455 | -0.0826 | 0.6591 | 0.5455 | -0.2687 | 0.1157 |
| C05 | A09 | 0.3223 | 0.3471 | 0.0248 | 0.5227 | 0.5227 | -0.1322 | 0.1570 |
| C06 | A10 | 0.5248 | 0.5785 | 0.0537 | 0.6136 | 0.6818 | -0.0331 | 0.1529 |
| C07 | A11 | 0.4050 | 0.4463 | 0.0413 | 0.5909 | 0.5227 | -0.0909 | 0.1694 |
| C08 | A12 | 0.6033 | 0.5496 | -0.0537 | 0.6591 | 0.6136 | -0.1818 | 0.0702 |
| C09 | A05 | 0.4959 | 0.5372 | 0.0413 | 0.6136 | 0.5909 | -0.0909 | 0.1777 |
| C10 | A06 | 0.4752 | 0.4959 | 0.0207 | 0.5682 | 0.5682 | -0.0786 | 0.1241 |
| C11 | A07 | 0.5207 | 0.6488 | 0.1281 | 0.6136 | 0.6591 | -0.0537 | 0.3347 |
| C12 | A08 | 0.4917 | 0.5661 | 0.0744 | 0.5682 | 0.6136 | -0.0785 | 0.2231 |

Bootstrap pareado estratificado por paciente, 2.000 reamostragens, seed 2026. Intervalos exploratórios: não corrigem seleção de época/modelo nem comparações múltiplas. Os aumentos positivos de AUC não excluíram zero nesses intervalos; não há evidência convincente de superioridade. C02 piorou numericamente; o intervalo negativo isolado também não é uma conclusão confirmatória.

CNN: GroupNorm não superou a referência e não resolveu universalmente o colapso. ResNet: BN congelada não eliminou sobreajuste; loss elevada. Mobile: adaptação parcial favoreceu AUC, principalmente C11, mas continua abaixo da melhor CNN e com grande diferença treino/validação.

Estabilidade das probabilidades, tempos, época e limiares completos estão no CSV/JSON. A avaliação final só começará após a recarga independente dos 27.

- **A01** `cnn__post-early__central__seed-42`: limiar `0.4938041865825653`, época 2; std escores `5.404452625553184e-06`, pior BA ±.01 `0.5`.
- **A02** `cnn__post-early__maior-tumor__seed-42`: limiar `0.49848052859306335`, época 6; std escores `9.827972956761987e-07`, pior BA ±.01 `0.5`.
- **A03** `cnn__tres__central__seed-42`: limiar `0.47722066491842274`, época 5; std escores `0.01360039355020709`, pior BA ±.01 `0.4772727272727273`.
- **A04** `cnn__tres__maior-tumor__seed-42`: limiar `0.4819288983941078`, época 5; std escores `0.07341089660157592`, pior BA ±.01 `0.5681818181818181`.
- **A05** `mobilenetv2__post-early__central__seed-42`: limiar `0.566841208934784`, época 12; std escores `0.0723449072729341`, pior BA ±.01 `0.5227272727272727`.
- **A06** `mobilenetv2__post-early__maior-tumor__seed-42`: limiar `0.42850990891456603`, época 4; std escores `0.04780480881616911`, pior BA ±.01 `0.43181818181818177`.
- **A07** `mobilenetv2__tres__central__seed-42`: limiar `0.5431677505373955`, época 15; std escores `0.08702112091004587`, pior BA ±.01 `0.5681818181818181`.
- **A08** `mobilenetv2__tres__maior-tumor__seed-42`: limiar `0.4458016857504845`, época 13; std escores `0.0785686274936012`, pior BA ±.01 `0.5227272727272727`.
- **A09** `resnet50__post-early__central__seed-42`: limiar `0.010471966626937501`, época 6; std escores `0.26473152694592367`, pior BA ±.01 `0.4318181818181818`.
- **A10** `resnet50__post-early__maior-tumor__seed-42`: limiar `0.756848182529211`, época 5; std escores `0.2857160730362979`, pior BA ±.01 `0.6136363636363636`.
- **A11** `resnet50__tres__central__seed-42`: limiar `0.7724838942289352`, época 6; std escores `0.2962080642155226`, pior BA ±.01 `0.5909090909090908`.
- **A12** `resnet50__tres__maior-tumor__seed-42`: limiar `0.3762133240699768`, época 4; std escores `0.25189159000400074`, pior BA ±.01 `0.6136363636363636`.
- **B01** `e1_cnn_lr_menor`: limiar `0.4913703888654709`, época 3; std escores `7.582695952661856e-05`, pior BA ±.01 `0.5`.
- **B02** `e2_cnn_leaky`: limiar `0.49338732063770296`, época 2; std escores `8.634798344079888e-06`, pior BA ±.01 `0.5`.
- **B03** `e3_resnet_imagenet`: limiar `0.5356386035680771`, época 1; std escores `0.024056802553388375`, pior BA ±.01 `0.5454545454545454`.
- **C01** `c_cnn_post_early_central`: limiar `0.4987002074718475`, época 2; std escores `0.0027808696829657214`, pior BA ±.01 `0.5`.
- **C02** `c_cnn_post_early_maior_tumor`: limiar `0.4819885969161987`, época 14; std escores `0.00023542510597140737`, pior BA ±.01 `0.5`.
- **C03** `c_cnn_tres_central`: limiar `0.3083760604262352`, época 16; std escores `0.15251002434033442`, pior BA ±.01 `0.5`.
- **C04** `c_cnn_tres_maior_tumor`: limiar `0.4945214733481407`, época 9; std escores `2.0654510124831346e-06`, pior BA ±.01 `0.5`.
- **C05** `c_resnet50_post_early_central`: limiar `0.6023294270038605`, época 1; std escores `0.19442355923025936`, pior BA ±.01 `0.4772727272727273`.
- **C06** `c_resnet50_post_early_maior_tumor`: limiar `0.5`, época 2; std escores `0.27664289541330583`, pior BA ±.01 `0.6590909090909092`.
- **C07** `c_resnet50_tres_central`: limiar `0.014535749325091274`, época 6; std escores `0.3331899512225036`, pior BA ±.01 `0.5`.
- **C08** `c_resnet50_tres_maior_tumor`: limiar `0.45322845429182057`, época 1; std escores `0.1928777659826169`, pior BA ±.01 `0.6136363636363636`.
- **C09** `c_mobilenetv2_post_early_central`: limiar `0.09664575625211`, época 11; std escores `0.21536653886530527`, pior BA ±.01 `0.5454545454545454`.
- **C10** `c_mobilenetv2_post_early_maior_tumor`: limiar `0.09836509246379138`, época 6; std escores `0.22393146649023465`, pior BA ±.01 `0.4772727272727273`.
- **C11** `c_mobilenetv2_tres_central`: limiar `0.061849540378898386`, época 15; std escores `0.2425584240164649`, pior BA ±.01 `0.5909090909090908`.
- **C12** `c_mobilenetv2_tres_maior_tumor`: limiar `0.3731868088245392`, época 5; std escores `0.16370973442366243`, pior BA ±.01 `0.5681818181818181`.
