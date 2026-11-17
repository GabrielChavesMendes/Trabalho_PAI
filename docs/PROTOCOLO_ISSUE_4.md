# Protocolo de treinamento e avaliacao — Issue #4

Este documento descreve o protocolo final executado em 17/11/2026. Os volumes
NIfTI permanecem fora do repositorio e sao localizados por `PAI_DATASET_DIR`.

## Dados e ausencia de vazamento

Foram usadas exclusivamente as particoes oficiais do CSV: 104 pacientes no
treino (`test=0`), 33 na validacao (`test=2`) e 35 no teste (`test=1`). A
auditoria confirmou IDs disjuntos, rotulos alinhados e as proporcoes 78/26 no
treino, 22/11 na validacao e 23/12 no teste.

Cada amostra fornece `imagens`, `rotulos` e `pacientes`. As probabilidades das
dez variantes de augmentation sao agregadas pela media antes de qualquer
metrica por paciente. O teste nao recebe augmentation.

## Diagnostico do baseline

O comportamento de prever quase todos como pCR=1 nao era causado por troca de
rotulos, vazamento, pesos ImageNet ausentes, falta de gradientes ou erro na
`BCEWithLogitsLoss`. Foram verificados carregamento dos pesos, gradientes da
cabeca, congelamento do backbone, BatchNorm, particoes e alinhamento das
predicoes.

As causas metodologicas encontradas foram:

- `pos_weight=3` desloca para aproximadamente 0,5 a probabilidade constante
  otima nesta prevalencia; a escolha do limiar por F1 favorecia a classe
  positiva e podia premiar a predicao quase constante;
- o vencedor era escolhido por F1, embora a separacao medida por AUC fosse
  fraca;
- checkpoint por menor loss e posterior ranking por outra metrica podiam
  selecionar epocas diferentes;
- a seed era aplicada depois de criar a cabeca aleatoria do modelo;
- a ResNet apresentava sobreajuste precoce: a loss de treino continuava caindo
  enquanto a loss de validacao subia.

A seed passou a ser aplicada antes da criacao do modelo. AUC-ROC e acuracia
balanceada foram adicionadas, o checkpoint passou a seguir a AUC e o limiar e
selecionado por acuracia balanceada apenas na validacao.

## Experimentos controlados

Os ajustes usaram a ResNet50 com tres fases e corte de maior tumor, seed 42.
Foram comparados `pos_weight` 1, 1,73 e 3; backbone congelado versus `layer4`
treinavel; taxas separadas para cabeca/backbone; normalizacao por percentis
versus z-score calculado somente no treino; ultimo bloco, regularizacao e uso
opcional da mascara como filtro. Os resultados estao em
`resultados/ajustes_issue4*`.

As rodadas em `ajustes_issue4_piloto_sem_seed` sao preservadas para auditoria,
mas excluidas das conclusoes porque precedem a correcao da seed. As rodadas em
`ajustes_issue4_checkpoint_loss` tambem sao historicas e nao participam da
selecao final, pois usaram checkpoint por loss.

## Protocolo final congelado

- seed 42; ate 30 epocas; early stopping com paciencia 3 pela AUC de validacao;
- Adam e `weight_decay=1e-4`;
- CNN: treinamento completo, LR `1e-3`, `pos_weight=sqrt(3)`, batch 64;
- MobileNetV2: pesos ImageNet, backbone congelado, LR `1e-4`,
  `pos_weight=sqrt(3)`, batch 64;
- ResNet50: pesos ImageNet, `layer4` e cabeca treinaveis, BatchNorm congelada,
  LR `1e-5` no backbone e `1e-4` na cabeca, `pos_weight=1`, batch 32;
- normalizacao conjunta por amostra nos percentis 1/99;
- uma fase (`post_early`) ou tres (`pre`, `post_early`, `post_late`);
- corte central ou corte da fatia de maior area tumoral;
- limiar por maior acuracia balanceada na validacao;
- ranking final por AUC de validacao, com desempate por acuracia balanceada,
  F1, loss e nome.

O corte de maior tumor requer mascara na inferencia. O melhor modelo com corte
central e mantido como alternativa para cenarios sem mascara.

## Execucao reproduzivel

```bash
PAI_DATASET_DIR=/mnt/d/PAI/dataset/BreastDCEDL_ISPY1_min_crop \
TORCH_HOME=/mnt/d/PAI/torch \
python scripts/executar_grade_issue4.py \
  --epocas 30 --batch 64 --batch-resnet 32 --paciencia 3 \
  --pos-weight 1.7320508075688772 --pos-weight-resnet 1 \
  --lr-cnn 1e-3 --lr-transferencia 1e-4 --lr-backbone 1e-5 \
  --fine-tuning-resnet layer4 --criterio-limiar acuracia_balanceada \
  --criterio-checkpoint auc_roc --tag final-v2 \
  --saida resultados/finais_issue4 --checkpoints checkpoints/finais_issue4
```

Depois de congelar a grade, os 12 checkpoints sao avaliados no teste sem novo
treino e sem recalibrar os limiares:

```bash
PAI_DATASET_DIR=/mnt/d/PAI/dataset/BreastDCEDL_ISPY1_min_crop \
python scripts/avaliar_grade_teste_issue4.py \
  --checkpoints checkpoints/finais_issue4 --tag final-v2 \
  --device cuda --saida resultados/finais_issue4
```

Uma configuracao isolada continua disponivel por
`scripts/executar_experimento.py`, e um checkpoint isolado por
`scripts/avaliar_teste_final.py`.

## Checkpoints

Cada `.pt` registra arquitetura, pesos, configuracao temporal, canais, corte,
normalizacao, hiperparametros, historico, epoca e metadados. Depois da validacao,
o mesmo arquivo recebe o limiar e as metricas de selecao. Os pesos ficam em
`checkpoints/finais_issue4/`, ignorados pelo Git, e devem ser entregues por um
canal separado.

O teste ja havia sido consultado no baseline anterior. Essa limitacao e
preservada no historico; nenhuma metrica de teste foi usada para escolher os
ajustes ou promover um modelo na execucao final.
