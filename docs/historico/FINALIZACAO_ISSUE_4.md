# Issue #4 — finalização experimental (instruções ao grupo)

> **Integração com a linha paralela (17/11):** a [síntese integrada](../RESULTADOS_FINAIS.md) fixa 12 configurações únicas (8 da coorte principal, 4 da linha do Gustavo). Os seis arquivos externos foram recarregados estruturalmente em CPU; os quatro escolhidos têm identificadores e SHA-256 documentados. **Ainda depende de reexecutar a validação dos 4 pesos externos no dataset real e conferir os 8 pesos locais antes do merge.** Os passos históricos abaixo não substituem esse roteiro.

> **Atualização de 17/11/2026:** a coorte definitiva de 27 checkpoints foi
> treinada/preservada, selecionada pela validação e avaliada uma única vez no teste
> após autorização expressa no prompt definitivo. Resultado e auditoria em
> [GRADE_DEFINITIVA_ISSUE_4_20261117.md](GRADE_DEFINITIVA_ISSUE_4_20261117.md).
> O texto abaixo preserva o roteiro e o estado histórico anterior. Não repetir
> a avaliação para ajustar modelos. Issue aberta, sem merge na develop.

**Branch:** `feature/felipe-issue4-finalizacao` (não altera diretamente `develop`).

Este roteiro mantém o protocolo com **172 pacientes I-SPY1**, partições oficiais
`test=0` treino (104), `test=2` validação (33), `test=1` teste (35).
Aumento ×10 exclusivamente em treino/validação; teste sem aumento.
**Não utilizar os resultados do teste para modificar hiperparâmetros, limiares ou escolher um modelo.**

**Limitação metodológica que deve ser relatada:** o grupo já consultou os resultados
originais do teste (ResNet50, 35 pacientes) antes desta revisão. Portanto, o teste
desse trabalho não é completamente cego para as decisões subsequentes. Nenhum
novo hiperparâmetro deve ser selecionado usando suas métricas; no artigo, indicar
claramente o que foi decidido com base na validação e essa limitação histórica.


## O que foi implementado nesta branch
1. Métricas adicionais de **AUC-ROC** e **acurácia balanceada**, calculadas **por paciente**, além das cinco métricas exigidas pelo professor.
2. Escolha do limiar por F1 **ou** acurácia balanceada usando **apenas a validação**.
3. Escolha da melhor época por Loss, AUC ou acurácia balanceada de **validação**.
4. Ajuste configurável de `pos_weight` sem alterar a divisão oficial dos pacientes.
5. Fine-tuning **opcional** do último bloco `layer4` da ResNet50, com taxa de aprendizado própria (os demais blocos permanecem congelados).
6. Script de **quatro experimentos de ajuste** que grava resultados separados, sem usar o teste.
7. Grade final de **12 configurações** e script próprio para avaliar essas 12 configurações **após o congelamento dos ajustes**.
8. Testes automatizados adicionais para métricas e grade.

As alterações **não modificam automaticamente** os resultados ou checkpoints antigos.
**A melhora de acurácia não é garantida**. Se não houver melhora validada, é legítimo
descrever o resultado negativo com suas limitações no relatório.

## Preparar o Windows (Git Bash ou terminal do VS Code)
Na raiz do projeto, após atualizar a branch:

```bash
git fetch origin
git switch feature/felipe-issue4-finalizacao
git pull origin feature/felipe-issue4-finalizacao
pip install -r requirements.txt
python -m pytest -q
```

O dataset deve existir em `dataset/BreastDCEDL_ISPY1_min_crop/`, com
`dce/` e `mask/`. Sem isso, o treinamento real não roda.

## ETAPA 1 — quatro experimentos de ajuste (somente validação)

```bash
python scripts/ajustar_issue4.py --device cuda --batch 4
```

O comando testa no cenário fixo **ResNet50, três fases, maior área tumoral**:

| Candidato | pos_weight | Backbone |
|---|---:|---|
| peso_3_congelada | 3,0 | congelado |
| peso_1_congelada | 1,0 | congelado |
| peso_1_73_congelada | 1,73 | congelado |
| peso_3_finetune | 3,0 | `layer4` liberada |

Todos usam seed 42 e o mesmo protocolo de fases/corte. Para comparar, usamos
AUC na validação como critério principal, com acurácia balanceada e F1.
A comparação será gravada em
`resultados/ajustes_issue4/comparacao_hiperparametros_validacao.csv`.

Cada variante grava seu checkpoint e curvas em subpastas distintas:
`resultados/ajustes_issue4/<candidato>/`.
**Os arquivos antigos de `resultados/` e `checkpoints/` não são apagados.**

Se ocorrer CUDA out of memory, execute com `--batch 2` ou `--batch 1`
em uma pasta de resultados nova, sem misturar execuções.
O fine-tuning parcial usa mais VRAM que o backbone congelado.

**Importante:** o resultado desses quatro candidatos não demonstra, sozinho,
que o ajuste escolhido será melhor em todas as outras 11 combinações.
É um estudo inicial de hiperparâmetros, e as comparações devem ser justificadas
com evidência de validação, não com base no conjunto de teste.

## ETAPA 2 — grade final: 3 redes × 2 fases × 2 cortes

Depois de examinar a comparação dos ajustes, **escolha e registre
a configuração** antes de executar a grade. A seguir há um exemplo **não definitivo**:

```bash
python scripts/executar_grade_issue4.py --device cuda --batch 8 --pos-weight 1.73 --criterio-checkpoint auc_roc --criterio-limiar acuracia_balanceada
```

Se o candidato selecionado realmente justificar fine-tuning da ResNet50,
adicionar `--finetune-resnet --lr-backbone 1e-5` ao comando. Isso libera
`layer4` somente para a ResNet50, não para CNN ou MobileNetV2.

Para escolher outro `pos_weight`, substitua o `1.73`.
**Não escolha valores com base no teste reservado.**
A configuração de uma fase é `post_early`. As outras configurações usam as
três fases temporais conjuntas. Os dois cortes são `central` e `maior_tumor`.

Arquivos novos, separados dos experimentos originais:
- `resultados/grade_issue4_revisada/*__validation.json`, `*__history.csv`, `*__loss.png`.
- `resultados/grade_issue4_revisada/comparacao_validacao.csv`.
- `resultados/grade_issue4_revisada/modelo_selecionado.json`.
- `checkpoints/grade_issue4_revisada/*.pt` (12 pesos; ignorados no Git).

**Importante:** se modificar parâmetros, use uma pasta de saída e de
checkpoints **nova** para não misturar configurações diferentes que possuem o
mesmo nome. `--refazer` sobrescreve arquivos; utilizar apenas deliberadamente.

## ETAPA 3 — teste final, somente depois de congelar os modelos

**Execute esta etapa uma única vez como avaliação final, sem utilizá-la para
novo ajuste**. Antes, confirme que há 12 arquivos JSON de validação e
12 checkpoints correspondentes. O script se recusa a iniciar se algum faltar.

```bash
python scripts/avaliar_grade_teste_issue4.py --device cuda --confirmar-validacao-fechada
```

O script usa os **12 checkpoints já treinados** e seus limiares de validação,
avalia **somente os 35 pacientes de teste** em cada configuração e salva:

- `resultados/grade_issue4_revisada/teste_final/comparacao_teste.csv`.
- JSON de probabilidades e métricas por paciente.
- Matrizes de confusão e exemplos de acerto/erro de todas as configurações.

**Não é necessário nem permitido retreinar sobre o teste para melhorar a nota.**

## ETAPA 4 — verificação e relatório

1. Confirmar recarga dos checkpoints sem executar treinamento.
2. Registrar a tabela dos quatro ajustes, incluindo hiperparâmetros, AUC,
   acurácia balanceada, F1, sensibilidade, especificidade e Loss.
3. Registrar as 12 combinações e resultados de validação e teste; comparar
   uma fase versus três fases e central versus maior tumor.
4. Usar curvas de Loss, tempos, matrizes de confusão, exemplos corretos/incorretos;
   analisar overfitting, underfitting e velocidade de convergência.
5. **Não declarar vitória** caso o modelo continue classificando quase todas as
   pacientes como positivas. Documentar a limitação: o conjunto de validação tem
   apenas 33 pacientes e AUC perto de 0,5 indica discriminação ruim.
6. Entregar os pesos escolhidos separadamente no Canvas. Não incluir dataset ou
   checkpoints no ZIP principal do trabalho.

### Limite da validação aqui
O código foi alterado no GitHub, **sem executar os treinamentos reais neste
ambiente**, que não possui os volumes NIfTI nem a GPU do grupo. Execute os
testes e envie logs para revisão antes do merge. Não confundir testes sintéticos
com resultados clínicos reais.
