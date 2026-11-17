# Execução da Issue #4 — 17/11/2026

> **Atualização posterior:** a execução definitiva concluiu 12 novos treinos,
> consolidou a coorte de 27 e realizou o teste final autorizado, sem alterar
> escolhas por validação. Ver [relatório definitivo](GRADE_DEFINITIVA_ISSUE_4_20261117.md).
> Esta página mantém o registro da primeira sessão, que terminou antes do teste.

**Estado final desta sessão: aguardando revisão do usuário, antes do teste.**
CUDA corrigido; 36 testes aprovados; quatro ajustes e 12 treinamentos da grade
concluídos. Resultados, tabelas completas, curvas e limitações estão na
[revisão anterior ao teste](../../resultados/execucao_issue4_20261117_01/REVISAO_ANTES_TESTE.md).
O teste não foi avaliado. Não houve merge ou fechamento da issue.

Configuração escolhida por AUC de validação: **CNN, `post_early`, maior área
tumoral, época 6**, peso 3,0, batch 4, LR 1e-3, Adam, weight decay 1e-4,
seed 42. Na comparação suplementar, limiar **0,49848052859306335**, AUC 0,6818,
acurácia balanceada 0,7273, F1 0,6316, sensibilidade 0,5455, especificidade
0,9091, precisão 0,7500 e acurácia 0,7879 (TN=20, FP=2, FN=5, TP=6).
Os escores quase constantes e o ajuste na mesma validação exigem cautela:
esses números não comprovam melhora robusta ou generalização.

Os 12 checkpoints originais e as 12 cópias com limiares refinados foram
recarregados estritamente. A inferência de validação do escolhido em processo
novo reproduziu exatamente as probabilidades e as classes (diferença máxima
0,0; zero classificações divergentes). Os hashes dos 55 artefatos anteriores,
dos 16 checkpoints treinados e dos respectivos JSONs originais foram conferidos.
Fontes de treinamento, pré-processamento e CSV permaneceram inalterados.

Manifesto da revisão e comando de teste **preparado, não executado**:
`resultados/execucao_issue4_20261117_01/PRE_FLIGHT_TESTE_PENDENTE.json`.
Não executar a etapa final antes da revisão explícita do usuário. Este trabalho
permanece na branch `feature/felipe-issue4-finalizacao`, com alterações locais
para revisão.

Branch: `feature/felipe-issue4-finalizacao`.
Issue lida integralmente: https://github.com/GabrielChavesMendes/Trabalho_PAI/issues/4
(aberta, sem comentários na consulta). Roteiro: `FINALIZACAO_ISSUE_4.md`, lido integralmente.

## Protocolo registrado antes dos experimentos

- Preservar todos os experimentos anteriores; artefatos desta sessão em
  `resultados/execucao_issue4_20261117_01/`.
- Partições oficiais por paciente: treino `test=0` (104), validação `test=2`
  (33), teste `test=1` (35). Separar antes do aumento ×10, aplicado somente
  a treino/validação. Métricas agregadas por paciente, pela média das probabilidades.
- Mapeamento confirmado na tabela 4 do artigo oficial:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12917191/
- Conferência local: nenhum arquivo necessário ausente; treino com 78 negativos
  e 26 positivos, razão 3,0. Registro em `preflight_dataset.txt`.
- Etapa 1: os quatro candidatos do roteiro, ResNet50, três fases, maior área
  tumoral, seed 42, batch 4, até 30 épocas completas, paciência 5; Adam,
  LR da cabeça 1e-4, weight decay 1e-4, LR de layer4 1e-5 quando liberada.
- Checkpoint pela AUC de validação (empate: menor loss); limiar pela acurácia
  balanceada de validação na grade já implementada de 0,05 a 0,95, passo 0,05.
- Ordenar os quatro candidatos por AUC, acurácia balanceada e F1 de validação,
  nessa ordem. Interpretar sensibilidade, especificidade e confusão, inclusive
  eventual colapso em uma classe. Não comparar losses ponderadas como se fossem
  diretamente equivalentes entre diferentes valores de `pos_weight`.
- Somente após analisar esses resultados, registrar a escolha para a grade
  3 redes × 2 fases (`post_early`/três) × 2 cortes (central/maior tumor).
- Apresentar as validações e a configuração para revisão do usuário antes de
  qualquer avaliação final dos 12 checkpoints no teste. Não executar a etapa 3
  sem essa revisão. Não mesclar na `develop` nem fechar a issue nesta sessão.

## Limitações que acompanharão o relatório

O grupo já consultou os resultados originais do teste antes desta revisão;
portanto, o teste não é completamente cego. Nenhuma nova métrica do teste será
usada para ajustar modelos. A validação tem apenas 33 pacientes, e os dez
aumentos de cada paciente não são observações independentes. A escolha inicial
na ResNet50 não prova melhora nas outras onze combinações. Melhorar desempenho
é uma tentativa; um resultado negativo deve ser relatado.

## Diagnóstico do ambiente

Python inicial: `C:/Python312/python.exe`, 3.12.7. PyTorch inicial:
`2.15.0+cpu`, `torch.version.cuda=None`, CUDA indisponível. O driver 616.92
reconhece a GTX 1650 com 4 GiB. Causa inicial: distribuição de PyTorch somente
CPU, independentemente da presença de CUDA Toolkit no Windows.

Correção concluída: `.venv` local, com acesso às dependências existentes
(`--system-site-packages`), instalando PyTorch `2.9.1+cu126` e torchvision
`0.24.1+cu126` sem remover os pacotes globais. O par e o índice CUDA 12.6
constam nas instruções oficiais: https://pytorch.org/get-started/previous-versions/
CUDA 12.6 disponível, GPU identificada como GTX 1650 SUPER, capacidade 7.5
presente nas arquiteturas da distribuição. Forward/backward CUDA com gradientes
finitos confirmado em `cuda_verificado.txt`. Versões registradas por `pip freeze`
em `ambiente_freeze.txt`; `pip check` não encontrou dependências quebradas.

Para usar o ambiente no terminal PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Também é possível selecionar .venv/Scripts/python.exe no editor.
```

## Estado

- Enunciado original lido integralmente (7 páginas), com extração arquivada em
  `enunciado_extraido.txt`. O leitor PDF é uma ferramenta auxiliar, não uma
  dependência do treinamento.
- Suíte: **27 testes passaram** em 4,69 s (`pytest_corrigido.txt`). A primeira
  tentativa teve 20 testes aprovados e 7 erros de acesso aos temporários do
  Windows (`pytest.txt`); não foram falhas nas asserções. Resolvido com
  `PYTEST_DEBUG_TEMPROOT=.venv/pytest-tmp` e `-o cache_dir=.venv/pytest-cache`.
- Smoke CUDA: as três arquiteturas com 1 e 3 canais, pesos ImageNet oficiais,
  forward/backward, treino sintético e recarga passaram (`smoke_cuda.txt`).
  Esses resultados não são evidência de desempenho experimental.
- Etapa 1 iniciada, sem avaliar teste:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) '.venv/torch-cache'
$env:PAI_NUM_WORKERS = '0'
.\.venv\Scripts\python.exe scripts/ajustar_issue4.py --device cuda --batch 4 --saida resultados/execucao_issue4_20261117_01/ajustes
```

Log completo: `etapa1_treinamento.txt`. Resultados e checkpoints em
`ajustes/<candidato>/resultados/` e `ajustes/<candidato>/checkpoints/`.
## Resultado dos quatro ajustes e decisão anterior à grade

Etapa 1 concluída, com recarga estrita em CPU dos quatro checkpoints em
processo independente, conferência dos 33 IDs de validação, limiares e métricas
recalculadas das probabilidades salvas. Auditoria: `auditoria_ajustes.json/.csv`.

| Candidato | AUC | Ac. bal. | F1 | Sens. | Espec. | Precisão | Acurácia | Loss | Limiar | Época escolhida/total | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| peso 3, fine-tuning layer4 | 0,6033 | 0,6364 | 0,4706 | 0,3636 | 0,9091 | 0,6667 | 0,7273 | 1,9822 | 0,50 | 4/9 | 278,0 |
| peso 1,73, congelada | 0,5620 | 0,5909 | 0,5000 | 0,6364 | 0,5455 | 0,4118 | 0,5758 | 0,8874 | 0,35 | 4/9 | 236,6 |
| peso 1, congelada | 0,5537 | 0,5682 | 0,4828 | 0,6364 | 0,5000 | 0,3889 | 0,5455 | 0,6507 | 0,25 | 3/8 | 227,2 |
| peso 3, congelada | 0,5496 | 0,5682 | 0,2857 | 0,1818 | 0,9545 | 0,6667 | 0,6970 | 1,1756 | 0,55 | 4/9 | 254,4 |

Escolha para a grade: **peso 3,0, fine-tuning de layer4 somente na ResNet50**,
com LR 1e-5 no bloco e 1e-4 na cabeça. Justificativa: maior AUC, critério
primário previamente registrado; também maior acurácia balanceada. O peso 3,0
coincide com a razão de classes calculada exclusivamente no treino.

Não foi escolhido o maior F1: a variante peso 1,73 tem F1 0,50, mas AUC menor.
Na escolhida, TN=20, FP=2, FN=7, TP=4. A sensibilidade é baixa; a acurácia
0,7273 isolada não caracteriza sucesso. Referências triviais na validação:
sempre negativo tem acurácia 22/33=0,6667 e F1=0; sempre positivo tem F1=0,50;
ambos têm acurácia balanceada 0,50 e AUC 0,50 para escore constante.

Há sobreajuste: no fine-tuning, loss de treino foi de 1,0027 para 0,0317,
enquanto a de validação passou de 1,1643 para 2,5815 na época 9. O checkpoint
por AUC é da época 4, com loss de validação 1,9822. A melhora de ordenação dos
escores não representa melhora de loss; calibração não foi avaliada por uma
métrica própria. Não há evidência de ganho
robusto ou generalização com apenas 33 pacientes e uma seed. As losses com
pesos de classe diferentes não são diretamente comparáveis.

### Configuração fixada antes da Etapa 2

- 12 configurações: CNN, MobileNetV2 e ResNet50 × `post_early`/três fases ×
  central/maior tumor. Mesmas partições, aumento e normalização da Etapa 1.
- Batch 4 em todas as redes (validado na GPU de 4 GiB), seed 42, até 30 épocas,
  paciência 5, Adam, weight decay 1e-4, BCEWithLogitsLoss com peso 3,0.
- CNN: LR 1e-3; transferidas: LR 1e-4. MobileNetV2 com backbone congelado;
  ResNet50 com layer4 liberada e LR 1e-5. Demais blocos congelados.
- Checkpoint por AUC de validação, desempate por menor loss; limiar por
  acurácia balanceada de validação, individual para cada checkpoint.
- Escolha da configuração na grade por AUC de validação; os desempates do
  consolidador são acurácia balanceada, sensibilidade, loss e nome.
- Não alterar esses parâmetros usando resultados do teste.

```powershell
.\.venv\Scripts\python.exe scripts/executar_grade_issue4.py --device cuda --batch 4 --pos-weight 3 --finetune-resnet --lr-backbone 1e-5 --lr-cnn 1e-3 --lr-transfer 1e-4 --criterio-checkpoint auc_roc --criterio-limiar acuracia_balanceada --criterio-modelo auc_roc --saida resultados/execucao_issue4_20261117_01/grade --checkpoints checkpoints/execucao_issue4_20261117_01/grade
```

Correção de registro identificada antes da grade: `consolidar_resultados.py`
informava a época de menor loss mesmo para checkpoint por AUC. Agora lê a época
efetiva dos pesos salvos, registra o critério e mantém a época de menor loss em
coluna separada. Não altera treino, pesos, probabilidades ou seleção. Um teste
de regressão cobre épocas divergentes; suíte completa: **28 testes passaram**
(`pytest_apos_correcao_consolidador.txt`).

## Estudo suplementar de granularidade dos limiares (somente validação)

Durante a grade original, a CNN `post_early`/maior tumor apresentou AUC 0,6818,
mas o limiar da grade fixa classificou todos como negativos. As probabilidades
dos 33 pacientes estavam entre **0,4984775841 e 0,4984809875**. Uma grade com
passo 0,05 não distingue esses escores. Diagnóstico preservado em
`diagnostico_granularidade_limiar.json`, sem modificar resultados anteriores.

Decisão registrada antes de aplicar o procedimento às 12 configurações:
comparar separadamente o mesmo critério de acurácia balanceada usando pontos
médios entre probabilidades distintas, um ponto de cada lado dos extremos
(dentro de 0 e 1), e 0,5. Desempate igual ao método original: especificidade,
sensibilidade e proximidade de 0,5. Candidatos usam somente escores de validação;
o objetivo usa somente os rótulos de validação. AUC e pesos não mudam.

O diagnóstico exploratório da CNN produziu limiar 0,4984805286, acurácia
balanceada 0,7273, F1 0,6316, sensibilidade 0,5455 e especificidade 0,9091
(TN=20, FP=2, FN=5, TP=6). **Isso é ajuste de decisão na mesma validação, não
evidência de melhora da rede ou de generalização.** A faixa de escores é quase
constante; o limiar é sensível a pequenas variações numéricas e de distribuição.
Não arredondar limiares ao aplicar os modelos. Não declarar esse modelo robusto
ou calibrado com base nesse resultado.

O estudo é adaptativo: foi motivado por inspeção da validação durante a grade,
não pré-registrado antes de todos os experimentos. Essa decisão e o otimismo
inerente a escolher época, modelo e limiar nos mesmos 33 pacientes devem ser
relatados. O teste permanece sem avaliação e não orientou essa alteração.

Procedimento preparado em `scripts/refinar_limiares_validacao_issue4.py`:
recusa pastas existentes, confere IDs/rótulos contra o CSV de validação e só
inicia se os 12 checkpoints e resultados estiverem presentes. Gera pastas novas,
compara os pesos tensor a tensor e registra a busca dentro do checkpoint.
Mantém os tempos e históricos originais: não representa novos treinamentos.
Não lê NIfTI nem cria carregador de teste. Testes de regressão cobrem escores
estreitos, saturados, repetidos, inválidos e proteção de experimentos anteriores.
Suíte completa após a implementação: **36 testes passaram** em 5,46 s
(`pytest_final_codigo.txt`).

Os resultados da grade original serão conservados em `grade/`, e o estudo
suplementar em `grade_limiares_validacao/`, com checkpoints correspondentes em
pastas diferentes. Ambos serão apresentados para revisão antes do teste.

## Registro de uso de IAG nesta execução

Ferramenta: Codex (OpenAI), terminal PowerShell/Python e conector GitHub para
leitura da issue. Finalidade: diagnóstico do ambiente CUDA, execução dos testes
e scripts existentes, registro e análise de validação. Foram gerados o arquivo
de dependências CUDA, este registro, auxiliares de auditoria, recarga e figuras,
o estudo suplementar de limiares e seus testes. Foi corrigida a informação da
época selecionada no consolidador e adicionado seu teste de regressão.
O código de treinamento e o pré-processamento existentes não foram alterados.
Os integrantes devem revisar a análise e incorporar este registro à declaração
suplementar de IAG da entrega.

Prompt do usuário:

> Leia integralmente `docs/FINALIZACAO_ISSUE_4.md` e a Issue #4 do GitHub.
>
> Quero concluir a Issue #4 cumprindo fielmente o enunciado, tentando melhorar o desempenho dos modelos sem comprometer a metodologia.
>
> Primeiro, diagnostique e corrija o problema de CUDA no meu ambiente Windows. Em seguida, execute os testes e siga as etapas do documento, uma por vez.
>
> Analise os resultados de validação antes de escolher os hiperparâmetros. Não utilize o conjunto de teste para ajustar modelos e não sobrescreva experimentos anteriores.
>
> Registre todos os resultados e checkpoints. Antes de executar a avaliação final dos 12 modelos no teste, apresente os resultados intermediários e a configuração escolhida para revisão.
>
> Não faça merge na `develop` nem feche a Issue #4 antes da validação final.
