# Grade definitiva e avaliação final da Issue #4

> **Leitura dos códigos de experimento:** A = primeira grade de modelos; B = três investigações intermediárias; C = rodada de intervenções finais. Os números (por exemplo, A12 ou C11) são identificadores internos de execução, não nomes de arquiteturas nem configurações adicionais. A comparação única das 12 configurações, em linguagem descritiva e com resultados integrados, está em [SINTESE_INTEGRADA_ISSUE_4.md](../RESULTADOS_FINAIS.md).

Branch `feature/felipe-issue4-finalizacao`. Registro iniciado antes do treino.
O prompt definitivo de 17/11/2026 autoriza 12 novos treinos e a avaliação automática
dos 27 após seleção e verificações. Substitui a espera por aprovação dos registros
históricos, que serão preservados. Não autoriza merge ou fechamento da issue.

## Inventário e escopo

A: 12 treinos originais, três redes × duas fases × dois cortes. B: três treinos
adicionais (CNN LR menor, CNN LeakyReLU, ResNet ImageNet). C: exatamente 12 novos.
As 12 cópias com limiares refinados representam os mesmos pesos de A, não novos
treinos. Os quatro ajustes iniciais ResNet são histórico de hiperparâmetros,
preservado, fora da coorte A+B+C de 27 solicitada. Não duplicar contagens.

Revisados: Issue #4 aberta, enunciado de sete páginas, os três roteiros anteriores,
artefatos A/B, arquiteturas, datasets, treinamento, consolidação, avaliadores e
testes. A issue remota ainda descreve estado antigo; este registro será a fonte
atual para revisão e integração. Nenhuma postagem externa é necessária.

## Hipóteses e intervenções fixadas antes dos resultados

**CNN:** o diagnóstico em oito pacientes de treino encontrou 87,5% dos canais
da última ReLU totalmente zero e variação reduzida. LR menor e LeakyReLU não
superaram a AUC 0,6818 da referência. Adicionar **GroupNorm com quatro grupos
depois de cada uma das quatro convoluções e antes da ReLU** é uma intervenção
de normalização interna; não muda pooling, dropout, ativação ou cabeça. Mantém
quatro convoluções e duas lineares, com inicialização idêntica à CNN original;
adiciona 480 parâmetros affine. Não usa estatísticas de outras pacientes do batch.
Hipótese: estabilizar as escalas internas pode reduzir canais inativos e preservar
variação entre pacientes. Pode falhar; não é promessa de maior discriminação.
LR 1e-3, dropout 0,3, pooling e inicialização originais permanecem.

**ResNet50:** A já liberava layer4 em todas as quatro configurações, com BN
desse bloco em treino. Houve forte sobreajuste e loss de validação elevada.
B testou backbone congelado com padronização ImageNet, sem ganho convincente.
C mantém **layer4+fc com a mesma LR, mas congela TODAS as BatchNorm**, tanto
affine quanto estatísticas correntes. É uma alteração da política de BN, não
mais profundidade de fine-tuning. Hipótese: evitar estimativas instáveis com
batch 4 preservando características ImageNet e permitindo adaptar convoluções
altas ao domínio. Backbone anterior a layer4 congelado. LR layer4 1e-5, fc 1e-4.

**MobileNetV2:** A treinava somente classifier. C libera **features[17:19]**,
último bloco residual invertido (17) e projeção final (18), além do classificador.
Features[0:17] ficam congeladas. Todas BN ficam congeladas, affine/estatísticas,
com eval reaplicado em cada época. LR final do backbone 1e-5, cabeça 1e-4.
Hipótese: adaptação pequena do extrator pode melhorar a ordenação dos pacientes.
Não haverá mudança de blocos ou LR após observar a primeira configuração.

Para ResNet/Mobile, pesos oficiais ImageNet preservados na inicialização; em
um canal, soma dos filtros RGB original. Todas C mantêm percentis 1/99 por
amostra e clip [0,1], **sem** padronização ImageNet adicional. Assim não se
confundem duas intervenções na comparação pareada com A. O desvio da escala
ImageNet é uma limitação conhecida; B já investigou a padronização isoladamente.
O comparador primário de cada C é a mesma arquitetura/fases/corte de A.

Fundamentação: [GroupNorm no PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html),
[artigo original](https://arxiv.org/abs/1803.08494),
[BatchNorm no PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
São hipóteses locais, não inferências de eficácia clínica a partir dessas fontes.

## Protocolo comum e seleção prévia

104 treino (78/26), 33 validação (22/11), 35 teste (23/12), partições CSV oficiais
0/2/1. Seed 42, batch 4, Adam, weight decay 1e-4, BCEWithLogitsLoss peso positivo
3,0 exclusivamente da proporção do treino, FP32, até 30 épocas completas,
early stopping paciência 5. Melhor época por AUC validação; empate menor loss.
Dez aumentos determinísticos em treino/validação (-20,-10,0,10,20 graus ×
espelhamento); teste sem aumento. Métricas de validação usam média das dez
probabilidades por paciente, nunca dez observações independentes. Loss usa
amostras aumentadas como no protocolo anterior e não a probabilidade média.

Limiar por BA na validação entre pontos médios dos escores, extremos e 0,5;
desempate por especificidade, sensibilidade e proximidade de 0,5. Os limiares A/B
não serão refeitos. Registrar todos os algarismos, sem arredondar na inferência.
As métricas por época nos históricos e logs usam limiar 0,5; AUC independe dele.
As tabelas finais usam o limiar de validação do checkpoint selecionado. Portanto,
o F1 do log da melhor época pode diferir do F1 final sem mudança de pesos.

Representantes e vencedores: AUC decrescente; em empate numérico exato,
BA decrescente, pior BA sob perturbação do limiar ±0,01 decrescente, sensibilidade
decrescente, F1 decrescente, loss crescente, ID lexicográfico. Esse critério
determinístico mantém AUC como primária; diferenças mínimas não serão chamadas
de melhora relevante. Discussão inclui sensibilidade/especificidade, faixa/std
dos escores, instabilidade, loss e over/underfitting. Sem corte artificial de
acurácia e sem excluir modelos só por resultado negativo.

Selecionar exatamente um por família/fases/corte: CNN inclui ReLU, Leaky e GN.
Congelar os 12 representantes, melhor por arquitetura e geral antes de ler
qualquer nova predição de teste. Avaliar **todos 27**, inclusive não representantes.
Ranking de teste é descritivo: BA, F1, AUC, ID; complementares AUC e F1. Nunca
substitui a escolha da validação, mesmo se outro modelo pontuar melhor.

## Preservação, verificações e limitações

Saídas exclusivas em `resultados/grade_definitiva_issue4_20261117_01/` e pesos
em `checkpoints/grade_definitiva_issue4_20261117_01/`. Plano imutável registra
hiperparâmetros, comparadores, data, versões, hashes do CSV e das fontes de treino.
Hashes anteriores serão conferidos depois. Recarga estrita e reprodução das
probabilidades de validação em processo independente precedem o teste; qualquer
inconsistência bloqueia o teste. Manifesto congelado não será regenerado após teste.

A validação pequena e reutilizada para época, limiar e modelo cria otimismo de
seleção. Uma seed, uma partição; não há prova de superioridade estatística nem
validação clínica. O grupo consultou resultados do teste em outra branch antes
desta revisão. Esta branch não será creditada por essa avaliação histórica;
o teste atual não pode ser chamado de completamente cego. Após o teste atual
não haverá mais treinamentos ou ajustes.
Além disso, conforme o protocolo anterior, a validação agrega dez transformações
e o teste usa uma imagem sem aumento. Essa diferença pode deslocar os escores
e afetar limiares frágeis; não será corrigida usando o teste. Variação pequena de
probabilidade não demonstra calibração; BCE ponderada também não garante que
o valor da sigmoid seja uma probabilidade clínica calibrada.

## Uso de IAG e revisão humana

Codex/OpenAI auxilia concepção das intervenções, implementação, execução,
auditoria, figuras e redação. Não declarar autoria integral dos integrantes.
Os integrantes devem revisar todo o código e interpretação, compreender as
intervenções e os cálculos, validar compatibilidade da interface/Grad-CAM e
revisar o texto e a declaração suplementar de IAG antes de qualquer entrega.
Integração em Python único, relatório SBC até 12 páginas, nomes/matrículas e
ZIP até 10 MB sem dataset/pesos permanecem tarefas da entrega integrada.

Prompt registrado em [PROMPT_GRADE_DEFINITIVA_ISSUE_4_20261117.md](PROMPT_GRADE_DEFINITIVA_ISSUE_4_20261117.md).
Na primeira verificação, dois testes novos omitiam campos obrigatórios ao criar
uma configuração sintética; foi corrigida a preparação dos testes, antes de
qualquer treino. A suíte passou com 54 testes antes do treino e 58 na revisão
do avaliador. O desenvolvimento dos relatórios e verificações em CPU ocorreram
durante parte dos treinos em GPU; tempos registrados são os efetivamente
observados nesta máquina, não um benchmark isolado das arquiteturas.

## Comandos reproduzíveis

Na raiz do projeto, em PowerShell. Estes comandos documentam o protocolo;
as saídas existentes são protegidas e não devem ser apagadas para repetir.
Para uma reprodução independente, usar uma nova pasta e registrar o motivo;
o estudo atual não autoriza mais rodadas de otimização.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_HOME = Join-Path (Get-Location) '.venv/torch-cache'
$env:PYTEST_DEBUG_TEMPROOT = Join-Path (Get-Location) '.venv/pytest-tmp'
.\.venv\Scripts\python.exe scripts/preparar_ultima_grade_issue4.py --base resultados/grade_definitiva_issue4_20261117_01
.\.venv\Scripts\python.exe scripts/executar_ultima_grade_issue4.py --base resultados/grade_definitiva_issue4_20261117_01 --checkpoints checkpoints/grade_definitiva_issue4_20261117_01 --device cuda
.\.venv\Scripts\python.exe scripts/consolidar_coorte_issue4.py --base resultados/grade_definitiva_issue4_20261117_01
.\.venv\Scripts\python.exe scripts/documentar_resultados_issue4.py --base resultados/grade_definitiva_issue4_20261117_01 --somente-validacao
.\.venv\Scripts\python.exe scripts/verificar_coorte_issue4.py --base resultados/grade_definitiva_issue4_20261117_01
.\.venv\Scripts\python.exe scripts/consolidar_coorte_issue4.py --base resultados/grade_definitiva_issue4_20261117_01 --congelar
.\.venv\Scripts\python.exe scripts/avaliar_manifesto_issue4.py --manifesto resultados/grade_definitiva_issue4_20261117_01/manifesto_final_27.json --saida resultados/grade_definitiva_issue4_20261117_01/teste_final --device cuda --confirmar-validacao-fechada
.\.venv\Scripts\python.exe scripts/relatorio_final_issue4.py --base resultados/grade_definitiva_issue4_20261117_01
.\.venv\Scripts\python.exe scripts/relatorio_final_issue4.py --base resultados/grade_definitiva_issue4_20261117_01 --somente-figura-limiares
.\.venv\Scripts\python.exe scripts/relatorio_final_issue4.py --base resultados/grade_definitiva_issue4_20261117_01 --somente-ranking-legivel
.\.venv\Scripts\python.exe scripts/documentar_resultados_issue4.py --base resultados/grade_definitiva_issue4_20261117_01
.\.venv\Scripts\python.exe scripts/auditar_finalizacao_issue4.py --base resultados/grade_definitiva_issue4_20261117_01
.\.venv\Scripts\python.exe -m pytest -q -o cache_dir=.venv/pytest-cache
```

O avaliador aceita o protocolo legado de 12–15 modelos e o protocolo definitivo
versionado de **27, com A=12, B=3, C=12**. Antes de abrir imagens de teste,
verifica pesos, metadados, hashes, recargas, limiares, cobertura e seleções
recalculadas somente dos arquivos de validação. A saída final existente bloqueia
repetições e sobrescritas. Nunca usar `--refazer` ou remover a saída para procurar
um teste mais favorável.

## Contrato para integração e inferência

Usar `modulos.rodada_final_issue4.carregar_modelo_registrado(caminho, dispositivo)`:
retorna modelo em eval, checkpoint completo e normalização adicional. Suporta
as três redes originais e as variantes `cnn_leaky` e `cnn_groupnorm`.
Reconstrução com `strict=True`, sem baixar pesos e sem treinar. O módulo antigo
de redes permanece disponível para as Issues anteriores.

Para entrada nova: extrair a fase/corte registrados em `configuracao`, normalizar
pelos percentis por amostra com `DatasetDCE`, aplicar `normalizar_adicional`
conforme o retorno da recarga, fornecer `[N,C,256,256]`, aplicar sigmoid nos
logits e comparar com `checkpoint['selecao_validacao']['limiar']` completo.
Não substituir por 0,5, não arredondar o limiar, não aumentar pacientes novos.
Máscara seleciona o corte de maior área, sem entrar como canal da rede.
Qualquer interface/Grad-CAM deve preservar esse contrato e ser validada pelos
integrantes antes da integração. A escolha demonstrada deve permanecer a da
validação, inclusive se um modelo diferente pontuar melhor no teste.

O avaliador reutiliza o cache de imagens entre checkpoints com as mesmas
fases/corte/normalização/batch/seed. Não reutiliza probabilidades de outro modelo.
Assim, o tempo registrado de teste inclui leitura inicial quando necessária e
não é um benchmark isolado de latência. O cache é criado somente após o preflight.

## Resultado e decisões de validação, antes do teste

**12 novos treinamentos concluídos**, uma seed, batch 4, sem falhas de CUDA,
sem repetição ou mudança de intervenção. A consolidação auditou os 27 históricos
e confirmou que cada época selecionada coincide com a maior AUC/menor loss.
Revisão humana legível registrada em UTC antes do teste:
[REVISAO_VALIDACAO_27.md](../../resultados/grade_definitiva_issue4_20261117_01/REVISAO_VALIDACAO_27.md).
Contém todas as cinco métricas obrigatórias, AUC/BA e comparação pareada C/A.

| Melhor por arquitetura | Código | Fases/corte | AUC val. | BA val. | Época/total |
|---|---|---|---:|---:|---:|
| CNN original | A02 | post_early / maior tumor | 0,6818 | 0,7273 | 6/11 |
| ResNet50 original | A12 | três / maior tumor | 0,6033 | 0,6591 | 4/9 |
| MobileNetV2 com fine-tuning | C11 | três / central | 0,6488 | 0,6591 | 15/20 |

**Vencedor geral: A02**, critério AUC. Representantes: CNN C01/A02/A03/A04;
ResNet C05/C06/C07/A12; Mobile C09/C10/C11/C12. A ordem em cada arquitetura
é uma fase central, uma fase maior tumor, três central, três maior tumor.
São **oito novos e quatro originais**; nenhum B. Isso não exclui B do teste:
os 27 serão avaliados, e os 27 pesos permanecem preservados.

CNN GroupNorm não superou a CNN original nas combinações de maior interesse:
C02 AUC 0,5331 contra 0,6818; C03 0,5909 contra 0,5950; C04 0,5455 contra
0,6281. C01 subiu de 0,4298 a 0,5372, mas ainda discrimina pouco. Amplitude
de escores e atividade interna devem ser interpretadas junto da AUC: reduzir
canais inativos não garante ordenação melhor entre pCR=0/1.

ResNet com BN congelada teve três ganhos numéricos, porém AUCs baixas e forte
sobreajuste. C06 foi a melhor nova ResNet, AUC 0,5785 e BA 0,6818, sensibilidade
0,3636 e especificidade 1,0. Sua loss de validação foi 4,5556. A melhor ResNet
anterior, A12, continua superior por AUC e representa três fases/maior tumor.

MobileNet melhorou AUC nas quatro células. C11 subiu de 0,5207 para 0,6488 e
BA de 0,6136 para 0,6591; sensibilidade 0,8182 e especificidade 0,5. A loss de
validação de 2,4516 versus loss de treino na época 15 de 0,1007 mostra grande
diferença e sobreajuste. A adaptação foi útil numericamente, sem superar A02.

Os intervalos bootstrap pareados dos ganhos positivos incluem zero. São
exploratórios, sem corrigir reutilização da validação, seleção ou multiplicidade.
Logo, não comprovam superioridade estatística. Representantes C05/C07 e outros
casos de AUC perto/abaixo de 0,5 são apenas os melhores disponíveis na célula,
não classificadores clinicamente adequados. Não se favoreceu acurácia majoritária
nem se criou um alvo artificial de desempenho. A avaliação final está autorizada
após verificar recarga, hashes, limiares e todas essas decisões.

## Avaliação final concluída e auditada

**Etapa 3 executada uma vez: 27 checkpoints × os mesmos 35 pacientes**,
sem augmentation, ajuste de limiar, época nova ou retreinamento. São 35 pacientes
independentes, não 945. Manifesto anterior à abertura do teste:
`b0efd0f922e46ec17f0bd3e2e1e31902d4bf7156e3423f41a56c07d3c530fe40`.
O arquivo utilizado no teste coincide com o congelado. Seleções e pesos não mudaram.

As probabilidades de validação dos 27 foram reproduzidas com diferença máxima
**0,0**, zero classes divergentes e métricas concordantes. Após o teste, nova
recarga estrita CPU dos 27, sem inferir novamente o teste, aprovou a integridade.
Os **266 arquivos anteriores** permanecem byte a byte idênticos. As métricas
e confusões de **81 relatórios** (treino/validação/teste dos 27) foram recalculadas
com scikit-learn. As partições são disjuntas e cada teste tem exatamente 35 IDs.

A grade C teve 148 épocas completas no total, todas dentro de 30/paciência 5;
tempo somado de épocas **2.846,912 s (47 min 26,912 s)**. Inclui validação por
época, não a auditoria posterior. Pico de tensores alocados pelo PyTorch:
614.225.408 bytes, cerca de 0,57 GiB; não inclui desktop/contexto do driver.
Não houve OOM nem redução de batch.

### Melhores por validação e seu comportamento no teste

| Escolha congelada | AUC val. | AUC teste | Sens. teste | Espec. teste | Precisão | Acurácia | F1 | BA teste |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A02 CNN, 1F/maior tumor — geral | 0,6818 | 0,5109 | 0,1667 | 0,8261 | 0,3333 | 0,6000 | 0,2222 | 0,4964 |
| A12 ResNet50, 3F/maior tumor | 0,6033 | 0,5761 | 0,5833 | 0,6522 | 0,4667 | 0,6286 | 0,5185 | 0,6178 |
| C11 MobileNetV2, 3F/central | 0,6488 | 0,5652 | 0,6667 | 0,3478 | 0,3478 | 0,4571 | 0,4571 | 0,5072 |

**Maior resultado observado no teste: C07, ResNet50, três fases/central, BN
congelada.** AUC **0,7101**, BA **0,6377**, sensibilidade **0,6667**, especificidade
**0,6087**, precisão **0,4706**, acurácia **0,6286**, F1 **0,5517**, loss **3,8854**.
Confusão TN=14, FP=9, FN=4, TP=8. Foi o maior em BA, AUC e F1, sem empate nesses
máximos. C07 já representava sua célula antes do teste, mas tinha AUC de validação
0,4463 e não era o melhor ResNet nem o melhor geral. **Não substituir A02 ou A12
por C07 na seleção metodológica.** O resultado é observacional e sujeito a
variabilidade e otimismo por observar o máximo entre 27 modelos.

### Discussão dos resultados e das 15 questões de comparação

1. **Vencedor da validação e generalização:** A02 permanece vencedor formal.
   Sua AUC caiu 0,6818→0,5109 e BA 0,7273→0,4964. No teste foram FN=10/12 e FP=4/23.
   A queda de AUC, que independe do limiar, demonstra que o problema não se limita
   à transferência de um limiar sensível. A discriminação não se confirmou.
2. **Arquitetura no teste:** a maior observação foi ResNet50/C07. Entre os três
   melhores fixados por validação, A12 teve maior BA no teste (0,6178), mas isso
   não muda o vencedor geral. Entre os quatro representantes de cada rede,
   BA média: ResNet 0,5942, Mobile 0,5344, CNN 0,4746; médias descritivas apenas.
3. **Mudanças de posições:** correlação de Spearman validação/teste entre os 27:
   AUC 0,1424; BA −0,2355; F1 0,3069. Há baixa estabilidade. Os modelos compartilham
   pacientes, pesos iniciais e critérios, portanto essas correlações não são
   testes independentes de superioridade.
4. **Fraco na validação, alto no teste:** C07 passou AUC 0,4463→0,7101. A09 teve
   sensibilidade 1,0 no teste, porém especificidade 0,1304 e AUC 0,4457: detectar
   todos os positivos sozinho não caracteriza bom modelo. Não promover nenhum
   checkpoint após examinar essas métricas.
5. **Forte na validação, pior no teste:** A02, B01 (AUC 0,6694→0,4710) e C11
   (0,6488→0,5652) mostram que ganhos na validação reutilizada podem não generalizar.
   Nenhuma tentativa de corrigir isso consultando novamente o teste foi feita.
6. **Ganho global da última grade:** A→C, médias das 12 células pareadas:
   AUC validação 0,5145→0,5313; BA validação 0,6098→0,6023; AUC teste
   0,5186→0,5385; BA teste 0,5036→0,5180. Há ganhos pequenos/mistos e resultados
   negativos. B tem só três casos, dois deles CNN na mesma célula, e sua média
   não é uma comparação fatorial equilibrada com A/C.
7. **Mobile com fine-tuning:** AUC aumentou nas quatro células da validação;
   C11 foi a melhor, com delta AUC +0,1281 sobre A07. No teste, C11 fez BA 0,5072,
   com 15 FP e quatro FN. Há alguma melhora exploratória de representação, mas
   não comprovação de melhora robusta. CNN e ResNet também têm falhas importantes.
8. **CNN e ativações:** na mesma sonda de oito pacientes de treino, a fração de
   canais totalmente zero da última ReLU convolucional caiu nas quatro células:
   67,19%→6,25%; 87,50%→43,75%; 63,28%→0%; 64,84%→56,25%. Entretanto, a cabeça
   ficou mais inativa. Em C04, os 32 neurônios da ReLU da cabeça ficaram zero
   na sonda e a saída foi constante, mesmo com variação nas features. Isso é
   evidência de melhora parcial das ativações intermediárias, sem restaurar
   discriminação de forma consistente. Não generalizar a sonda para todos os
   pacientes. O teste não motivou uma nova intervenção.
9. **ResNet e BN:** três células subiram numericamente na validação, uma caiu;
   a melhor anterior A12 permaneceu escolhida. C07 teve a maior observação no
   teste, mas forte overfitting e loss alta. Congelar BN não resolve sozinho
   a adaptação DCE-MRI nem demonstra boa calibração.
10. **Uma ou três fases:** entre representantes, AUC média teste 1F=0,5157 e
    3F=0,5918; BA 0,5130 e 0,5559. Nos contrastes dentro de grade/rede, três
    fases aumentaram a AUC teste da CNN e ResNet em A e C, mas a Mobile A piorou
    (delta −0,0960) e a Mobile C subiu pouco (+0,0272), com BA menor. Não há
    benefício universal nem prova estatística de superioridade de três fases.
11. **Central ou maior tumor:** representantes tiveram AUC média teste central
    0,5459 e maior tumor 0,5616; BA 0,5269 e 0,5420. Na ResNet C, maior tumor
    piorou a AUC teste em 0,0507, apesar da maior AUC média na validação. O
    resultado depende de rede/intervenção. Usar os contrastes pareados fornecidos,
    não apenas comparar dois melhores modelos escolhidos em grupos distintos.
12. **Tipos de erro:** nos quatro representantes CNN, cada modelo deixou passar
    dez dos 12 positivos. Na Mobile houve mais FP (média 13,75/modelo) que FN
    (4/modelo). Na ResNet houve média sete FN e 5,25 FP/modelo. São descrições
    de previsões nos mesmos 35 pacientes, não novas amostras independentes.
13. **Over/underfitting:** CNNs de uma fase e C04 apresentaram saída quase constante
    e loss próxima da referência ponderada constante; há underfitting/colapso.
    C03 aprendeu no treino, mas divergiu na validação. ResNets e Mobile parcial
    frequentemente tiveram loss de treino muito baixa e val/teste elevadas:
    overfitting. Não confundir aprendizado de treino com generalização.
14. **Melhor treino:** oito checkpoints empataram em AUC de treino 1,0 na avaliação
    em modo eval do checkpoint já escolhido. Isso não os torna vencedores.
    Treino AUC 1,0, val baixa e teste variável são evidência de grande gap.
    Não foram selecionadas novas épocas pelo treino ou teste.
15. **Equilíbrio de classes:** C07 teve 0,6667/0,6087 e A12 0,5833/0,6522 de
    sensibilidade/especificidade no teste. São os exemplos mais equilibrados
    entre os de BA mais alta, mas as taxas ainda são modestas. Acurácia 0,6857
    de C04 acompanha sensibilidade só 0,0833; não deve ser interpretada como
    superioridade. Sempre negativo teria acurácia 23/35=0,6571 e BA 0,5.

Os resultados não sustentam uso clínico ou superioridade confirmatória. Uma
paciente positiva muda a sensibilidade do teste em 8,33 pontos percentuais.
Pequenas diferenças de ranking são incertas; reutilização da validação,
comparações múltiplas, uma seed, uma partição e consulta histórica do teste em
outra branch limitam a interpretação. Não atribuir a esta execução a avaliação
antiga do grupo; nesta rodada houve exatamente a avaliação final documentada.

## Artefatos finais, apresentação e revisão dos integrantes

- [Relatório completo, rankings e mapa dos checkpoints](../../resultados/grade_definitiva_issue4_20261117_01/RELATORIO_FINAL.md).
- `comparacao_validacao_27.csv`: métricas, intervenção, época, tempo, limiar e estabilidade.
- `artigo/comparacao_validacao_teste_27.csv`: treino/validação/teste, confusões e mudanças de posições.
- `artigo/ranking_teste_27_acuracia_balanceada.csv` e `ranking_teste_12_acuracia_balanceada.csv`.
- Rankings complementares AUC e F1, contrastes fases/cortes e comparações por grupo/rede.
- Matrizes individuais dos 27 e painel dos 12; exemplos reais TN/TP/FP/FN de A02;
  curvas por época, topologia, ROC e tempos; tabelas LaTeX dos representantes.
- `manifesto_final_27.json/.sha256`, seleção prévia, recargas e `auditoria_final.json`.
- Figura de probabilidades revisada para legibilidade em PNG e PDF, sem remover
  a primeira versão: `artigo/figuras_revisadas/probabilidades_e_limiares.*`.
- Ranking gráfico revisado com os códigos únicos A01…C12, distinguindo B01/B02,
  em `artigo/figuras_revisadas/ranking_teste_27.*` (PNG e PDF).

Para a apresentação, manter **A02 na demonstração principal**, **A12 e C11**
como melhores por arquitetura, e os 12 representantes para comparação. Mostrar
C07 como observação do teste e discutir a inversão de ranking, sem promovê-lo
a vencedor. Os demais permanecem como histórico. Não modificar a inferência
padrão por causa do resultado de teste. Nenhum peso ou dataset entra no Git/ZIP.

Os integrantes precisam revisar a interpretação, verificar a interface/Grad-CAM,
integrar em um Python único, finalizar artigo SBC e declaração IAG e preparar a
entrega limitada a 10 MB. **Não houve merge na develop nem fechamento da Issue #4.**

## Verificação final do código

**58 testes passaram em 6,65 s** depois de terminar a avaliação e os relatórios
(`testes_finais.txt`). `compileall` aprovou módulos/scripts/testes e `git diff
--check` não apontou erros de whitespace. A suíte cobre métricas, agrupamento
por paciente, augmentation/partições, cargas estritas, compatibilidade anterior,
GroupNorm, BN imutável, grupos de LR, cobertura/seleção, bloqueio de teste sem
flag, adulteração de pesos e proteção de saídas existentes. Nenhum teste
automatizado executou novamente a inferência clínica do conjunto de teste.

Não foram adicionadas dependências. CUDA: torch 2.9.1+cu126 / torchvision
0.24.1+cu126 na `.venv`, GTX 1650 SUPER. A receita CUDA existente permanece;
o cache `/Microsoft/`, ambiente virtual, dataset e pesos continuam ignorados.
Atributos de fim de linha preservam os bytes das fontes e dos artefatos cujo
SHA foi registrado, inclusive ao clonar no Windows. Os arquivos de resultados
anteriores não foram reformatados ou regenerados.
