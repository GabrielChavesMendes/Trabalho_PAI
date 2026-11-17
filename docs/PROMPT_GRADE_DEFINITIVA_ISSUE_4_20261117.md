# Registro do prompt definitivo — 17/11/2026

Ferramenta: Codex/OpenAI. Finalidade: revisão, planejamento, implementação,
execução e auditoria da última grade e avaliação final. Transcrição do pedido
do usuário, com escapes de transporte removidos e listas compactadas.
As partes geradas por IAG precisam de revisão e validação dos integrantes.

---

# Finalização experimental completa da Issue #4 — PAI

Estamos trabalhando no repositório `GabrielChavesMendes/Trabalho_PAI`, na branch `feature/felipe-issue4-finalizacao`.

Quero concluir definitivamente a parte experimental da Issue #4, realizando **uma última grade de 12 treinamentos**, comparando os resultados de todos os experimentos já executados e, posteriormente, executando a **Etapa 3 de avaliação final no conjunto de teste**.

**Este é o prompt completo e definitivo. Não existem solicitações anteriores pendentes de execução além das atividades descritas aqui.**

Nosso objetivo é terminar com:

- 27 checkpoints treinados e preservados;
- Comparação completa dos 27 na validação;
- Seleção dos 12 melhores representantes, um por combinação de arquitetura, fases e corte;
- Melhor checkpoint de cada arquitetura;
- Melhor checkpoint geral pela validação;
- Avaliação dos 27 checkpoints no conjunto de teste;
- Ranking completo dos 27 e ranking dos 12 representantes no teste;
- Gráficos, métricas, análises e documentação;
- Código testado, organizado e commitado na nossa branch;
- Issue #4 tecnicamente pronta para revisão e integração.

**Não realize merge na `develop` nem feche a Issue #4 nesta execução.**

## 1. Revisão inicial obrigatória

Antes de modificar ou executar qualquer código, examine cuidadosamente o estado atual da branch e os seguintes materiais:

- Issue #4 no GitHub;
- Enunciado oficial do trabalho;
- `docs/FINALIZACAO_ISSUE_4.md`;
- `docs/EXECUCAO_ISSUE_4_20261117.md`;
- `docs/RODADA_FINAL_ISSUE_4_20261117.md`;
- `resultados/execucao_issue4_20261117_01/`;
- `resultados/rodada_final_issue4_20261117_01/`;
- Implementações atuais das três arquiteturas;
- Scripts existentes de treinamento, avaliação, consolidação e seleção;
- Testes automatizados.

Confirme os modelos e experimentos efetivamente existentes, evitando duplicações.

Atualmente, esperamos encontrar:

**Conjunto A — Grade original**: 12 configurações: CNN: 4; ResNet50: 4; MobileNetV2: 4. Todas combinando uma/três fases e corte central/maior tumor.

**Conjunto B — Experimentos adicionais**: três modelos: 1. CNN com learning rate reduzido; 2. CNN com LeakyReLU; 3. ResNet50 com normalização ImageNet. Nenhum desses três apresentou melhora convincente sobre a melhor CNN da grade original.

**Conjunto C — Nova grade**: ainda não executada. Será composta por 12 treinamentos adicionais, definidos abaixo. Total esperado após a execução: **27 checkpoints**. Preserve rigorosamente todos os resultados anteriores.

## 2. Planejar a última grade de 12 treinamentos

Quero investigar **uma melhoria específica para cada arquitetura**, aplicada às quatro configurações de fases e cortes dessa arquitetura. Não quero 36 novos treinamentos nem uma busca extensa de hiperparâmetros.

| Arquitetura | Nova intervenção | Treinamentos |
|---|---|---:|
| CNN | Combater o colapso das ativações | 4 |
| ResNet50 | Melhorar adaptação ao domínio DCE-MRI | 4 |
| MobileNetV2 | Fine-tuning parcial | 4 |
| **Total** | | **12** |

Antes de treinar, documente: a hipótese investigada em cada arquitetura; o comportamento observado nos experimentos anteriores; a intervenção escolhida; os hiperparâmetros novos; os hiperparâmetros mantidos; o comparador mais adequado da execução anterior; os critérios objetivos de avaliação.

Escolha as intervenções com base nos diagnósticos e no código existente, sem usar informações do conjunto de teste.

### 2.1. CNN — Quatro treinamentos

Os experimentos anteriores identificaram evidências de colapso das ativações da CNN: muitos canais da última ReLU ficaram com saída zero; as probabilidades se concentraram em intervalos extremamente pequenos; a variação entre pacientes diminuiu; os gradientes ficaram reduzidos, embora não totalmente nulos.

Já tentamos reduzir learning rate e substituir ReLU por LeakyReLU. Essas tentativas não produziram melhora convincente.

**Agora escolha uma nova intervenção principal, tecnicamente fundamentada, para melhorar a representação aprendida pela CNN.** Investigue preferencialmente uma mudança controlada relacionada a pooling, regularização, normalização interna ou preservação de informação nas camadas intermediárias. Não altere várias características simultaneamente sem necessidade. Não repita simplesmente os experimentos já executados. Mantenha a arquitetura em conformidade com o requisito de pelo menos cinco camadas com parâmetros. Aplique a intervenção escolhida nas quatro configurações de fases e cortes, sem mudar a intervenção após observar os primeiros resultados.

### 2.2. ResNet50 — Quatro treinamentos

Já investigamos: backbone congelado; ajustes de pesos das classes; fine-tuning da `layer4` em configuração específica; padronização com parâmetros ImageNet. Os resultados ainda não demonstraram melhora convincente sobre a melhor CNN.

**Escolha uma única estratégia adicional e fundamentada de adaptação ao domínio DCE-MRI.** Considere as tentativas anteriores e priorize uma intervenção que permita comparação controlada com os checkpoints existentes. Caso escolha fine-tuning ou outra mudança já parcialmente explorada, explique exatamente o que muda em relação às execuções anteriores e por que aplicar essa estratégia às quatro configurações pode ser útil. Preserve os pesos pré-treinados ImageNet e trate adequadamente taxas de aprendizado, camadas congeladas e BatchNorm. Evite mudanças extensas ou muito custosas.

### 2.3. MobileNetV2 — Quatro treinamentos

Essa arquitetura merece especial atenção porque, nos experimentos originais, o backbone permaneceu essencialmente congelado e o classificador foi a principal parte treinada.

**Priorize fine-tuning parcial das últimas camadas do backbone da MobileNetV2.**

Orientações: inicializar com pesos ImageNet; manter as camadas iniciais congeladas; liberar apenas um bloco final apropriado para fine-tuning; treinar também o classificador; utilizar taxas de aprendizado diferenciadas e conservadoras; tratar adequadamente BatchNorm com batch pequeno; evitar alterações excessivas em uma única tentativa; preservar o pré-processamento apropriado e registrar suas características.

O objetivo é testar se a MobileNetV2 melhora quando consegue adaptar parcialmente suas características às imagens médicas. Não é necessário que ela supere a CNN ou a ResNet50. Queremos uma comparação experimental justa e fundamentada.

## 3. Grade exata de experimentos

Para cada arquitetura, executar:

| Configuração | Fases | Corte |
|---|---|---|
| 1 | `post_early` | `central` |
| 2 | `post_early` | `maior_tumor` |
| 3 | `tres` | `central` |
| 4 | `tres` | `maior_tumor` |

Isso representa quatro novas CNNs, quatro novas ResNet50 e quatro novas MobileNetV2. **Total obrigatório: 12 novos treinamentos completos, desde que tecnicamente viáveis.**

Mantenha: partição oficial de treino: 104 pacientes; validação: 33; teste: 35; seed 42; batch 4, salvo impossibilidade técnica justificada; estratégia de aumento de dados exigida no enunciado; seleção do checkpoint por AUC de validação; early stopping; critérios de avaliação por paciente; controle do desbalanceamento; configurações anteriores que não sejam objeto da intervenção.

Utilize como referência o limite anterior de 30 épocas e paciência 5. Não ultrapasse desnecessariamente esses limites. A GPU disponível é uma GTX 1650 SUPER de 4 GB. Tome cuidados para evitar estouro de VRAM.

**Proibição de busca adicional:** não realize novas tentativas, seeds alternativas ou retreinamentos apenas porque um resultado foi inferior ao esperado. Se um treinamento falhar por erro técnico, corrija a falha e continue, sem aproveitar os resultados para otimizar hiperparâmetros.

## 4. Preservar os resultados e checkpoints

Crie um novo diretório exclusivo para esta última grade. Não sobrescreva pesos dos 12 modelos originais, pesos dos três experimentos intermediários, arquivos de validação anteriores, limiares selecionados anteriormente, gráficos, logs, relatórios ou manifestos já existentes.

Cada novo checkpoint deve armazenar ou permitir recuperar: arquitetura; configuração temporal; estratégia de corte; seed; hiperparâmetros; normalização/preprocessamento; época escolhida; pesos; limiar escolhido na validação; métricas de validação; referência ao experimento original. Registre os tempos efetivos de treinamento e todas as condições relevantes para reprodutibilidade.

## 5. Consolidar os 27 checkpoints na validação

Após terminar a última grade, reúna A: 12 checkpoints da madrugada; B: 3 checkpoints dos experimentos intermediários; C: 12 checkpoints desta última grade. Total: **27 checkpoints**.

Gere tabela completa com identificador, arquitetura, fases, corte, grupo, intervenção, AUC-ROC, sensibilidade, especificidade, precisão, acurácia, F1-score, acurácia balanceada, loss, tempo de treinamento, melhor época, limiar e indicadores de estabilidade das probabilidades.

Sensibilidade, especificidade, precisão, acurácia e F1-score são obrigatórias no artigo. AUC-ROC e acurácia balanceada são complementares. Analise convergência, probabilidades, sobreajuste e possível colapso. Não trate os dez aumentos de um paciente como dez pacientes independentes.

## 6. Selecionar os 12 melhores representantes

Esta etapa deve ocorrer **antes de acessar o teste**. Quero exatamente 12 checkpoints, um para cada combinação única de arquitetura, uma/três fases e central/maior tumor. Quatro CNNs, quatro ResNet50, quatro MobileNetV2.

Para cada combinação, compare os checkpoints correspondentes existentes nos três conjuntos e selecione o melhor pela validação. Nunca selecione duas versões da mesma combinação para ocupar duas vagas. Preserve os não selecionados.

Mantenha AUC-ROC como critério primário, mas examine BA, sensibilidade, especificidade, precisão, F1, estabilidade das probabilidades, robustez dos limiares, overfitting, underfitting e colapso/ausência de discriminação real. Diferenças mínimas de AUC não são necessariamente relevantes. Registre previamente os desempates. Não favoreça um checkpoint só por acurácia alta enquanto ignora uma classe.

Identifique e congele melhor CNN, melhor ResNet50, melhor MobileNetV2 e melhor checkpoint geral pela validação.

## 7. Congelar todas as decisões

Antes da Etapa 3: registre os 27 checkpoints, caminhos, arquivos; verifique recarga; registre preprocessamento; congele limiares; registre os 12 representantes, melhor por arquitetura e geral; gere SHA-256; crie manifesto final imutável; registre metodologia e decisões anteriores ao teste.

Inclua evidências de seleção sem resultados desta avaliação final. Se faltar checkpoint, metadados inconsistentes ou recarga falhar, não execute Etapa 3 até resolver problemas técnicos. Não substitua modelos ausentes silenciosamente.

## 8. Executar a Etapa 3 automaticamente

**AUTORIZO EXPRESSAMENTE a execução da Etapa 3 nesta mesma tarefa, sem necessidade de solicitar confirmação adicional, desde que os 12 treinamentos tenham terminado corretamente e todas as verificações do item anterior estejam aprovadas.**

Adapte o avaliador atual para os 27 checkpoints. Execute uma única avaliação final no teste oficial. Exatamente 35 pacientes, sem augmentation, métricas por paciente, mesmas partições, checkpoints congelados, limiar próprio e preprocessamento correto. Nenhum ajuste de hiperparâmetro/limiar, retreinamento, nova época ou substituição de representantes.

Avalie os 27, preservando resultados individuais. O teste não pode escolher novos modelos ou alterações. Se não representante pontuar melhor no teste, mantenha escolhas da validação; pode destacar como observação.

## 9. Rankings dos resultados de teste

Ranking A: todos 27, inclusive não representantes. Ranking B: só os 12 definidos na validação, um por combinação. Ordene principais por **BA no teste**, F1 como primeiro desempate e AUC complementar. Não modifica critério de seleção na validação.

Inclua posição, ID, arquitetura, fases, corte, grupo, sensibilidade, especificidade, precisão, acurácia, F1, BA, AUC, loss. Rankings complementares AUC e F1, sem alterar modelo selecionado.

Destaque vencedor da validação, maior BA observada no teste, maior AUC observada no teste e maior F1 observado no teste. Podem ser diferentes. **Não confunda melhor teste observado com vencedor metodologicamente selecionado pela validação.**

## 10. Comparar validação e teste

Analise: (1) escolhido na validação; (2) comportamento no teste; (3) arquitetura com melhores métricas observadas no teste; (4) mudanças de posições; (5) fracos na validação e altos no teste; (6) fortes na validação que pioraram; (7) ganhos da última grade; (8) Mobile fine-tuning; (9) ativações CNN; (10) adaptação ResNet; (11) uma/três fases; (12) central/maior tumor; (13) tipos de erro; (14) over/underfitting por arquitetura; (15) equilíbrio sensibilidade/especificidade.

Distinga melhor validação, melhor teste, melhor treinamento, melhor configuração por arquitetura e melhor representante de cada combinação. Considere dataset pequeno e não afirme superioridade estatística de diferenças pequenas sem evidências. O grupo consultou teste em outra branch: documente a limitação sem atribuir à nossa branch avaliações que ela não executou.

## 11. Gráficos e resultados para o artigo

Gere tabela validação 27, teste 27, representantes 12, rankings 27/12, comparações dos três grupos/redes, uma/três fases, central/maior tumor, curvas treino/val, matrizes, gráficos de métricas, exemplos corretos/incorretos, tempos e análise de melhorias/limitações.

Use tabelas compactas e figuras reutilizáveis no LaTeX SBC até 12 páginas. Priorize cinco métricas exigidas; AUC/BA enriquecem sem substituí-las. Não invente resultados, imagens, tempos ou métricas.

## 12. Consolidar a documentação da Issue #4

Atualize documentos com arquiteturas, hiperparâmetros e justificativas, fases/cortes, augmentation, desbalanceamento, treino/val/teste, comparação, generalização, limitações, negativos, rankings e reprodução. Documente ajustes e intermediários exigidos pelo professor. Registre IAG, prompts e finalidades, não declare autoria integral. Explicite partes que precisam de revisão humana.

## 13. Organização do repositório

Preserve organização das issues anteriores; separe treino/val/teste; evite duplicações; preserve checkpoints locais; não adicione pesos grandes ou dataset ao Git; scripts reproduzíveis, comandos, caminhos relativos, sem dependências desnecessárias; compatibilidade com interface e inferência; resultados prontos para integração.

Proponha checkpoints para apresentação/inferência: 12 representantes identificados, melhor por arquitetura destacado, vencedor geral pela validação para demonstração principal, demais como histórico. Não coloque 27 pesos no ZIP nem altere checkpoints para reduzir espaço.

## 14. Validações técnicas finais

Execute suíte; confira 27 checkpoints e 12 representantes únicos, quatro por rede; 35 pacientes teste; ausência de vazamento; cálculos das seis métricas e AUC; matrizes; correspondência pesos/limiar/preprocessamento; recarga independente; seleções inalteradas pelo teste; relatórios fiéis aos cálculos. Corrija falhas técnicas sem otimizar pelo teste.

## 15. Commit final na branch

Conclua/verifique, organize arquivos, atualize docs, registre resultados, faça commit na `feature/felipe-issue4-finalizacao`, informe hash e alterações, pendências/limitações. Não merge, não feche issue, não treine após Etapa 3.

## 16. Resposta final obrigatória

**A. Treinamentos:** 12 novos, intervenções, resultados e comparação.
**B. Validação:** comparação 27, 12 representantes, melhores por rede/geral e justificativas.
**C. Teste:** avaliação 27, ranking27 BA, ranking12, melhor observado, diferenças val/teste.
**D. Qualidade:** testes, recarga, consistência, limitações.
**E. GitHub:** commit, arquivos, pendências para integração/fechamento.

## Regra final

Esta é a última rodada. Após os 12, congele pela validação e execute Etapa 3 automaticamente se verificações aprovadas. Não faça ciclos adicionais, seleção pelo teste, merge ou fechamento. Objetivo: Issue #4 executada, testada, documentada e pronta para revisão/integração.
