# Síntese integrada — 12 combinações, três arquiteturas

**Objetivo:** reunir em **uma única tabela de resultados** os melhores valores de **AUC-ROC de validação relatados** pelas duas linhas de experimentação da Issue #4. Cada combinação de arquitetura, fases e estratégia de corte aparece uma única vez. Este documento é uma **comparação exploratória pós-execução**, não substitui o manifesto congelado dos 27 experimentos nem altera as suas escolhas anteriores ao teste.

## Tabela consolidada de validação

| Arquitetura | Fases | Corte | Variante/ID | AUC val. | BA val. | F1 val. | AUC teste relatada | Disponibilidade de pesos |
|---|---|---|---|---:|---:|---:|---:|---|
| CNN | 1 | central | `cnn__post-early__central__seed-42__final-v2` | 0.6033 | 0.5909 | 0.4800 | 0.5906 | integrar e validar recarga |
| CNN | 1 | maior tumor | `cnn__post-early__maior-tumor__seed-42` | 0.6818 | 0.7273 | 0.6316 | 0.5109 | na coorte local (Git ignora .pt) |
| CNN | 3 | central | `cnn__tres__central__seed-42` | 0.5950 | 0.6591 | 0.5806 | 0.5616 | na coorte local (Git ignora .pt) |
| CNN | 3 | maior tumor | `cnn__tres__maior-tumor__seed-42` | 0.6281 | 0.6591 | 0.5882 | 0.5942 | na coorte local (Git ignora .pt) |
| ResNet50 | 1 | central | `resnet50__post-early__central__seed-42__final-v2` | 0.4215 | 0.5227 | 0.5000 | 0.4638 | integrar e validar recarga |
| ResNet50 | 1 | maior tumor | `resnet50__post-early__maior-tumor__seed-84__imagenet-resnet-s84` | 0.6983 | 0.6818 | 0.5714 | 0.5580 | integrar e validar recarga |
| ResNet50 | 3 | central | `resnet50__tres__central__seed-42__final-v2` | 0.4504 | 0.5227 | 0.1538 | 0.7319 | integrar e validar recarga |
| ResNet50 | 3 | maior tumor | `resnet50__tres__maior-tumor__seed-42__final-v2` | 0.6074 | 0.6591 | 0.5600 | 0.6341 | integrar e validar recarga |
| MobileNetV2 | 1 | central | `c_mobilenetv2_post_early_central` | 0.5372 | 0.5909 | 0.5405 | 0.4783 | na coorte local (Git ignora .pt) |
| MobileNetV2 | 1 | maior tumor | `mobilenetv2__post-early__maior-tumor__seed-42__imagenet-mobile-s42` | 0.6074 | 0.5227 | 0.5000 | não avaliada | integrar e validar recarga |
| MobileNetV2 | 3 | central | `c_mobilenetv2_tres_central` | 0.6488 | 0.6591 | 0.5806 | 0.5652 | na coorte local (Git ignora .pt) |
| MobileNetV2 | 3 | maior tumor | `c_mobilenetv2_tres_maior_tumor` | 0.5661 | 0.6136 | 0.4762 | 0.5435 | na coorte local (Git ignora .pt) |

**Resumo da comparação:** das 12 combinações, 6 maiores AUCs de validação constam da coorte de 27 checkpoints, 4 da grade paralela final-v2 e 2 de experimentos paralelos com normalização ImageNet. O maior valor reportado é **0.6983** (resnet50, 1 fase, maior tumor; `resnet50__post-early__maior-tumor__seed-84__imagenet-resnet-s84`). A diferença em relação à CNN A02 (0,6818) é pequena e não comprova superioridade estatística.

**Disponibilidade:** os checkpoints da coorte local foram recarregados na auditoria anterior. Os checkpoints da grade paralela são mantidos em outro ambiente; **não foram carregados nem retestados por esta integração**. Um asset do melhor ResNet50 ImageNet foi publicado em [release issue-4-modelo-selecionado-v2](https://github.com/GabrielChavesMendes/Trabalho_PAI/releases/tag/issue-4-modelo-selecionado-v2). Antes de adotar qualquer peso externo em inferência/Grad-CAM, copiar o arquivo, verificar metadados, implementar/validar o mesmo pré-processamento no carregador final e conferir as previsões. Os demais pesos precisam ser disponibilizados por quem os treinou. Não usar o ranking consolidado como declaração de que 12 pesos já estão prontos para demonstração.

## Encadeamento dos ajustes: hipótese → experimento → decisão

| Arquitetura | Diagnóstico na grade original | Investigações intermediárias | Última grade e conclusão |
|---|---|---|---|
| **CNN** | Previsões muito concentradas, canais de ativação inativos e capacidade discriminativa limitada. | Diminuir LR (AUC 0,6694) e usar LeakyReLU (0,5702) não superou a referência de 0,6818. | Adicionar GroupNorm após convoluções para estabilizar ativações, sem trocar pooling nem dropout. A nova grade não superou a CNN original na configuração principal. |
| **ResNet50** | Distância elevada entre desempenho de treino e validação, indicando sobreajuste. | Normalização ImageNet isolada na cabeça congelada (AUC 0,5785) explorou incompatibilidade de escala; ajustes de pesos de classe/fine-tuning foram avaliados separadamente. | Congelar estatísticas e parâmetros de BatchNorm com `layer4` liberada. A grade nova não resolveu o sobreajuste. Outra linha de experimentação combinou pré-processamento ImageNet e seed 84, com AUC de validação 0,6983 em uma fase/maior tumor. |
| **MobileNetV2** | Backbone congelado, baixa AUC em múltiplas combinações. | A normalização ImageNet foi explorada na linha paralela para uma fase/maior tumor (AUC 0,6074). | Fine-tuning de `features[17:19]` e classificador, mantendo BatchNorm congelada. Três fases/corte central subiu de AUC 0,5207 para 0,6488, sem demonstrar generalização robusta. |

Dessa forma, **os três ajustes intermediários NÃO aparecem como testes soltos**: eles explicam as hipóteses e limitações que motivaram a última rodada; as intervenções finais foram fixadas antes de observar seus resultados. Para o artigo, usar este quadro como narrativa e escolher apenas as principais curvas de loss, uma figura de topologia, uma tabela compacta de 12 combinações e exemplos de acertos/erros. Os CSVs e relatórios integrais ficam como material de apoio e não precisam entrar todos no ZIP.

## Critérios de comparação e limitações

- A ordem desta tabela usa **AUC de validação** dentro de cada combinação. Sensibilidade, especificidade, precisão, acurácia, F1 e BA estão disponíveis no [CSV consolidado](../resultados/grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.csv). AUC independe do limiar; as métricas de classificação dependem dos limiares escolhidos em cada linha experimental e não são perfeitamente intercambiáveis.
- A coorte de 27 tem seleção congelada anterior à sua avaliação de teste. A comparação integrada acrescenta resultados posteriormente e tem caráter exploratório: **não deve ser apresentada como uma nova seleção confirmada por teste inteiramente independente**. O histórico de desenvolvimento inclui consulta prévia ao conjunto de teste; documentar a limitação de forma impessoal e precisa, sem imputar decisões de uma branch a outra.
- As duas linhas usam os 172 pacientes oficiais (104 treino, 33 validação e 35 teste), porém ainda é necessário conferir compatibilidade completa dos IDs, das transformações, da agregação por paciente e dos arquivos de pesos antes de afirmar equivalência plena dos experimentos.
- Treinamentos diferentes usaram limiares, otimizadores, pesos e normalizações em condições distintas. Não comparar diretamente os valores de loss como se fossem a mesma função e escala sem verificar as configurações. Diferenças de AUC pequenas não provam superioridade. **Nenhum resultado autoriza afirmar utilidade clínica**.
- O ensemble com cinco folds é um experimento metodologicamente distinto (validação OOF) e não entra nos 12 representantes únicos da grade fase/corte. Pode ser citado como discussão de variação/generalização se houver espaço, não como uma 13ª combinação equivalente.

## Fontes e próximo passo técnico

- Coorte de 27: [GRADE_DEFINITIVA_ISSUE_4_20261117.md](GRADE_DEFINITIVA_ISSUE_4_20261117.md), [RELATORIO_FINAL.md](../resultados/grade_definitiva_issue4_20261117_01/RELATORIO_FINAL.md), [ranking dos 12 da coorte](../resultados/grade_definitiva_issue4_20261117_01/artigo/ranking_teste_12_acuracia_balanceada.csv).
- Grade paralela: [resultados final-v2](https://github.com/GabrielChavesMendes/Trabalho_PAI/blob/release/treinamento-issue-4/resultados/finais_issue4/comparacao_validacao.csv) e [experimentos ImageNet](https://github.com/GabrielChavesMendes/Trabalho_PAI/blob/release/treinamento-issue-4/resultados/busca_imagenet_issue4/comparacao_validacao.csv).
- **Integração futura:** validar carregamento e correspondência das predições dos seis checkpoints externos. Não duplicar a implementação principal; adaptar o carregador existente para reconhecer os metadados/normalizações necessários, sem reescrever o pipeline da coorte principal. Se faltarem pesos, manter resultado como comparativo reportado, nunca como modelo executável.
- **Para entrega final:** consolidar código Python único, interface, inferência e Grad-CAM, relatório SBC/LaTeX/PDF e declaração fiel de uso de IA. Os pesos não entram no ZIP principal.
