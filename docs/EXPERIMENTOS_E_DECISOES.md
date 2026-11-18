# Experimentos e decisões técnicas — Grupo G10

Este documento resume as discussões de desenvolvimento. Não substitui os arquivos de auditoria nem os resultados da avaliação por paciente.

## Conjunto de dados e projeto fatorial

O CSV oficial possui 172 pacientes I-SPY1: 104 treino, 33 validação e 35 teste. As 12 condições obrigatórias combinam três arquiteturas (CNN, ResNet50 e MobileNetV2), uma ou três fases (post_early isolada ou pre/post_early/post_late) e corte central ou de maior área tumoral. As máscaras servem para seleção espacial e inspeção, **não** entram como quarto canal. Normalização por percentis 1/99 e aumento geométrico 10× foram usados conforme as variantes documentadas.

### Experimentos intermediários: por que foram feitos?

O diagnóstico inicial mostrou pouca variação nas probabilidades da CNN e canais inativos, sem demonstrar que imagens fossem uniformes. Foram realizados três testes controlados:

| Variante | Hipótese | AUC validação | Resultado |
|---|---|---:|---|
| CNN com LR reduzida (1e-3 → 3e-4) | Evitar passos grandes e preservar aprendizagem | 0,6694 | Não superou a CNN de referência (0,6818) |
| CNN com LeakyReLU(0,01) | Manter gradiente em entradas negativas | 0,5702 | Não resolveu a discriminação |
| ResNet50 com normalização ImageNet adicional | Aproximar a escala da esperada pelo backbone pré-treinado | 0,5785 | Melhorou um controle congelado específico, mas não foi o melhor modelo geral |

O efeito sobre sensibilidade, falsos positivos, dispersão dos escores e estabilidade do limiar também foi examinado. Resultados detalhados em [rodada intermediária](historico/RODADA_FINAL_ISSUE_4_20261117.md).

## Última grade experimental (C): escolhas antes do treinamento

| Arquitetura | Modificação controlada | Motivação |
|---|---|---|
| CNN | GroupNorm com quatro grupos depois das convoluções | Diminuir instabilidade de ativações com batch pequeno |
| ResNet50 | Congelar integralmente as BatchNorm durante ajuste de layer4+fc | Evitar estatísticas de lote pouco estáveis |
| MobileNetV2 | Ajustar features[17:19] e cabeça, com BN congeladas | Adaptar parte final do extrator ao domínio DCE-MRI |

Os 12 treinamentos dessa grade foram conduzidos com seed 42, batch 4, até 30 épocas, parada antecipada e seleção conforme validação. As hipóteses, hiperparâmetros, comparações pareadas, gráficos e auditoria dos **27 checkpoints** das rodadas A+B+C estão em [grade definitiva](historico/GRADE_DEFINITIVA_ISSUE_4_20261117.md).

**Interpretação:** GroupNorm reduziu alguns sinais de inatividade, sem assegurar maior AUC; BN congelada teve ganhos mistos, mantendo o forte risco de overfitting; o fine-tuning da MobileNetV2 melhorou a AUC de validação nas quatro células da comparação original, mas sem comprovação de melhora robusta no teste.

## Seleção e generalização

Os doze representantes integrados contêm oito checkpoints da linha experimental principal e quatro da linha paralela do Gustavo. Essa seleção é registrada em [resultados finais](RESULTADOS_FINAIS.md), com limiares específicos de cada checkpoint.

A ResNet50 de uma fase e maior tumor apresentou a maior AUC de validação entre os 12 representantes (0,6983). A ResNet50 de três fases e maior tumor apresentou a maior AUC de teste **entre esses 12** (0,6341), mas o teste não deve ser usado para redefinir o vencedor pré-selecionado. O modelo não é adequado a uso clínico sem avaliação externa independente.

## Grad-CAM e máscaras

A interface usa a máscara oficial do mesmo paciente/corte e mostra relevância espacial Grad-CAM. O mapa de calor **não é uma segmentação tumoral** e não prova causalidade. O resultado visual deve identificar a classe explicada, o corte e o checkpoint.

## Artigo e reprodutibilidade

O artigo final deve conter, além dos 12 resultados principais, explicação sucinta das hipóteses experimentais e dos resultados intermediários. Os registros detalhados permanecem no histórico para permitir a discussão e avaliação crítica.

O `trabalho_pai.py` de entrega **não incorpora todas as variantes históricas** de treino, mas oferece o fluxo básico com as três arquiteturas, as 12 combinações, inferência e Grad-CAM. Os arquivos em `modulos/`, `scripts/` e `resultados/` documentam os procedimentos científicos adicionais.
