# Síntese integrada — 12 configurações definitivas da Issue #4

Este documento reúne **uma única configuração para cada combinação** de arquitetura, fase temporal e corte: CNN, ResNet50 e MobileNetV2; uma fase (`post_early`) ou três fases temporais (`pre`, `post_early`, `post_late`); corte central ou maior área tumoral. Não inclui 2,5D nem ensemble.

Os resultados combinam treinamentos realizados em duas branches do mesmo grupo. São **resultados experimentais registrados**, não uma nova execução uniforme dos doze modelos. Escolhemos os representantes por comparação retrospectiva das métricas de **validação**. A comparação contém diferenças de protocolo e deve ser interpretada com cautela.

## Comparação definitiva na validação — 33 pacientes

Todos os valores são de validação, antes da avaliação do teste. Sens. = sensibilidade, Esp. = especificidade, Prec. = precisão, Acc. = acurácia, BA = acurácia balanceada.

| Arquitetura | Fases temporais | Corte | Origem | Sens. | Esp. | Prec. | Acc. | F1 | AUC | BA |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 1 | Central | Equipe (treinamento Gustavo) | 0.5455 | 0.6364 | 0.4286 | 0.6061 | 0.4800 | 0.6033 | 0.5909 |
| CNN | 1 | Maior tumor | Equipe (linha principal) | 0.5455 | 0.9091 | 0.7500 | 0.7879 | 0.6316 | 0.6818 | 0.7273 |
| CNN | 3 | Central | Equipe (linha principal) | 0.8182 | 0.5000 | 0.4500 | 0.6061 | 0.5806 | 0.5950 | 0.6591 |
| CNN | 3 | Maior tumor | Equipe (linha principal) | 0.9091 | 0.4091 | 0.4348 | 0.5758 | 0.5882 | 0.6281 | 0.6591 |
| ResNet50 | 1 | Central | Equipe (treinamento Gustavo) | 0.9091 | 0.1364 | 0.3448 | 0.3939 | 0.5000 | 0.4215 | 0.5227 |
| ResNet50 | 1 | Maior tumor | Equipe (treinamento Gustavo) | 0.5455 | 0.8182 | 0.6000 | 0.7273 | 0.5714 | 0.6983 | 0.6818 |
| ResNet50 | 3 | Central | Equipe (linha principal) | 0.2727 | 0.9091 | 0.6000 | 0.6970 | 0.3750 | 0.4050 | 0.5909 |
| ResNet50 | 3 | Maior tumor | Equipe (treinamento Gustavo) | 0.6364 | 0.6818 | 0.5000 | 0.6667 | 0.5600 | 0.6074 | 0.6591 |
| MobileNetV2 | 1 | Central | Equipe (linha principal) | 0.2727 | 0.9545 | 0.7500 | 0.7273 | 0.4000 | 0.4959 | 0.6136 |
| MobileNetV2 | 1 | Maior tumor | Equipe (linha principal) | 0.9091 | 0.2273 | 0.3704 | 0.4545 | 0.5263 | 0.4752 | 0.5682 |
| MobileNetV2 | 3 | Central | Equipe (linha principal) | 0.8182 | 0.5000 | 0.4500 | 0.6061 | 0.5806 | 0.6488 | 0.6591 |
| MobileNetV2 | 3 | Maior tumor | Equipe (linha principal) | 0.4545 | 0.7727 | 0.5000 | 0.6667 | 0.4762 | 0.5661 | 0.6136 |

**Resumo:** 8 checkpoints da linha experimental principal e 4 treinados na linha paralela do Gustavo. Cada linha corresponde a uma das doze configurações, **sem repetições**. A métrica AUC auxilia a análise, mas não substitui sensibilidade, especificidade, precisão, acurácia e F1, exigidas no enunciado. O CSV contém também limiar, loss, tempo e métricas de teste *reportadas*.

[CSV da seleção e das métricas](../resultados/grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.csv) · [Tabela LaTeX](../resultados/grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.tex)

## Rastreabilidade e diferenças de treinamento

O identificador interno do arquivo de checkpoint indica arquitetura, fase, corte, seed e, quando presente, variante de treinamento. Ele **não** designa uma configuração adicional. Os códigos usados na investigação histórica (grupos A, B e C) serviram apenas para diferenciar rodadas internas e não aparecem como rótulos desta tabela.

- **CNN:** arquitetura com quatro convoluções e duas camadas lineares (seis camadas com parâmetros). Ajustes de taxa de aprendizagem, LeakyReLU e GroupNorm foram investigados na linha principal; a seleção final preserva os melhores resultados por configuração.
- **ResNet50:** rede iniciada com pesos pré-treinados ImageNet, com cabeça de classificação binária e ajustes de fine-tuning conforme os experimentos. A linha paralela também testou normalização ImageNet após a normalização por percentis. O melhor modelo de uma fase/maior tumor utiliza essa opção e seed 84.
- **MobileNetV2:** rede pré-treinada ImageNet; a linha principal investigou fine-tuning parcial para aperfeiçoar a validação. Os modelos definitivos mantêm a entrada temporal prevista no enunciado.

As linhas experimentais usam o CSV oficial de 172 pacientes com 104 treino, 33 validação e 35 teste. A avaliação de métricas ocorre por paciente, agregando as probabilidades das variações de validação. As diferenças de normalização, função de perda e limiares devem ser preservadas na reprodução dos checkpoints; **loss entre protocolos distintos não deve ser interpretada como comparação direta**. A documentação completa das 27 execuções anteriores permanece em [GRADE_DEFINITIVA_ISSUE_4_20261117.md](historico/GRADE_DEFINITIVA_ISSUE_4_20261117.md).

## Integração dos quatro checkpoints externos

Os **seis arquivos externos** recebidos foram abertos com `weights_only=True` (com suporte a `TorchVersion`), verificados quanto a fase/canais/corte e recarregados com correspondência estrita de `state_dict` em CPU. Todos produziram saída finita de dimensão `[1,1]` em entrada artificial. Somente **quatro** fazem parte da tabela definitiva:

- **CNN, uma fase, central**: `cnn__post-early__central__seed-42__final-v2.pt`; SHA-256 `8ea882df35704e58273e42df9303864395a9b832e5b2ec75578e03701c2f3afd`.
- **ResNet50, uma fase, central**: `resnet50__post-early__central__seed-42__final-v2.pt`; SHA-256 `b027d09f78bfb908302a9d871ef1d86e56d3bb37536bb8839242df24f4087a39`.
- **ResNet50, uma fase, maior tumor**: `resnet50__post-early__maior-tumor__seed-84__imagenet-resnet-s84.pt`; SHA-256 `924a6b00ca7ee951b6e79f698f2a63f61ee7324de44c616ddfe15ece78dd9733`.
- **ResNet50, três fases, maior tumor**: `resnet50__tres__maior-tumor__seed-42__final-v2.pt`; SHA-256 `ec6a1fac483c0fc188d6a580e5aee777aaeb5ce7dde4531ba7f750bcad8ec95b`.

O código adicional é restrito a `modulos/checkpoints_integrados_issue4.py`, ao modo opcional de normalização ImageNet em `modulos/fabrica_carregadores.py` e ao script de verificação. O pipeline e os experimentos anteriores continuam com a normalização padrão `percentil`; **nenhuma alteração retroativa dos pesos é feita**.

Arquivos de histórico, loss, validação, teste e matrizes de confusão dos quatro modelos externos estão em `resultados/grade_definitiva_issue4_20261117_01/integracao_gustavo/`. Os checkpoints binários **não vão para Git** nem para o ZIP principal.

### Procedimento de verificação local (sem retreino)

Copiar os quatro pesos citados para `checkpoints/integrados_issue4/` (pasta plana, preservando os nomes) e executar na raiz do projeto:

```bash
python -m pytest -q
python scripts/verificar_checkpoints_integrados_issue4.py --device cpu
# Com PAI_DATASET_DIR e PAI_CSV configurados e GPU disponível:
python scripts/verificar_checkpoints_integrados_issue4.py --device cuda --validar-dataset
```

**Verificação local concluída:** os quatro externos passaram pela recarga estrita CPU e reproduziram as cinco métricas obrigatórias, AUC, BA, loss e confusão nos 33 pacientes reais em CUDA, preservando seus limiares. Os outros oito, em seus caminhos originais, também foram recarregados em CPU; os 12 produziram previsão CUDA de um paciente de validação. Isso não representa nova validação completa dos oito nesta sessão. Evidências em [verificação da integração](../resultados/verificacao_integrados_issue4_20261117_01/README.md).

Execução simplificada: `python scripts/verificar_issue4.py`. Inferência individual: `python scripts/inferir_paciente_issue4.py --listar` e `python scripts/inferir_paciente_issue4.py --modelo 6 --paciente ISPY1_1009`. Os comandos usam a `.venv` existente e detectam CUDA/CPU. A seleção de 12 configurações e os resultados históricos não foram modificados.

## Apresentação dos resultados e limitações

Sensibilidade, especificidade, precisão, acurácia, F1, AUC, BA e matriz de confusão por paciente devem acompanhar a comparação; gráficos de loss e tempos de treinamento preservam o histórico de cada treinamento. Não há evidência suficiente para inferir uso clínico ou superioridade estatística a partir de diferenças pequenas.

**Limitação metodológica:** esta comparação integrada foi organizada retrospectivamente e o histórico experimental já continha consultas ao conjunto de teste. Portanto, os resultados relatados no teste não representam uma avaliação externa totalmente intocada; a escolha dos representantes desta síntese foi baseada em validação. A limitação deve ser descrita fielmente no artigo.

## Próxima etapa

Revisar o PR #17 e a integração com os demais módulos. O terminal de inferência está disponível; interface gráfica definitiva/Grad-CAM pertencem à Issue #5. Arquivo Python único, artigo e entrega integrada continuam pendentes. Esta execução não faz merge na `develop` nem fecha a Issue #4.
