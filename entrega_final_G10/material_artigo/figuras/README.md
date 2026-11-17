# Figuras do artigo — Grupo G10

As figuras abaixo foram copiadas para esta pasta sem alterar os arquivos de origem. Os arquivos-fonte permanecem em `resultados/` e `modulos/documentacao/`.

## Figuras reunidas

| Arquivo | Conteúdo | Fonte |
|---|---|---|
| `topologia_cnn.png` | Topologia da CNN autoral. | `modulos/documentacao/topologia_cnn.png` |
| `topologia_resnet50.png` | Topologia da ResNet50. | `modulos/documentacao/topologia_resnet50.png` |
| `topologia_mobilenetv2.png` | Topologia da MobileNetV2. | `modulos/documentacao/topologia_mobilenetv2.png` |
| `convergencia_grade.png` | Curvas de convergência da grade definitiva. | `resultados/grade_definitiva_issue4_20261117_01/artigo/figuras/convergencia_grade_C.png` |
| `comparacao_arquiteturas.png` | Comparação entre arquiteturas. | `.../artigo/figuras/comparacao_arquitetura.png` |
| `comparacao_fases.png` | Comparação entre uma e três fases temporais. | `.../artigo/figuras/comparacao_fases.png` |
| `comparacao_estrategias_corte.png` | Comparação entre as estratégias de seleção de corte. | `.../artigo/figuras/comparacao_corte.png` |
| `curvas_roc_validacao.png` | Curvas ROC dos modelos selecionados por validação. | `.../artigo/figuras/roc_melhores_por_validacao.png` |
| `probabilidades_e_limiares.png` | Distribuições de probabilidade e limiares. | `.../artigo/figuras/probabilidades_e_limiares.png` |
| `tempos_treinamento.png` | Comparação dos tempos de treinamento. | `.../artigo/figuras/tempos_treinamento.png` |
| `matrizes_confusao_representantes.png` | Matrizes de confusão de configurações representativas. | `.../artigo/figuras/matrizes_representantes.png` |
| `validacao_versus_teste.png` | Comparação de métricas de validação e teste. | `.../artigo/figuras/validacao_versus_teste.png` |

## Cuidados para o texto científico

- Identificar explicitamente a partição usada em cada figura e tabela.
- A seleção de arquitetura, checkpoint e limiar deve ser descrita como feita pela validação; o conjunto de teste não deve orientar novos ajustes.
- A avaliação final versionada registra 27 modelos em 35 pacientes, com seleções congeladas antes da execução. A documentação também registra que resultados originais de teste já haviam sido consultados anteriormente; portanto, não caracterizar o teste como completamente cego.
- Conferir as legendas, unidades, resolução e legibilidade de cada figura no PDF final antes da submissão.
- **Grad-CAM com exame real e máscara do mesmo paciente/corte ainda não foi reunido aqui.** Não substituir por uma imagem sintética ou por uma figura de ativação de arquitetura. Incluir somente após validar checkpoint, paciente, corte e máscara na interface.
- Não incluir volumes NIfTI, máscaras originais ou checkpoints nesta pasta nem no `G10.zip`.

