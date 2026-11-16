# Resultados dos experimentos

Guardar aqui tabelas pequenas (CSV/Markdown), matrizes de confusão e gráficos (PNG) dos experimentos reais, com identificação de modelo, fase, corte, hiperparâmetros e seed.

Não inserir dados clínicos identificáveis, imagens brutas volumosas ou checkpoints.

Convenção recomendada: `arquitetura__fases__corte__seed-N_TIPO.ext`, em que `TIPO`
pode ser `history`, `validation`, `test`, `loss` ou `confusion-matrix`. Resultados de
smoke tests e dados sintéticos não devem ser armazenados nesta pasta.

Os artefatos atuais correspondem a 12 treinamentos reais (seed 42). A tabela
principal e `comparacao_validacao.csv`; o protocolo e a interpretacao estao em
`docs/RESULTADOS_ISSUE_4.md`. Apenas o checkpoint escolhido pela validacao possui
artefatos `__test`; os checkpoints `.pt` permanecem ignorados pelo Git.
`modelo_selecionado.json` e o contrato rastreavel para inferencia, Grad-CAM e
empacotamento dos pesos nas Issues #5 e #8.
