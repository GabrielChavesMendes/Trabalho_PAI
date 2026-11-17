# Material do artigo — Grupo G10

Esta pasta organiza os dados tabulares e as figuras que devem acompanhar o relatório SBC. Ela não contém imagens médicas de pacientes nem checkpoints.

## Arquivos desta pasta

- `dados_modelos_validacao.csv`: catálogo integrado das 12 configurações selecionadas por validação, com arquitetura, fases, corte, origem, métricas, limiar e caminhos/hashes dos checkpoints. Fonte: `resultados/grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.csv`.
- `figuras/`: destino para cópias finais das figuras selecionadas para o artigo.
- `tabelas/`: destino para tabelas consolidadas em CSV/LaTeX, se forem necessárias além do CSV principal.

## Figuras já versionadas no projeto

Priorizar as figuras já produzidas e revisadas, sem recriar métricas manualmente:

- Topologias: `modulos/documentacao/topologia_cnn.png`, `topologia_resnet50.png`, `topologia_mobilenetv2.png`.
- Convergência e comparação: `resultados/rodada_final_issue4_20261117_01/figuras_revisadas/convergencia.png`, `comparacao_validacao.csv` e as curvas individuais em `resultados/`.
- Distribuição e limiar: `resultados/rodada_final_issue4_20261117_01/figuras_revisadas/distribuicao_probabilidades.png`, `estabilidade_limiar.png`.
- Comparações da grade: `resultados/execucao_issue4_20261117_01/comparacao_fases_cortes_validacao.png`, `comparacao_limiares_validacao.png`, `curvas_doze_modelos.png`.
- Matrizes de confusão e exemplos: arquivos `*__test__confusion-matrix.png` e `*__test__examples.csv` sob `resultados/grade_definitiva_issue4_20261117_01/teste_final/`.
- Grad-CAM e máscara: produzir após carregar um checkpoint compatível e validar com imagem real; inserir somente imagens autorizadas e sem identificadores pessoais.

## Tabelas obrigatórias no artigo

Apresentar por configuração: sensibilidade, especificidade, precisão, acurácia, F1; complementar com AUC, acurácia balanceada, loss e tempo de treinamento quando disponíveis. Explicar partições, limiar de validação, diferenças entre protocolos e limitações de generalização.

## Regras de integridade

1. Não inventar valores nem preencher lacunas com estimativas.
2. Não misturar validação e teste. A escolha de checkpoint deve continuar baseada na validação.
3. Não afirmar que o maior resultado observado no teste substitui o vencedor selecionado na validação.
4. Não incluir dataset, volumes NIfTI, máscaras originais ou pesos no `G10.zip`.
5. Antes de incluir Grad-CAM, confirmar paciente/corte correspondente e deixar claro que o mapa não é segmentação nem prova causal.
6. Registrar a declaração de IAG de forma fiel ao uso efetivo e revisar o texto com todos os integrantes.

## Pendências editoriais

- Selecionar as figuras finais e conferir legibilidade em escala de impressão.
- Montar tabela final única com métricas de validação e teste, respeitando a origem/protocolo de cada checkpoint.
- Inserir topologias e exemplos Grad-CAM reais, se a validação for concluída.
- Conferir nomes, matrículas, curso e campus dos quatro integrantes.
- Conferir limite de 12 páginas e compilar PDF SBC.
