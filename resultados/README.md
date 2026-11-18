# Resultados experimentais — G10

Este diretório preserva os dados e figuras das diferentes rodadas, inclusive os resultados negativos. **Não é o conteúdo do ZIP de entrega.** Todos os checkpoints `.pt` são mantidos localmente e não são versionados.

## Ponto de partida

- [Tabela consolidada dos 12 representantes](grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.csv)
- [Relatório da investigação de 27 experimentos](grade_definitiva_issue4_20261117_01/RELATORIO_FINAL.md)
- [Síntese final para o artigo](../docs/RESULTADOS_FINAIS.md)
- [Justificativas e decisões de hiperparâmetros](../docs/EXPERIMENTOS_E_DECISOES.md)

Há artefatos históricos de três rodadas, auditorias, logs, tempos, matrizes de confusão e gráficos. Eles continuam preservados para rastreabilidade; não se devem misturar métricas de validação e teste nem alterar resultados antigos após observar o teste.

Convenção predominante: `arquitetura__fases__corte__seed-N__tipo.ext`, com `tipo` como `history`, `validation`, `test`, `loss` ou `confusion-matrix`.

Na versão limpa do projeto, os principais executáveis encontram-se em `scripts/verificar_modelos.py`, `scripts/inferir_paciente.py` e `entrega_final_G10/trabalho_pai.py`. Os nomes antigos de scripts podem ser encontrados no histórico Git.

**Privacidade:** não acrescentar imagens DCE brutas, máscaras volumétricas nem dados clínicos pessoais ao repositório.
