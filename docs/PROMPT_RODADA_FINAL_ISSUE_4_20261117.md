# Solicitação da última rodada — registro de uso de IAG

Ferramenta: Codex (OpenAI), com PowerShell/Python e leitura da Issue #4 pelo
conector GitHub. Finalidade: diagnóstico, execução de três experimentos,
análise exclusivamente por validação, testes, documentação e preparação da
avaliação final sem executá-la. Conteúdo da solicitação do usuário abaixo,
com formatação Markdown normalizada.

---

Estamos na branch `feature/felipe-issue4-finalizacao` do Trabalho de PAI, finalizando a Issue #4.

Já executamos quatro experimentos de ajuste de hiperparâmetros e uma grade completa de **12 configurações**, considerando as três redes (CNN, ResNet50 e MobileNetV2), uma ou três fases temporais e duas estratégias de corte.

Nosso melhor modelo atual na validação é a **CNN com uma fase `post_early` e corte de maior área tumoral**, com:

- **AUC-ROC:** 0,6818
- **Acurácia:** 78,79%
- **Acurácia balanceada:** 72,73%
- **F1-score:** 0,6316
- **Sensibilidade:** 54,55%
- **Especificidade:** 90,91%

Entretanto, identificamos duas limitações importantes: sensibilidade relativamente baixa e probabilidades quase constantes, tornando o limiar de classificação extremamente sensível a pequenas variações.

## Objetivo

**Realizar uma última rodada curta, controlada e tecnicamente justificada de melhorias antes da Etapa 3**, buscando aumentar a capacidade dos modelos de distinguir pacientes pCR=0 e pCR=1.

Queremos preservar os resultados atuais, comparar os novos experimentos pela validação e, posteriormente, avaliar tanto os modelos originais quanto os novos candidatos no conjunto de teste.

Não temos uma meta obrigatória de acurácia ou AUC. A prioridade é obter resultados confiáveis, preservar a metodologia e cumprir fielmente o enunciado.

## Etapa 1 — Revisão e diagnóstico

1. Leia integralmente:
   - `docs/FINALIZACAO_ISSUE_4.md`
   - `docs/EXECUCAO_ISSUE_4_20261117.md`
   - `resultados/execucao_issue4_20261117_01/REVISAO_ANTES_TESTE.md`
   - O enunciado oficial e a Issue #4 do GitHub.
2. Revise os códigos de treinamento, pré-processamento, normalização, arquiteturas e avaliação.
3. Investigue as possíveis causas das probabilidades quase constantes da CNN. Considere inicialização, normalização, convergência, função de perda, regularização, hiperparâmetros e outros fatores tecnicamente relevantes.
4. Verifique se há problemas de implementação que estejam prejudicando o aprendizado, sem presumir previamente uma causa específica.

## Etapa 2 — Última rodada de melhorias

5. Proponha e execute **no máximo tres experimentos adicionais**, priorizando modificações simples, justificáveis e que tenham potencial de melhorar a capacidade discriminativa dos modelos.
6. Antes de executar, registre as hipóteses investigadas, quais parâmetros serão modificados, o que será mantido constante e os critérios de avaliação.
7. Utilize **exclusivamente treino e validação** para investigar, ajustar e selecionar modelos. Não consulte o conjunto de teste durante essa etapa.
8. Preserve integralmente os 12 modelos atuais, checkpoints, métricas, gráficos e documentação. Salve os novos experimentos em diretórios separados, sem sobrescrever nada.
9. Compare os resultados novos e antigos na validação, considerando:
   - AUC-ROC;
   - Acurácia balanceada;
   - Sensibilidade;
   - Especificidade;
   - Precisão;
   - F1-score;
   - Acurácia;
   - Loss e curvas de convergência.
10. Analise especialmente a distribuição das probabilidades, o sobreajuste e a estabilidade do limiar de classificação.
11. Diferencie uma melhoria real na capacidade de distinguir as classes de uma melhoria obtida somente por ajustar um limiar muito sensível.
12. Caso não haja melhoria convincente, mantenha os modelos anteriores. Não precisamos alcançar artificialmente determinada porcentagem.

## Etapa 3 — Preparação da avaliação final

13. Após concluir os novos experimentos, compare todos os resultados de validação e escolha quais novos candidatos realmente merecem participar da avaliação final.
14. Preserve os **12 checkpoints originalmente treinados** e os checkpoints dos novos candidatos escolhidos.
15. Antes da avaliação final, congele e registre:
   - Arquitetura;
   - Fases utilizadas;
   - Estratégia de corte;
   - Normalização;
   - Hiperparâmetros;
   - Época selecionada;
   - Pesos treinados;
   - Limiar de classificação.
16. Confirme que todos os checkpoints podem ser carregados corretamente sem executar novo treinamento.
17. Prepare a avaliação final para incluir **as 12 configurações atuais e os novos candidatos selecionados previamente pela validação**. Todos devem utilizar o mesmo conjunto de teste e o mesmo protocolo de avaliação, respeitando as configurações de pré-processamento correspondentes a cada checkpoint.
18. Registre os resultados separadamente, permitindo comparar modelos antigos e novos. O teste servirá exclusivamente para avaliação final, **não para escolher vencedores, modificar hiperparâmetros ou realizar novos ajustes**.
19. **Não execute a Etapa 3 ainda.** Primeiro apresente os resultados de validação dos novos experimentos para nossa revisão. Após a decisão, executaremos a avaliação final uma única vez.

## Etapa 4 — Documentação e verificação

20. Atualize a documentação da Issue #4 com:
   - Hipóteses investigadas;
   - Alterações realizadas;
   - Justificativas técnicas;
   - Hiperparâmetros testados;
   - Resultados intermediários;
   - Comparação entre modelos antigos e novos;
   - Curvas e gráficos relevantes;
   - Limitações observadas;
   - Decisões tomadas com base na validação.
21. Execute os testes automatizados e corrija eventuais problemas encontrados.
22. Confirme que todas as exigências do enunciado continuam atendidas, especialmente treinamento e avaliação das três redes, comparação de uma e três fases e utilização de duas estratégias de cortes.
23. Mantenha o código organizado, claro e compreensível para os integrantes, pois posteriormente todos os módulos serão integrados em um único arquivo Python.

## Restrições obrigatórias

- Não modificar as partições oficiais dos 172 pacientes.
- Não permitir vazamento de dados entre treinamento, validação e teste.
- Não usar o conjunto de teste durante a investigação ou escolha dos novos modelos.
- Não apagar nem sobrescrever checkpoints ou resultados anteriores.
- Não alterar o funcionamento correto das Issues #1, #2 e #3.
- Não remover funcionalidades exigidas pelo enunciado.
- Não realizar uma busca extensa ou indefinida de hiperparâmetros.
- Não fazer merge na `develop`.
- Não fechar a Issue #4.
- Não executar a Etapa 3 sem nossa revisão dos resultados de validação.

## Entrega esperada desta execução

Ao terminar, apresente:

1. Um resumo objetivo do diagnóstico realizado.
2. Quais experimentos adicionais foram executados e por quê.
3. Uma **tabela comparativa de validação**, com resultados anteriores e novos.
4. Uma análise indicando se houve melhoria real na distinção das classes, especialmente na AUC-ROC e na acurácia balanceada.
5. Uma recomendação fundamentada sobre quais checkpoints devemos preservar para o teste final.
6. A confirmação dos testes automatizados e dos arquivos gerados.
7. Os comandos exatos para executar a Etapa 3 posteriormente.

**Nosso objetivo final é concluir a Issue #4 com resultados tão bons quanto os dados permitirem, experimentos metodologicamente corretos e documentação suficiente para o artigo e a apresentação.**

Não é necessário alcançar um valor mínimo de desempenho para cumprir o enunciado. O mais importante é demonstrar que treinamos, avaliamos, comparamos e investigamos os resultados corretamente.
