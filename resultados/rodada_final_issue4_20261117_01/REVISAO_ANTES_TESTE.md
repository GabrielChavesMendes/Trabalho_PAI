# Revisão da última rodada — teste NÃO executado

Relatório completo, hipóteses, controles, resultados, limitações, figuras e
comandos: [RODADA_FINAL_ISSUE_4_20261117.md](../../docs/RODADA_FINAL_ISSUE_4_20261117.md).

| Candidato | AUC | Ac. bal. | Sens. | Espec. | Precisão | F1 | Acurácia | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN anterior | 0,6818 | 0,7273 | 0,5455 | 0,9091 | 0,7500 | 0,6316 | 0,7879 | 1,1563 |
| CNN LR menor | 0,6694 | 0,7045 | 0,7273 | 0,6818 | 0,5333 | 0,6154 | 0,6970 | 1,1612 |
| CNN LeakyReLU | 0,5702 | 0,6136 | 0,7273 | 0,5000 | 0,4211 | 0,5333 | 0,5758 | 1,1598 |
| ResNet50 padronizada | 0,5785 | 0,6364 | 0,2727 | 1,0000 | 1,0000 | 0,4286 | 0,7576 | 1,1432 |

Recomendação: manter os 12 modelos atuais, sem acrescentar os três candidatos,
pelos critérios comparativos registrados antes dos treinos. As duas novas CNNs
continuam sensíveis ao limiar. A ResNet50 tem escores mais espalhados e estáveis,
mas AUC e sensibilidade inferiores à referência principal. Não há melhoria
convincente de discriminação; não iniciar uma quarta tentativa.

Todos os modelos/pesos anteriores e novos foram preservados. Os 266 artefatos
anteriores mantiveram seus hashes; recargas independentes dos três candidatos
reproduziram exatamente as probabilidades e classes. Suíte final: 50 testes
passaram. As 12 configurações originais continuam cobrindo três redes,
uma/três fases e dois cortes.

`manifesto_teste_congelado.json` contém as configurações propostas, **aguardando
revisão do usuário**. Inclui os 12 pesos originais com limiares de validação já
refinados; as versões brutas também permanecem preservadas. O avaliador por
manifesto suporta até três novos, mas a recomendação atual inclui zero novos.
O teste não foi usado para decisão nem executado. Não houve merge ou fechamento
da Issue #4.
