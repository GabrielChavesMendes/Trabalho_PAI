# Verificação local da integração da Issue #4 — sem retreino

Branch atualizada por fast-forward de `1df7b16` para `b7e922f` antes do trabalho.
Nenhum treinamento de modelos reais, avaliação do conjunto de teste ou ajuste
de hiperparâmetros/limiares nesta execução. Os testes de treinamento sintético
da suíte continuam existindo, sem tocar nos checkpoints recebidos.

## Resultado aprovado

**Usar `verificacao_final/` como evidência do comando único aprovado.**
`verificacao_final/resumo.json`: status `aprovado`, device `cuda`.

- Antes das alterações: **65 testes aprovados em 6,66 s**, quatro recargas
  estritas CPU e quatro validações CUDA reais aprovadas.
- Comando único depois das alterações: **72 testes aprovados em 6,04 s**,
  quatro recargas CPU e quatro validações CUDA nos mesmos 33 pacientes.
- `inferencia_12_selecionados.json`: os 12 do catálogo final passaram por
  recarga segura/estrita CPU e previsão CUDA do paciente **ISPY1_1009**,
  sem augmentation, sem alterar os arquivos de pesos.
- `inferencia_cli_modelo6.txt`: comando de demonstração executado usando
  `python` global, que relançou corretamente na `.venv`. Probabilidade
  0,3113216161727905, limiar 0,25, classe pCR=1, corte Z=13.

## Métricas reproduzidas — 33 pacientes

| Modelo | Fases/corte | Sens. | Espec. | Precisão | Acurácia | F1 | AUC | BA | Limiar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN Gustavo | post_early / central | 0,5455 | 0,6364 | 0,4286 | 0,6061 | 0,4800 | 0,6033 | 0,5909 | 0,39999999999999997 |
| ResNet50 Gustavo | post_early / central | 0,9091 | 0,1364 | 0,3448 | 0,3939 | 0,5000 | 0,4215 | 0,5227 | 0,2 |
| ResNet50 seed 84 | post_early / maior tumor | 0,5455 | 0,8182 | 0,6000 | 0,7273 | 0,5714 | 0,6983 | 0,6818 | 0,25 |
| ResNet50 Gustavo | três / maior tumor | 0,6364 | 0,6818 | 0,5000 | 0,6667 | 0,5600 | 0,6074 | 0,6591 | 0,05 |

Os JSONs em `verificacao_final/validacao/` têm todas as probabilidades por paciente,
confusões, loss e metadados. `relatorio.json` guarda valores obtidos/salvos,
hashes, fases, cortes, normalização, batch e limiar. A tolerância preexistente de
0,001 não foi ampliada. O batch de inferência foi quatro, sem alterar o batch
de treinamento que permanece no checkpoint. A normalização é percentil 1/99
conjunto em todos; apenas a ResNet seed 84 acrescenta ImageNet monocanal
(0,449/0,226), antes do aumento. Validação: dez variações agregadas por paciente.
Inferência individual: uma imagem, sem aumento. Esses escores podem diferir.

## Compatibilidade e preservação

`modulos/checkpoints_integrados_issue4.py` reutiliza as arquiteturas e os datasets
existentes. O adaptador aceita também o esquema de metadados dos oito pesos
próprios selecionados, sem normalizações adicionais novas. As 12 linhas do
catálogo não foram reescolhidas nem reescritas. A inferência lê fases/cortes e
limiar dos pesos, não os valores arredondados da tabela CSV.

Os hashes antes/depois estão em `integridade_antes.json` e `integridade_depois.json`.
Não foram adicionados pesos, imagens do dataset, 2,5D, ensemble, GUI ou Grad-CAM.
Não houve merge nem fechamento da issue.

## Correções técnicas durante a verificação

O primeiro teste novo assumia padding zero nas rotações; o módulo existente usa
`nearest`. Foi corrigido o teste para observar a normalização na entrada da
rotação, sem alterar o augmentation. O relançamento via `subprocess` preserva
caminhos com espaços. A execução pytest usa raiz/configuração explícitas e
temporários/cache exclusivos na `.venv`, evitando diretórios antigos com ACL
de outro usuário. As tentativas anteriores falharam no teste de software ou
nas permissões/temporários; não modificaram pesos nem resultados anteriores.
Seus logs foram preservados para rastreabilidade. O conjunto aprovado está em
`verificacao_final/`, e nenhuma falha foi contornada relaxando métricas.

## Reproduzir e demonstrar

```text
python scripts/verificar_issue4.py
python scripts/inferir_paciente_issue4.py --listar
python scripts/inferir_paciente_issue4.py --modelo 6 --paciente ISPY1_1009
```

O resultado é por terminal. A demonstração gráfica definitiva e o Grad-CAM são
da Issue #5. Os integrantes devem revisar código, documentação e integração.

## Registro de IAG

Codex/OpenAI auxiliou implementação dos CLIs, testes, execução e documentação.
Pedido do usuário: atualizar apenas a branch de finalização; executar testes,
recarga CPU e validação CUDA dos quatro pesos recebidos; criar comando único e
inferência individual; documentar demonstração; preservar 12 configurações e
resultados; não treinar, avaliar teste, incluir pesos/dataset, fazer merge ou
fechar a issue; finalizar com commit/push e atualização do PR #17. O prompt
completo está registrado na conversa de 17/11/2026. Esse auxílio deve constar
na declaração suplementar da entrega e passar pela revisão dos integrantes.
