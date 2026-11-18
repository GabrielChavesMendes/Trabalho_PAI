# Executar modelos treinados — Issue #4

Na raiz do projeto, em PowerShell ou Git Bash, com a `.venv` existente:

```text
python scripts/verificar_modelos.py
python scripts/inferir_paciente.py --listar
python scripts/inferir_paciente.py --modelo 6 --paciente ISPY1_1009
```

Os comandos usam automaticamente o Python da **`.venv` do projeto**, mesmo que
`python` aponte para a instalação global. Detectam CUDA; se indisponível, usam CPU.
Não instalam pacotes, treinam redes ou ajustam limiares. Para solicitar CPU:
adicionar `--device cpu`. `--device cuda` exige uma GPU CUDA disponível.

## O que significa receber um modelo já treinado?

Precisamos do arquivo **`.pt` com os pesos**, código da arquitetura compatível e
configurações de entrada/saída. Os checkpoints atuais já guardam arquitetura,
fases, corte, normalização e limiar. Hiperparâmetros sozinhos não recuperam os
pesos aprendidos. Não é preciso baixar ImageNet ou treinar novamente: a recarga
reconstrói a rede, carrega `state_dict` com `strict=True` e coloca em `eval()`.

Os quatro recebidos do Gustavo devem permanecer em `checkpoints/integrados_issue4/`:

```text
cnn__post-early__central__seed-42__final-v2.pt
resnet50__post-early__central__seed-42__final-v2.pt
resnet50__post-early__maior-tumor__seed-84__imagenet-resnet-s84.pt
resnet50__tres__maior-tumor__seed-42__final-v2.pt
```

Os outros oito selecionados continuam em seus caminhos originais. O comando
`--listar` informa os 12 e se cada peso está disponível. A fonte dessa lista é
o [catálogo integrado](../resultados/grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.csv),
que não foi alterado: três redes × duas configurações temporais × dois cortes.
Não acrescentamos 2,5D, ensembles ou candidatos novos.

## Dataset

O padrão é o CSV `dataset/BreastDCEDL_SPY1_min_crop_metadata.csv` e a pasta
`dataset/BreastDCEDL_ISPY1_min_crop/`, contendo `dce/` e `mask/`.
Para dados em outro local, configurar `PAI_CSV` e `PAI_DATASET_DIR` antes de executar.
Exemplo em PowerShell:

```powershell
$env:PAI_DATASET_DIR = 'D:\dados\BreastDCEDL_ISPY1_min_crop'
$env:PAI_CSV = 'D:\dados\BreastDCEDL_SPY1_min_crop_metadata.csv'
```

Não é necessário definir essas variáveis se o dataset já está no local padrão.

## Verificar tudo com um comando

```text
python scripts/verificar_modelos.py
```

Executa, em ordem, interrompendo diante de qualquer erro:

1. Suíte completa de testes, com temporários/cache próprios na `.venv`.
2. SHA-256, metadados, recarga estrita e saída sintética dos quatro pesos em **CPU**.
3. Validação real desses quatro nos **33 pacientes oficiais**, em **CUDA ou CPU**.

A etapa sintética só verifica a estrutura. A etapa real recalcula sensibilidade,
especificidade, precisão, acurácia, F1, AUC, acurácia balanceada, loss e confusão.
Compara com os valores salvos no checkpoint, com tolerância absoluta de 0,001.
Confere também os 33 IDs/rótulos. **Não abre imagens de treino/teste nessa etapa.**

A validação mantém as dez variações de cada paciente; as métricas usam a média
das probabilidades por paciente. São 33 pacientes, não 330 independentes.
O limiar vem de `selecao_validacao.limiar`, sem novo ajuste ou arredondamento.
O batch de inferência é no máximo quatro para caber na GPU de 4 GB; os pesos,
o batch de treinamento registrado e os metadados não são modificados.

Cada chamada cria uma saída exclusiva em `resultados/verificacoes_issue4/<data-hora>/`:
logs, `resumo.json`, relatórios CPU/validação e probabilidades dos 33 pacientes.
O sucesso termina com `APROVADO`. Uma falha exibe sua causa e retorna código não
zero. Para escolher outro destino, usar `--saida pasta_nova`; saídas existentes
são preservadas e recusadas, sem sobrescrita.

Os comandos anteriores continuam disponíveis para verificações isoladas:

```text
.venv/Scripts/python.exe scripts/verificar_checkpoints.py --device cpu
.venv/Scripts/python.exe scripts/verificar_checkpoints.py --device cuda --validar-dataset
```

## Prever um paciente individual

```text
python scripts/inferir_paciente.py --listar
python scripts/inferir_paciente.py --modelo 6 --paciente ISPY1_1009
```

Escolha o número de 1 a 12 mostrado na lista e um ID existente no CSV. Exemplos:
modelo 1 = CNN/uma fase/central; 2 = CNN/uma fase/maior tumor;
6 = ResNet50/uma fase/maior tumor/ImageNet; 8 = ResNet50/três fases/maior tumor;
11 = MobileNetV2/três fases/central. Para fornecer o arquivo diretamente:

```text
python scripts/inferir_paciente.py --checkpoint checkpoints/integrados_issue4/cnn__post-early__central__seed-42__final-v2.pt --paciente ISPY1_1009
```

O programa lê apenas as imagens do paciente escolhido, usa as fases e o corte
do checkpoint, aplica percentis 1/99 conjuntos e, quando registrado, ImageNet.
Na ResNet50 monocanal ImageNet, média 0,449 e desvio 0,226. A validação aplica
essa padronização **antes** das rotações/espelhamentos, como no treinamento.
A máscara seleciona o corte; não vira canal nem filtro da imagem.

A previsão individual não aplica augmentation. Exibe arquivo, device, fases,
corte Z, normalização, probabilidade, limiar completo e classe pCR=0/1.
Usa `inference_mode()`, sem gradientes ou atualização de parâmetros. O rótulo do
CSV não entra na previsão. Uma previsão individual pode diferir da média das
dez transformações usada nas tabelas de validação; isso é esperado.

## Demonstração mínima ao professor

```text
python scripts/verificar_modelos.py
python scripts/inferir_paciente.py --modelo 6 --paciente ISPY1_1009
```

No computador verificado: **72 testes passaram**, quatro recargas CPU e quatro
validações CUDA reproduzidas. Os 12 selecionados também foram recarregados em CPU
e produziram previsão CUDA para esse paciente. Com o modelo 6, a previsão única
registrada foi 0,3113216161727905, limiar 0,25, classe pCR=1.
[Evidências da execução](../resultados/verificacao_integrados_issue4_20261117_01/README.md).

**A demonstração gráfica definitiva e o Grad-CAM serão concluídos na Issue #5.**
Este terminal demonstra a inferência; não substitui essa interface.
Não incluir dataset ou pesos no Git ou no ZIP principal. Consultar a
[síntese integrada](RESULTADOS_FINAIS.md) para resultados e limitações.

Código/documentação desta integração tiveram auxílio de Codex/OpenAI; os
integrantes devem revisar ambos e registrar esse auxílio na declaração de IAG.
