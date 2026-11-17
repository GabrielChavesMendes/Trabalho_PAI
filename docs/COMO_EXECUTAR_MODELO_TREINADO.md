# Como executar um modelo já treinado (sem treinar novamente)

Este guia explica como executar modelos recebidos de outro integrante, como os quatro checkpoints da Issue #4. **Não é necessário treinar novamente.**

## O que precisamos receber?

1. **Pesos treinados** (arquivo `.pt`): contêm os parâmetros aprendidos.
2. **Arquitetura/código compatível**: por exemplo, CNN ou ResNet50, com a entrada e a saída esperadas.
3. **Configuração de pré-processamento**: fases temporais, corte, normalização, canais e limiar de decisão. Nos checkpoints atuais, esses dados estão salvos dentro do próprio `.pt`.
4. **Imagens reais** e, para calcular métricas, os **rótulos corretos** do CSV.

> Receber **só os hiperparâmetros ou a configuração, sem o arquivo de pesos treinados**, não permite recuperar um modelo já treinado. Nesse caso seria necessário obter os pesos ou realizar novo treinamento.

## O que acontece ao executar?

O programa carrega a rede com os **pesos já aprendidos**, aplica nas imagens **o mesmo pré-processamento usado no treinamento** e calcula uma probabilidade de pCR. Usa-se o limiar escolhido na validação para gerar a classificação 0/1.

Para avaliar um conjunto rotulado, comparamos as previsões com os rótulos e calculamos **sensibilidade, especificidade, precisão, acurácia, F1**, AUC e acurácia balanceada. **Isso é avaliação/inferência, não treinamento.** O código não executa `backward()` nem atualiza os pesos.

## Exemplo do projeto: os quatro modelos integrados da Issue #4

Na raiz do repositório, coloque estes quatro arquivos recebidos em `checkpoints/integrados_issue4/`:

```text
cnn__post-early__central__seed-42__final-v2.pt
resnet50__post-early__central__seed-42__final-v2.pt
resnet50__post-early__maior-tumor__seed-84__imagenet-resnet-s84.pt
resnet50__tres__maior-tumor__seed-42__final-v2.pt
```

O script `scripts/verificar_checkpoints_integrados_issue4.py` foi feito especificamente para **esses quatro modelos**. Ele confere a integridade dos arquivos (SHA-256), os metadados e a recarga estrita dos pesos.

### 1. Conferir o código (Git Bash, Windows)

```bash
mkdir -p .venv/pytest-tmp
export PYTEST_DEBUG_TEMPROOT="$(pwd)/.venv/pytest-tmp"
./.venv/Scripts/python.exe -m pytest -q -o cache_dir=.venv/pytest-cache
```

### 2. Carregar os pesos sem abrir o dataset

```bash
./.venv/Scripts/python.exe scripts/verificar_checkpoints_integrados_issue4.py --device cpu
```

Esse comando executa **apenas uma verificação estrutural**, com entrada artificial. Não treina e **não mede desempenho real**. O esperado é exibir `Recarga estrita OK` para cada um dos quatro arquivos.

### 3. Reproduzir os resultados na validação real

O dataset oficial deve estar na pasta `dataset/BreastDCEDL_ISPY1_min_crop/` com `dce/` e `mask/`, e o CSV oficial deve estar em `dataset/`. Alternativamente, configurar as variáveis `PAI_DATASET_DIR` e `PAI_CSV`.

```bash
./.venv/Scripts/python.exe scripts/verificar_checkpoints_integrados_issue4.py --device cuda --validar-dataset
```

Se a GPU não estiver disponível, substitua `--device cuda` por `--device cpu`.

O script recalcula as métricas nos **33 pacientes de validação**, usando as configurações registradas dentro de cada checkpoint, e compara os resultados com os metadados salvos. Ele **não treina** e **não avalia o conjunto de teste**. Uma divergência de métricas gera erro para podermos investigar diferenças no pré-processamento ou no ambiente.

A classificação é obtida usando a probabilidade e o limiar salvo; não é recalibrado um novo limiar. Esse teste não salva um novo modelo.

## Como usar o modelo depois, em uma imagem nova?

A lógica é a mesma: carregar arquitetura e pesos, preparar a imagem exatamente como foi preparada no treino (fases, corte, normalização, tamanho), executar `modelo.eval()`, aplicar `sigmoid` à saída e apresentar a probabilidade de pCR. Não precisamos do rótulo verdadeiro para fazer uma previsão; ele é necessário apenas para avaliar se a previsão foi correta.

A integração dessa previsão com a **interface gráfica e o Grad-CAM** é tratada na Issue #5; o script de verificação da Issue #4 é destinado à recarga e à conferência das métricas, não à interface para escolher pacientes.

**Importante:** o ZIP principal da entrega não deve incluir o dataset nem os pesos; os checkpoints são distribuídos separadamente conforme o enunciado. Consulte também [a síntese da Issue #4](SINTESE_INTEGRADA_ISSUE_4.md).
