# Trabalho Prático — Processamento e Análise de Imagens (PAI)

**PUC Minas — classificação da resposta patológica completa (pCR) ao tratamento de câncer de mama usando DCE-MRI.**

> **Estado:** preparação e implementação em andamento. A branch `develop` contém a estrutura inicial, sem substituir a implementação anterior preservada na `main`.

## Objetivo
Treinar e avaliar **três modelos** (CNN autoral ≥5 camadas com parâmetros, ResNet50 ImageNet e MobileNetV2) para prever pCR a partir de imagens DCE-MRI. Comparar uma fase temporal isolada com a combinação de três fases e ao menos duas estratégias de cortes de volumes 3D.

## Dados
Dataset obrigatório: **BreastDCEDL ISPY1 MinCrop**, disponível em https://zenodo.org/records/18114231 — baixar exclusivamente `BreastDCEDL_ISPY1_min_crop.tar.gz` (~1,3GB).

A estrutura local recomendada é:
```text
dados/
├── BreastDCEDL_SPY1_min_crop_metadata.csv
└── BreastDCEDL_ISPY1_min_crop/
    ├── dce/
    └── mask/
```
O CSV oficial do Canvas possui **172 pacientes I-SPY1** (123 pCR=0, 49 pCR=1). O CSV `dataset/BreastDCEDL_metadata_min_crop.csv` que já existia no repositório contém **2070 registros** de vários conjuntos; **não treinar indiscriminadamente com todos**. Os 172 registros I-SPY1 conferem com o CSV do Canvas.

**Não incluir os arquivos de imagem nem pesos no Git ou no ZIP final.** A pasta `dataset/` histórica permanece no repo por enquanto, mas o novo `.gitignore` impede a adição de novos arquivos nessa pasta; arquivos já versionados continuam rastreados até remoção explícita.

## Começando
1. Instalar Python 3.10+ e ambiente virtual.
2. Instalar dependências: `pip install -r requirements.txt`. Para PyTorch com GPU NVIDIA, escolher a distribuição compatível com o driver no site oficial.
3. Baixar e extrair dataset localmente em `dados/` e copiar CSV do Canvas para `dados/`.
4. Implementar e executar cada etapa via as instruções do guia em `docs/GUIA_DO_TRABALHO.md`. Os módulos ainda são **temporários** e deverão ser integrados em **um único Python** na entrega.

## Pastas
- `docs/`: enunciado local e guia de conceitos/execução.
- `modulos/`: desenvolvimento das letras A–E antes da integração.
- `dados/`: dataset e CSV locais, ignorados pelo Git.
- `resultados/`: tabelas e gráficos compactos dos experimentos.
- `checkpoints/`: pesos treinados, ignorados pelo Git.
- `entrega_final/`: material a empacotar para Canvas.

## Fluxo Git
Criar branches `feature/*` a partir de `develop`, enviar PRs para `develop`, integrar/testar; somente depois promover para `main`. Preservar trabalhos antigos para referência e refatorar/reutilizar apenas trechos validados com requisitos.

## Entrega e apresentação
- **16/11/2026 até 19h, sem atraso.**
- Código **Python em arquivo único**, relatório **SBC/LaTeX + PDF até 12 páginas** e declaração de IAG, empacotados em **Gnn.zip ≤10 MB**.
- Dataset e checkpoints **não entram no ZIP**.
- Na tarefa separada **Modelo do Trabalho**, subir pesos da melhor configuração para avaliação com dataset surpresa.
- Consultar [issues #1–#8](https://github.com/GabrielChavesMendes/Trabalho_PAI/issues) para tarefas completas.

**Importante:** manter o repositório privado, conforme o enunciado; e registrar as ferramentas de IA efetivamente utilizadas, respeitando a autorização relatada pelo grupo.
