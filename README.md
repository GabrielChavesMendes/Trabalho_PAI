# Trabalho Prático — Processamento e Análise de Imagens (PAI)

**Grupo G10 — PUC Minas — classificação da resposta patológica completa (pCR) ao tratamento de câncer de mama usando DCE-MRI.**

> **Estado (17/11/2026):** a execução experimental da Issue #4 foi concluída na branch `feature/felipe-issue4-finalizacao` (27 checkpoints, avaliação final com 35 pacientes, 12 representantes). A integração à `develop`, a interface final/Grad-CAM e o arquivo Python único ainda exigem revisão. Resultados: [`docs/GRADE_DEFINITIVA_ISSUE_4_20261117.md`](docs/GRADE_DEFINITIVA_ISSUE_4_20261117.md).

## Integrantes e professor

**Grupo G10**

- Felipe Barros Ratton de Almeida
- Gustavo Henrique Rodrigues de Castro
- Gabriel Chaves Mendes
- Alessandra Faria Rodrigues

**Professor:** Alexei Manso Correa Machado

## Objetivo
Treinar e avaliar **três modelos** (CNN autoral ≥5 camadas com parâmetros, ResNet50 ImageNet e MobileNetV2) para prever pCR a partir de imagens DCE-MRI. Comparar uma fase temporal isolada com a combinação de três fases e ao menos duas estratégias de cortes de volumes 3D.

## Dataset — leitura obrigatória para todos os integrantes

**Download oficial (link exato do enunciado):** https://zenodo.org/records/18114231/files/BreastDCEDL_ISPY1_min_crop.tar.gz?download=1 (~1,3 GB). Cada integrante deve baixar este arquivo uma vez e extrair localmente.

**Documentação:** consultar o artigo e os READMEs BreastDCEDL fornecidos no Canvas, especialmente o da versão MinCrop.

## Dados
Dataset obrigatório: **BreastDCEDL ISPY1 MinCrop**. Usar o link exato do enunciado apresentado acima. Não baixar os subconjuntos DUKE ou ISPY2.

A estrutura local recomendada é:
```text
dataset/
├── BreastDCEDL_SPY1_min_crop_metadata.csv
└── BreastDCEDL_ISPY1_min_crop/
    ├── dce/
    └── mask/
```
O CSV oficial do Canvas possui **172 pacientes I-SPY1** (123 pCR=0, 49 pCR=1). O CSV antigo de 2.070 pacientes foi removido da branch `develop` porque o projeto utiliza **somente os 172 pacientes I-SPY1**; o histórico Git conserva o arquivo antigo.

**Não incluir os arquivos de imagem nem pesos no Git ou no ZIP final.** Os arquivos de exemplo NIfTI e o CSV geral antigo foram removidos da branch `develop` para evitar confusão. Permanecem acessíveis no histórico Git. O `.gitignore` bloqueia a adição de novos volumes.

## Enunciado oficial

O PDF do enunciado já está versionado em **[`docs/enunciado.pdf`](docs/enunciado.pdf)**.

## Começando
1. Instalar Python 3.10+ e ambiente virtual.
2. Instalar dependências: `pip install -r requirements.txt`. Para PyTorch com GPU NVIDIA, escolher a distribuição compatível com o driver no site oficial.
3. Baixar o arquivo do **link oficial exato** acima e extrair em `dataset/BreastDCEDL_ISPY1_min_crop/`, de modo que `dce/` e `mask/` estejam imediatamente dentro dessa pasta. Se o descompactador produziu duas pastas com o mesmo nome, mover apenas a pasta interna que contém `dce/` e `mask/`. O CSV oficial do Canvas já está versionado em `dataset/`.
4. Implementar e executar cada etapa via as instruções do guia em `docs/GUIA_DO_TRABALHO.md`. Os módulos ainda são **temporários** e deverão ser integrados em **um único Python** na entrega.

## Tkinter no Windows

`tkinter` integra a distribuição tradicional de Python para Windows; **não é um pacote para listar no `requirements.txt`**. Antes de trabalhar com a GUI, testar: `python -m tkinter`. Se não abrir uma janela, reparar/reinstalar Python com suporte Tcl/Tk. Em Linux é comum instalar `python3-tk` pelo gerenciador do sistema, não por pip.

## Primeira implementação disponível — Issue #2

O módulo `modulos/modulo_dataset.py` valida CSV/caminhos, abre os volumes NIfTI reais, seleciona cortes 2D (central ou maior área tumoral) e prepara tensores PyTorch de 1/3 canais. Na raiz do projeto:

```bash
python modulos/modulo_dataset.py
python modulos/modulo_dataset.py --verificar-arquivos
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase tres --estrategia maior_tumor
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase pre --estrategia central
python -m pytest -q
```

O segundo comando verifica os caminhos de todas as fases DCE e máscaras; os dois comandos seguintes **carregam imagens reais** e verificam cortes/canais e normalização. Os testes automatizados do dataset incluem exemplos sintéticos e regressão do eixo Z. **Treinamento e augmentation já foram executados para a Issue #4 na branch de finalização**; os 58 testes automatizados da coorte experimental passaram, conforme relatório. A integração final permanece pendente.

Partições oficiais da coluna `test`: `0=treino` (104 pacientes), `1=teste` (35) e `2=validação` (33), conforme o artigo do BreastDCEDL. Cada paciente permanece numa única partição.

**Correção importante de orientação (15/11):** os volumes reais estão no formato `[Z,256,256]`, por exemplo `[30,256,256]`. O corte deve ser extraído como `volume[z,:,:]`, não `volume[:,:,z]`. O código e os testes foram corrigidos após os primeiros testes reais; **reexecutar os comandos antes do merge**. A máscara é uma segmentação única do paciente (sempre um canal), mesmo com 3 fases DCE. Validar a visualização no MRIcro com o dataset real antes do treinamento.

Metodologia e decisões para o relatório: [`modulos/documentacao/documentacao_dataset.md`](modulos/documentacao/documentacao_dataset.md).

## Pastas
- `docs/`: guia de conceitos/execução e o PDF original `docs/enunciado.pdf`.
- `modulos/`: desenvolvimento das letras A–E antes da integração.
- `dataset/`: CSV oficial versionado; pasta `BreastDCEDL_ISPY1_min_crop/` baixada localmente e ignorada pelo Git.
- `resultados/`: tabelas e gráficos compactos dos experimentos.
- `checkpoints/`: pesos treinados, ignorados pelo Git.
- `entrega_final_G10/`: material final a empacotar para Canvas; comprimir o conteúdo como `G10.zip`.

## Fluxo Git
Criar branches `feature/*` a partir de `develop`, enviar PRs para `develop`, integrar/testar; somente depois promover para `main`. Preservar trabalhos antigos para referência e refatorar/reutilizar apenas trechos validados com requisitos.

## Entrega e apresentação
- **16/11/2026 até 19h, sem atraso.**
- Código **Python em arquivo único**, relatório **SBC/LaTeX + PDF até 12 páginas** e declaração de IAG, empacotados em **G10.zip ≤10 MB**.
- Dataset e checkpoints **não entram no ZIP**.
- Na tarefa separada **Modelo do Trabalho**, subir pesos da melhor configuração para avaliação com dataset surpresa.
- Consultar [issues #1–#8](https://github.com/GabrielChavesMendes/Trabalho_PAI/issues) para tarefas completas.

**Importante:** manter o repositório privado, conforme o enunciado; e registrar as ferramentas de IA efetivamente utilizadas, respeitando a autorização relatada pelo grupo.
