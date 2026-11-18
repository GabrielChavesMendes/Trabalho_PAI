# Trabalho PAI — Grupo G10

**PUC Minas — Ciência da Computação — Classificação de pCR por DCE-MRI.**

**Equipe:** Alessandra Faria Rodrigues, Felipe Barros Ratton de Almeida, Gabriel Chaves Mendes e Gustavo Henrique Rodrigues de Castro.  
**Professor:** Alexei Manso Correa Machado.

## Estado do projeto

As Issues de implementação foram integradas à `develop`: preparação dos dados, três arquiteturas, treinamento experimental, inferência, interface Tkinter e Grad-CAM. A entrega do Canvas é separada deste repositório e não depende da inclusão dos pesos ou dos volumes de imagem no Git.

- [Guia e índice da documentação](docs/README.md)
- [Resultados finais: 12 representantes](docs/RESULTADOS_FINAIS.md)
- [Histórico e justificativas dos experimentos](docs/EXPERIMENTOS_E_DECISOES.md)
- [Como executar/checkpoints](docs/COMO_EXECUTAR.md)
- [Arquivo único de entrega](entrega_final_G10/trabalho_pai.py)

## Início rápido

Instale Python 3.10+ e execute `pip install -r requirements.txt`. A GUI usa Tkinter, normalmente presente na distribuição Python do Windows.

**Dados:** baixar o conjunto oficial I-SPY1 MinCrop (aproximadamente 1,3 GB) em
https://zenodo.org/records/18114231/files/BreastDCEDL_ISPY1_min_crop.tar.gz?download=1
e extrair de modo que a pasta `dataset/BreastDCEDL_ISPY1_min_crop/` contenha `dce/` e `mask/`. O CSV oficial de 172 pacientes está em `dataset/BreastDCEDL_SPY1_min_crop_metadata.csv`.

```bash
python entrega_final_G10/trabalho_pai.py
python entrega_final_G10/trabalho_pai.py --treinar --grade
python entrega_final_G10/trabalho_pai.py --treinar --arquitetura resnet50 --fases tres --corte maior_tumor
python -m pytest -q
```

A GUI usa um checkpoint `.pt` compatível, o CSV de metadados e os volumes correspondentes. **Classificar não retreina.** Os pesos são enviados separadamente da entrega principal.

## Validação e execução de modelos já treinados

```bash
python scripts/verificar_modelos.py
python scripts/inferir_paciente.py --listar
python scripts/inferir_paciente.py --modelo 6 --paciente ISPY1_1009
```

Esses scripts dependem dos checkpoints disponíveis localmente. A pasta `checkpoints/` ignora arquivos `.pt` no Git.

## Qual a diferença entre 12 configurações e os 12 checkpoints finais?

`--treinar --grade` percorre as **12 combinações estruturais obrigatórias**: CNN, ResNet50 e MobileNetV2 × uma/três fases × corte central/maior tumor. Isso **não reproduz automaticamente** as 12 variantes de hiperparâmetros escolhidas na investigação histórica (por exemplo, GroupNorm, políticas de congelamento de BN e ajuste fino por blocos), nem reproduz os 27 treinamentos originais. Seus checkpoints e métricas são registros experimentais específicos.

O arquivo único é orientado à demonstração e ao treinamento básico. A reprodução integral das rodadas históricas exige os scripts, os parâmetros e os artefatos documentados em `docs/historico/` e `resultados/`.

**Limitação para datasets novos:** o código único foi validado para o CSV oficial I-SPY1 com 172 pacientes e os caminhos/IDs dessa coorte. Um novo CSV ou conjunto de imagens pode exigir adaptação do carregador e das partições; apenas aumentar a quantidade de linhas não garante compatibilidade.

## Organização

- `entrega_final_G10/`: código único e documentos finais; a entrega em ZIP deve ter no máximo 10 MB e não inclui pesos nem DCE-MRI.
- `docs/`: guias atuais, resultados e justificativas; `docs/historico/` preserva registros detalhados de execução e prompts históricos.
- `modulos/`: componentes reutilizáveis e variantes experimentais.
- `scripts/`: inferência, verificações e reprodução de experimentos.
- `tests/`: regressões automatizadas de dados, modelos e interface/Grad-CAM.
- `resultados/`: gráficos, planilhas e auditorias por rodada.
- `dataset/`: metadados; volumes ficam somente no ambiente local.
- `checkpoints/`: pesos locais (não versionados).

## Entrega

O arquivo `G10.zip` da atividade principal deve reunir código único, relatório SBC em PDF (máx. 12 páginas), fontes LaTeX e declaração fiel de utilização de IA, observando o limite de 10 MB. O checkpoint selecionado é entregue em outra atividade. Ver [README de entrega](entrega_final_G10/README.md).

**Nota:** os comandos históricos arquivados podem citar caminhos anteriores à reorganização. Use os nomes acima para execução atual. A reorganização não alterou o conteúdo do monolítico.
