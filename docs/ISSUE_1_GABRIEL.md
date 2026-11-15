# Issue #1 — Interface NIfTI, máscara e aumento de dados

## Implementação nesta atualização

A branch foi atualizada a partir de `develop` após o merge do PR #14. A implementação da Issue #2 é reutilizada diretamente; não há cópia de leitura de NIfTI, seleção de cortes ou manifesto.

### Interface e integração com a Issue #2
- A interface agora carrega o CSV oficial e a pasta BreastDCEDL I-SPY1 MinCrop.
- Seleciona o paciente pelo identificador do CSV, sem presumir índices temporais fixos.
- Permite exibir `pre`, `post_early`, `post_late` ou as três fases juntas.
- Reutiliza `ler_volume` e `selecionar_corte` da Issue #2.
- Exibe as fases e a máscara no mesmo índice Z; a região da máscara é contornada em vermelho sobre cada fase.
- Permite navegar todos os cortes Z e escolher corte inicial central ou de maior área tumoral.
- Confere dimensões e affine de cada fase em relação à máscara.
- Exibe a distribuição de pCR por treino, teste e validação a partir do CSV oficial.

### Aumento de dados (letra B)
- Dez variantes: rotações -20°, -10°, 0°, 10° e 20°, cada uma original e espelhada.
- Todas as fases/canais recebem a mesma transformação espacial.
- Imagem usa interpolação linear; máscara usa nearest-neighbor, preservando rótulos binários.
- O adaptador PyTorch aumenta treino e validação, conforme o guia do trabalho, mas rejeita explicitamente teste.
- A partição é conferida nos objetos `Paciente` do `DatasetDCE`; datasets vazios, mistos ou de teste são rejeitados.
- A geração é sob demanda; as variantes não são armazenadas em memória ou no Git.

### Balanceamento
- `contar_classes` e `resumo_particoes` mostram a distribuição pCR.
- `calcular_pos_weight_treino` calcula negativos/positivos somente com pacientes de treino e rejeita partições misturadas.
- O valor fica disponível para a etapa de treinamento; a conexão com a loss deve ser feita no módulo responsável pelo treinamento, sem duplicar essa lógica nesta Issue.

## Como executar

Na raiz do repositório:

```bash
pip install -r requirements.txt
python -m modulos.interface_nifti
python -m pytest -q
```

Na interface, selecione o CSV oficial, a pasta extraída `BreastDCEDL_ISPY1_min_crop`, carregue os pacientes e abra um paciente. Compare uma fase ou as três fases com a máscara no mesmo Z.

Para uso do aumento com a Issue #2:

```python
from modulos.modulo_dataset import (
    carregar_metadados, criar_manifesto, selecionar_particao, DatasetDCE,
)
from modulos.modulo_dataset_issue1 import DatasetDCEAumentado

tabela = carregar_metadados("dataset/BreastDCEDL_SPY1_min_crop_metadata.csv")
manifesto = criar_manifesto(tabela, "dataset/BreastDCEDL_ISPY1_min_crop")

pacientes_treino = selecionar_particao(manifesto, "treino")
base_treino = DatasetDCE(pacientes_treino, fase="tres", estrategia="maior_tumor")
treino = DatasetDCEAumentado(base_treino, particao="treino")

pacientes_validacao = selecionar_particao(manifesto, "validacao")
base_validacao = DatasetDCE(pacientes_validacao, fase="tres", estrategia="maior_tumor")
validacao = DatasetDCEAumentado(base_validacao, particao="validacao")

# Teste: utilizar DatasetDCE diretamente, sem DatasetDCEAumentado.
```

## Decisões e limites de validação

- Os volumes MinCrop são `[Z,256,256]`; o plano XY é `volume[z,:,:]`.
- A GUI mostra a ordem de armazenamento do array NIfTI. A orientação visual deve ser comparada com MRIcro no computador que contém o dataset; não se deve inferir orientação radiológica apenas pelo `origin` do Matplotlib.
- O aumento de validação segue o roteiro disponibilizado em `docs/GUIA_DO_TRABALHO.md`. Para relatar métricas comparáveis, manter explícito se a avaliação usa a validação aumentada ou a amostra original; não usar o conjunto de teste para seleção.
- O PDF original `docs/enunciado.pdf` permanece como referência normativa. A lista acima segue os requisitos funcionais resumidos no guia do projeto; a conferência final de cada item do PDF e a validação visual com MRIcro devem ser feitas no ambiente do grupo.

## Pendências que dependem do ambiente local

- [ ] Executar `python -m pytest -q` após esta atualização.
- [ ] Executar a interface com os 1,3 GB reais e conferir pacientes/fases no MRIcro.
- [ ] Conferir pacientes com índices temporais não consecutivos e alternar fases/estratégias.
- [ ] Registrar capturas de tela reais para o relatório.
- [ ] Conectar `pos_weight` à função de perda quando o módulo de treinamento estiver disponível e verificar a regra de balanceamento no enunciado integral.
- [ ] Confirmar com o professor a forma de apresentar métricas quando a validação também é aumentada.

## Arquivos desta entrega

### Criados na implementação anterior e mantidos
- `modulos/interface_nifti.py` — interface gráfica.
- `modulos/modulo_augmentacao.py` — transformações geométricas.
- `modulos/modulo_dataset_issue1.py` — adaptador de dataset.
- `modulos/modulo_balanceamento.py` — distribuição e peso de classe.
- `tests/test_issue1.py` — testes da Issue #1.
- `docs/ISSUE_1_GABRIEL.md` — documentação da Issue.

### Alterados nesta atualização
- `modulos/interface_nifti.py` — integração direta com CSV, manifesto, fases e cortes da Issue #2; comparação sincronizada DCE/máscara.
- `modulos/modulo_augmentacao.py` — simplificação de validações e fluxo das transformações, preservando a interpolação adequada.
- `modulos/modulo_dataset_issue1.py` — validação real da partição e suporte a treino/validação, com bloqueio de teste.
- `tests/test_issue1.py` — testes para as duas partições permitidas e rejeição de teste/mistura.
- `docs/ISSUE_1_GABRIEL.md` — documentação, integração e pendências atualizadas.

### Não alterados
- `modulos/modulo_dataset.py`, `tests/test_modulo_dataset.py` e os arquivos da Issue #2 foram mantidos como estão em `develop`.
- `docs/enunciado.pdf` e `requirements.txt` não foram modificados.

## Fora desta Issue
Treinamento/comparação de arquiteturas, avaliação final, Grad-CAM e consolidação do projeto em um único arquivo Python pertencem às demais etapas do trabalho.
