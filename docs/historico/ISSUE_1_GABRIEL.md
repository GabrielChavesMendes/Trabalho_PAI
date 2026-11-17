# Issue #1 — Interface NIfTI, máscara e aumento de dados

## Implementação nesta atualização

A branch foi atualizada a partir de `develop` após o merge do PR #14. A implementação da Issue #2 é reutilizada diretamente; não há cópia de leitura de NIfTI, seleção de cortes ou manifesto.

### Interface e integração com a Issue #2
- A interface carrega o CSV oficial e a pasta BreastDCEDL I-SPY1 MinCrop.
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
- A distribuição oficial identificada no CSV é: treino 104 pacientes (78 pCR 0, 26 pCR 1); teste 35 (23 pCR 0, 12 pCR 1); validação 33 (22 pCR 0, 11 pCR 1).
- Para o treino, o valor calculado é `pos_weight = 78 / 26 = 3.0`.
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
- A GUI mostra a ordem de armazenamento do array NIfTI. A orientação visual foi comparada com o MRIcro no ambiente local.
- O aumento de validação segue o roteiro disponibilizado em `docs/GUIA_DO_PROJETO.md`. Para relatar métricas comparáveis, manter explícito se a avaliação usa a validação aumentada ou a amostra original; não usar o conjunto de teste para seleção.
- O PDF original `docs/enunciado.pdf` permanece como referência normativa.
- O módulo desta Issue disponibiliza o cálculo de `pos_weight`, mas não implementa treinamento nem conecta o peso à função de perda. Essa integração pertence ao módulo responsável pelo treinamento, quando disponível.

## Validações realizadas no ambiente local

- [x] Testes específicos da Issue #1: `python -m pytest tests/test_issue1.py -v` — 8 testes aprovados.
- [x] Suíte completa: `python -m pytest -v` — 17 testes aprovados.
- [x] Dataset real: 516 arquivos DCE e 172 máscaras; verificação sem arquivos ausentes.
- [x] Interface aberta com o dataset real; navegação Z sincronizada entre imagem e máscara.
- [x] Exibição de uma fase e das três fases.
- [x] Teste visual com pacientes distintos, incluindo exemplos das classes pCR 0 e pCR 1.
- [x] Contorno da máscara conferido em diferentes cortes.
- [x] Orientação visual comparada com o MRIcro.
- [x] Aumento de dados validado com as 10 variantes; imagens e máscaras mantiveram dimensões compatíveis, valores finitos e máscara binária.
- [x] Aumento validado separadamente para treino (104 pacientes, 1.040 amostras) e validação (33 pacientes, 330 amostras).
- [x] Aumento no conjunto de teste bloqueado corretamente.
- [x] IDs de treino, validação e teste conferidos como partições independentes.
- [x] Distribuição pCR e `pos_weight=3.0` conferidos com os dados reais; partições misturadas rejeitadas.

## Pendências e itens de outras etapas

- A conexão de `pos_weight` à função de perda será realizada quando o módulo de treinamento estiver disponível.
- A forma de apresentar métricas da validação aumentada deve ser mantida explícita no relatório e, se necessário, confirmada com o professor.
- Capturas de tela e evidências visuais podem ser incorporadas ao relatório final.
- A declaração de uso de IA será consolidada posteriormente no documento/relatório final do trabalho; não faz parte do fechamento técnico desta documentação da Issue.

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
- `docs/ISSUE_1_GABRIEL.md` — documentação, resultados dos testes locais e situação das pendências.

### Não alterados
- `modulos/modulo_dataset.py`, `tests/test_modulo_dataset.py` e os arquivos da Issue #2 foram mantidos como estão em `develop`.
- `docs/enunciado.pdf` e `requirements.txt` não foram modificados.

## Fora desta Issue

Treinamento/comparação de arquiteturas, avaliação final, Grad-CAM e consolidação do projeto em um único arquivo Python pertencem às demais etapas do trabalho.
