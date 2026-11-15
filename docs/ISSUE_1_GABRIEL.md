# Issue #1 — Interface NIfTI, máscara e aumento de dados

## Implementado nesta branch

- Interface Tkinter/Matplotlib para abrir arquivos `.nii` e `.nii.gz`, separadamente ou como par.
- Visualização lado a lado do plano XY, navegação pelo eixo Z e informação do índice atual.
- Validação de volumes 3D, valores finitos, dimensões, affine e identificador de paciente inferido do nome padrão I-SPY1.
- Aumento geométrico sincronizado para imagem e máscara: rotações de -20°, -10°, 0°, 10° e 20°, cada uma na versão original e espelhada (10 variantes).
- Interpolação linear para imagem e nearest-neighbor para máscara, preservando rótulos discretos.
- Adaptador PyTorch sob demanda que expande exclusivamente a partição de treino; validação/teste não são aumentados.
- Contagem pCR por partição e cálculo de `pos_weight = negativos/positivos` somente a partir do treino.
- Testes automatizados para as transformações, sincronização, multiplicação de amostras, isolamento de partição e balanceamento.

## Decisões metodológicas

1. Os volumes MinCrop descritos na Issue #2 são `[Z,H,W]`; o visualizador usa `volume[z,:,:]` para exibir o plano XY. Não reutiliza o eixo 2 da interface antiga.
2. As fases DCE devem receber exatamente a mesma transformação geométrica. A máscara usa interpolação de ordem zero para não criar classes fracionárias.
3. A lista de 10 variantes inclui 0° original (imagem original) e 0° espelhada.
4. Aumento é aplicado apenas ao treino. A validação deve refletir dados não vistos e não deve ter suas métricas infladas por variantes duplicadas; teste permanece intocado.
5. O peso de classe é calculado após selecionar pacientes de treino, nunca usando teste/validação. O valor deve ser passado à função de perda pelo módulo de treinamento.

## Integração com Issue #2

A branch da Issue #2 ainda está em PR. O adaptador `DatasetDCEAumentado` recebe um `DatasetDCE` já construído a partir dos pacientes da partição de treino:

```python
from modulos.modulo_dataset import DatasetDCE, selecionar_particao
from modulos.modulo_dataset_issue1 import DatasetDCEAumentado

treino = selecionar_particao(manifesto, "treino")
base_treino = DatasetDCE(treino, fase="tres", estrategia="maior_tumor")
dataset_treino = DatasetDCEAumentado(base_treino, particao="treino")
```

Para validação e teste, usar `DatasetDCE` diretamente, sem `DatasetDCEAumentado`.

## Execução

```bash
pip install -r requirements.txt
python -m pytest -q
python -m modulos.interface_nifti
```

Tkinter faz parte da distribuição padrão do Python para Windows; em Linux pode exigir o pacote Tcl/Tk do sistema. Os volumes reais (1,3 GB) não são versionados e precisam estar baixados localmente para a validação visual final.

## Pendências antes de fechar a Issue

- [ ] Executar a GUI com volumes reais e conferir a orientação no MRIcro.
- [ ] Confirmar em imagens reais que o affine/orientação exibidos correspondem ao MRIcro; o visualizador mostra o array na ordem de armazenamento [Z,H,W], não faz reorientação canônica.
- [ ] Integrar e executar contra o `modulo_dataset.py` final após merge/revisão da Issue #2.
- [ ] Executar os testes no ambiente do grupo e validar em Windows com Tcl/Tk.
- [ ] Gerar capturas de tela reais para relatório (não usar imagens sintéticas como evidência clínica).
- [ ] Conectar o `pos_weight` ao critério de perda no módulo de treinamento.
- [ ] Confirmar a política final de aumento de validação com o professor; a implementação deliberadamente não aumenta validação para preservar avaliação não inflada.

## Fora do escopo desta branch

Treinamento, comparação de arquiteturas, Grad-CAM e integração final em arquivo Python único permanecem nas respectivas issues/etapas.
