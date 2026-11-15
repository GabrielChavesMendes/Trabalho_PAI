# Documentação técnica — Issue #2: preparação de imagens BreastDCEDL I-SPY1 MinCrop

**Situação:** implementação disponível na branch `feature/felipe-issue-2`, ainda em revisão no PR #14. **Os resultados de treinamento são tarefa da Issue #4.**

## 1. Objetivo e dados
Converter volumes tridimensionais DCE-MRI dos **172 pacientes I-SPY1 MinCrop** em entradas bidimensionais para as redes neurais do projeto. O dataset é obtido em [BreastDCEDL_ISPY1_min_crop.tar.gz](https://zenodo.org/records/18114231/files/BreastDCEDL_ISPY1_min_crop.tar.gz?download=1). O CSV oficial `dataset/BreastDCEDL_SPY1_min_crop_metadata.csv` fornece o ID da paciente (`pid`), o rótulo `pCR`, o código de partição (`test`) e os índices das três fases (`pre`, `post_early`, `post_late`).

O CSV possui 172 pacientes: 123 com pCR=0 e 49 com pCR=1. Os códigos oficiais são **`test=0` treino (104), `test=1` teste (35), `test=2` validação (33)**. A separação é feita **por paciente antes do aumento de dados**, para impedir vazamento de informações entre os conjuntos.

## 2. Imagens DCE e máscara
- `dce/`: um volume de intensidade para cada aquisição de cada paciente. A rede classifica **somente estas imagens**.
- `mask/`: uma segmentação tridimensional **única por paciente**, identificando os voxels tumorais. É usada para selecionar o corte de maior área tumoral e pode ser exibida ao lado da imagem. Não representa uma quarta fase temporal nem é um canal de entrada obrigatório.
- O CSV associa volumes às fases. **Os índices temporais nem sempre são 0,1,2**: é necessário consultar os campos por paciente.

## 3. Orientação dos volumes: cuidado essencial
Nos arquivos reais testados pelo grupo, a forma espacial é **`[Z, 256, 256]`**, por exemplo `[30, 256, 256]`. O número 30 indica **30 cortes**, não a altura da imagem. O corte XY correto é obtido com **`volume[z, :, :]`**, resultando em 256×256. O código anterior utilizava `volume[:, :, z]` e gerava equivocadamente 30×256. A correção está no PR #14.

O módulo conserva a matriz affine do NIfTI e verifica se as três fases e a máscara possuem mesmas dimensões e geometria compatível. Isso não substitui a **inspeção visual de orientação no MRIcro** prevista no enunciado.

## 4. Duas estratégias para volumes 3D → cortes 2D
- **Corte central:** escolhe `z = n_z // 2`, sem usar informação sobre a posição do tumor. Para `n_z=30`, o corte central é `z=15` em índices começando no zero.
- **Corte de maior área tumoral:** conta os pixels positivos da máscara em cada corte `mask[z,:,:]` e escolhe `argmax_z`; prioriza a fatia com maior região tumoral visível. A máscara não precisa ser fornecida à rede.
- **MIP não implementada:** é uma projeção das intensidades máximas ao longo do volume e não equivale à estratégia de máximo **entre previsões** de cortes citada no enunciado. As duas estratégias acima cumprem a exigência de investigação; a **comparação quantitativa** ficará para a Issue #4.

## 5. Uma fase, três fases e tensor entregue
- **Uma fase:** escolhemos uma das imagens `pre`, `post_early` ou `post_late`, formato `[1,256,256]`.
- **Três fases:** empilhamos os cortes correspondentes das três aquisições no mesmo índice Z: `[3,256,256]`. Esses canais são **momentos temporais**, não RGB.
- **Máscara:** retorna separadamente como `[1,256,256]`, mesmo quando há três fases, pois a segmentação é única por paciente.
- O `DataLoader` do PyTorch reunirá amostras em lotes com formato `[N,C,256,256]`. A implementação do carregamento individual (`DatasetDCE`) foi concluída; configuração final dos lotes será feita na Issue #4.

## 6. Normalização das intensidades
Para cada amostra, calculamos os **percentis 1 e 99** dos cortes selecionados **em conjunto entre os canais**, redimensionamos linearmente e limitamos os resultados a `[0,1]`. Isso reduz influência de valores extremos e mantém as diferenças relativas entre fases dentro da mesma amostra. Essa escolha **não** equivale a medir a acurácia do modelo; sua adequação será avaliada durante os experimentos.

O intervalo `0.0000–1.0000` impresso pelo programa é o valor **mínimo e máximo da imagem já normalizada**, não a probabilidade de pCR nem a intensidade física original do exame.

## 7. Testes e validação
O código inicialmente passou 8 testes automatizados, mas os exames reais revelaram o erro de orientação. A branch agora contém a correção e um **nono teste de regressão**. Executar novamente e registrar o resultado:
```bash
python -m pytest -q
python modulos/modulo_dataset.py --verificar-arquivos
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase tres --estrategia maior_tumor
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase pre --estrategia central
```
**Resultados esperados após a correção:** imagem `(3,256,256)` e máscara `(1,256,256)` no primeiro exemplo; imagem `(1,256,256)`, máscara `(1,256,256)`, corte central `z=15` no segundo.

### Pendências
1. Reexecutar todos os testes depois da correção.
2. Conferir imagens e máscaras reais de vários pacientes, incluindo fases com índices não consecutivos; validar orientação com MRIcro e com a GUI da Issue #1.
3. Na Issue #4, treinar modelos e **comparar as duas estratégias de cortes**, as configurações 1 fase/3 fases e as três arquiteturas. Registrar métricas, curvas e tempos.

## 8. Decisões para o relatório
Registrar no artigo: os 172 pacientes e a partição oficial, o uso exclusivo de DCE na classificação, o papel auxiliar das máscaras, o significado dos 1/3 canais, o eixo Z correto, as duas estratégias de corte, a normalização por percentis e a prevenção de vazamento. **Não inventar resultados de treinamento** antes da Issue #4.
