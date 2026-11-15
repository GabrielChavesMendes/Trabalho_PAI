# Documentação técnica — Issue #2: preparação de imagens BreastDCEDL I-SPY1 MinCrop

**Situação:** implementação disponível na branch `feature/felipe-issue-2`, ainda em revisão no PR #14. **Os resultados de treinamento são tarefa da Issue #4.**

## Guia de conceitos para o grupo (leitura antes de apresentar)

| Termo | O que significa | Uso no trabalho |
|---|---|---|
| **DCE-MRI** | *Dynamic Contrast-Enhanced Magnetic Resonance Imaging*: ressonância magnética dinâmica com contraste; diferentes aquisições registram o tecido antes e após a administração do contraste | É a **imagem de entrada** da rede |
| **Fase temporal** | Uma das aquisições da mesma paciente: `pre` (antes do contraste), `post_early` (pós-contraste precoce), `post_late` (pós-contraste tardio) | Compara-se **uma fase** com **as três fases juntas** |
| **Canal** | Uma camada 2D de intensidade fornecida à rede | `C=1` para uma fase; `C=3` para as três fases. **Não são canais RGB** |
| **Máscara (mask)** | Volume de segmentação que indica onde se encontra a região tumoral, representada aqui por valores binários após processamento | Escolher corte com maior área de tumor, visualizar e comparar com Grad-CAM. **Não é uma quarta fase nem um rótulo pCR** |
| **Voxel** | Equivalente tridimensional do pixel: um ponto do volume `[Z,H,W]` | Cada volume contém voxels; ao escolher um corte, obtemos pixels 2D |
| **NIfTI / `.nii.gz`** | Formato de arquivo de imagem médica tridimensional, compactado com GZIP quando termina em `.gz` | Armazena os volumes DCE e máscaras com informações de geometria espacial |
| **CSV / metadados** | Tabela de pacientes e informações associadas | Informa identificador, rótulo pCR, divisão treino/teste/validação e **quais arquivos correspondem a cada fase** |
| **pCR** | *Pathological Complete Response*: resposta patológica completa após tratamento | **Rótulo-alvo da classificação**: 0 = ausência; 1 = presença de pCR |
| **Corte / slice / Z** | Uma das imagens bidimensionais ao longo do volume | Escolhemos corte central ou aquele com maior área tumoral |
| **Tensor** | Estrutura numérica multidimensional usada pelo PyTorch | A imagem chega às redes como `[C, H, W]`, ou `[N, C, H, W]` em lotes |
| **Normalização** | Reescala dos valores de intensidade para facilitar o processamento numérico | As intensidades da imagem são limitadas a `[0,1]`, **sem se transformarem em probabilidades** |
| **Probabilidade prevista** | Resultado final após treinamento e inferência da rede, por exemplo após aplicar sigmoid ao logit | Um valor entre 0 e 1 para a **pCR=1**; não confundir com intensidades da entrada |

### Exemplo real: paciente `ISPY1_1001`

- Volume de entrada `[30,256,256]`: **30 cortes**, cada um com **altura 256** e **largura 256**. A dimensão 30 **não** significa que a imagem tenha 30 pixels de altura.
- Corte central `z=30//2=15`: o programa extrai `volume[15,:,:]` e obtém uma imagem de **256×256**.
- **Uma fase**: a imagem preparada tem `(1,256,256)`. O número inicial 1 é o **canal temporal**.
- **Três fases**: a imagem preparada tem `(3,256,256)`. O número inicial 3 representa os **três momentos da ressonância**, não três pacientes.
- **Máscara**: `(1,256,256)` em ambos os casos, porque há uma única segmentação do tumor para o corte do paciente, mesmo quando usamos três aquisições DCE.
- **Corte de maior área tumoral**: a seleção usa a máscara tridimensional para contar pixels tumorais em cada corte e escolher o que contém a maior área; não significa que foi encontrado automaticamente por uma rede treinada.
- **Faixa de intensidade `0.0000–1.0000`**: mínimo e máximo dos **pixels de entrada após normalização**. **Não** é probabilidade de pCR, confiança da rede nem qualidade do tratamento.
- **pCR=0** exibido na inspeção vem do **CSV**; é o resultado conhecido utilizado como rótulo durante o treinamento. A rede **ainda não previu nada** nesse teste.

### Da imagem à previsão, em ordem

```text
CSV (identificação e pCR) + DCE (imagens 3D) + máscara (região tumoral 3D)
  -> escolher paciente, fase(s) e corte Z
  -> extrair imagem(s) XY de 256×256
  -> normalizar intensidades para [0,1]
  -> gerar tensor de 1 ou 3 canais (entrada do modelo)
  -> [Issue #4] treinar a rede comparando previsão com pCR do CSV
  -> [Issue #4/#5] obter logit e aplicar sigmoid para probabilidade estimada de pCR=1
```

**Por que usar a máscara sem passá-la como entrada?** Nossa implementação usa a máscara *antes* da rede apenas para selecionar o corte de maior tumor. A tarefa opcional de acrescentar a máscara como canal ou remover o fundo é diferente e **não foi implementada** nesta issue.

---

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
O código inicialmente passou 8 testes automatizados, mas os exames reais revelaram o erro de orientação. Após a correção do eixo Z e a inclusão de um **nono teste de regressão**, o grupo executou novamente os testes: **9 passed in 2.40s** (15/11/2026). Também inspecionou `ISPY1_1001` usando imagens reais com uma e três fases, ambos os cenários exibindo os formatos corretos:
```bash
python -m pytest -q
python modulos/modulo_dataset.py --verificar-arquivos
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase tres --estrategia maior_tumor
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase pre --estrategia central
```
**Resultados efetivamente observados:** imagem `(3,256,256)`, máscara `(1,256,256)`, `z=15` no cenário de 3 fases/maior tumor; imagem `(1,256,256)`, máscara `(1,256,256)`, corte central `z=15` no cenário de 1 fase/central. Em ambos, intensidade normalizada `0.0000–1.0000`.

### Pendências
1. **Concluído:** reexecução dos 9 testes e inspeção real do paciente `ISPY1_1001` nos cenários 1/3 fases.
2. **Pendente:** conferir imagens e máscaras reais de **outros pacientes**, incluindo fases com índices não consecutivos; validar visualmente a orientação no MRIcro e, quando pronta, na GUI da Issue #1.
3. **Issue #4:** treinar modelos e comparar quantitativamente as duas estratégias de cortes, as configurações 1 fase/3 fases e as três arquiteturas. Registrar métricas, curvas e tempos.

## 8. Decisões para o relatório
Registrar no artigo: os 172 pacientes e a partição oficial, o uso exclusivo de DCE na classificação, o papel auxiliar das máscaras, o significado dos 1/3 canais, o eixo Z correto, as duas estratégias de corte, a normalização por percentis e a prevenção de vazamento. **Não inventar resultados de treinamento** antes da Issue #4.
