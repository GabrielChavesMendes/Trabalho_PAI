# Protocolo de treinamento e avaliacao — Issue #4

Este documento define a fronteira entre as Issues #2 e #4. O treinamento nao le
CSV ou NIfTI e nao escolhe fases, cortes, normalizacao ou augmentation. Ele recebe
`DataLoader` preparados pela Issue #2.

## Contrato de integracao

Cada lote precisa fornecer:

```python
imagens, rotulos, pacientes
```

- `imagens`: tensor `float32` com formato `[N,1,H,W]` ou `[N,3,H,W]`;
- `rotulos`: tensor binario convertivel para `[N,1]`;
- `pacientes`: sequencia com um ID para cada imagem.

Um mapping com as chaves `imagens`, `rotulos` e `pacientes` tambem e aceito. Os IDs
sao obrigatorios porque as dez variacoes de augmentation e eventuais multiplos cortes
precisam ser agregados antes das metricas por paciente.

O carregador deve fornecer metadados serializaveis descrevendo, no minimo, as colunas
de fase usadas, a normalizacao, a estrategia de corte, as transformacoes e os codigos
de particao. Esses metadados sao gravados no checkpoint, mas nao interpretados pelo
modulo de treinamento.

O modulo rejeita o inicio do treino se `fases`, `normalizacao`, `corte`,
`augmentation` ou `particoes` estiverem ausentes. A funcao
`calcular_pos_weight_treino` confere a proporcao por paciente; para a particao oficial,
o resultado esperado e `78/26 = 3.0`.

## Protocolo inicial

- Particoes informadas pelo projeto: treino `test=0`, validacao `test=2`, teste
  `test=1`.
- `BCEWithLogitsLoss(pos_weight=3.0)`, calculado apenas a partir de 78 negativos e
  26 positivos do treino.
- Adam; taxa inicial `1e-3` para a CNN e `1e-4` para as cabecas transferidas;
  `weight_decay=1e-4`.
- Batch inicial 8, sujeito a medicao na GPU.
- Ate 30 epocas, parada antecipada com paciencia 5.
- Melhor checkpoint selecionado pela menor loss de validacao.
- Limiar inicial 0,5; a funcao `selecionar_limiar_validacao` pode otimizar F1
  somente com predicoes da validacao.
- Seeds de Python, NumPy, PyTorch e CUDA registradas.
- Backbone congelado inicialmente na ResNet50 e MobileNetV2; BatchNorm congelada
  permanece em modo de avaliacao.

Esses valores sao pontos de partida e so devem ser descritos como resultados depois
de experimentos reais.

## Ordem dos experimentos

| Prioridade | Arquitetura | Configuracao temporal | Corte |
|---:|---|---|---|
| 1 | CNN | `post_early` | central |
| 2 | CNN | tres fases | central |
| 3 | MobileNetV2 | `post_early` | central |
| 4 | ResNet50 | `post_early` | central |
| 5 | MobileNetV2 | tres fases | central |
| 6 | ResNet50 | tres fases | central |
| 7–12 | mesmas tres redes | uma/tres fases | maior area tumoral |

A fase isolada proposta e `post_early`. Se o grupo escolher outra fase, deve manter a
mesma escolha nas tres redes e registrar a decisao.

## Artefatos por configuracao

Usar nomes como:

```text
cnn__post-early__central__seed-42.pt
cnn__post-early__central__seed-42_history.csv
cnn__post-early__central__seed-42_validation.json
```

O checkpoint guarda pesos, estado do otimizador, configuracao, historico, melhor
epoca, ambiente e metadados de preprocessamento. O CSV registra loss e tempo por
epoca. O JSON registra as cinco metricas, matriz de confusao e probabilidades por
paciente. As funcoes de exportacao tambem geram o grafico de convergencia e a matriz
de confusao em PNG.

## Regras de avaliacao

1. Ajustar hiperparametros e limiar somente na validacao.
2. Avaliar o teste reservado apenas depois de selecionar a configuracao.
3. Agregar probabilidades repetidas do mesmo paciente pela media antes das metricas.
4. Registrar sensibilidade, especificidade, precisao, acuracia, F1, `TN/FP/FN/TP`,
   loss e tempo.
5. Nao apresentar smoke tests ou tensores artificiais como resultados.
6. Confirmar a recarga do melhor checkpoint em um processo novo.

## Validacao sem dataset

O comando abaixo testa formatos, forward/backward, CUDA e recarga com dados artificiais:

```bash
python scripts/validar_issue4.py --device cuda --baixar-pesos
```

Ele nao treina uma configuracao real e nao gera metricas para o relatorio.

Quando a Issue #2 expuser uma fabrica com o contrato descrito no inicio de
`scripts/executar_experimento.py`, o primeiro treino real podera ser iniciado por:

```bash
PAI_DATASET_DIR=/mnt/d/PAI/dataset/BreastDCEDL_ISPY1_min_crop \
python scripts/executar_experimento.py \
  --loader-factory modulos.fabrica_carregadores:criar_carregadores \
  --arquitetura cnn --canais 1 --fases post_early --corte central
```

O teste reservado nao e avaliado nesse comando. A opcao `--avaliar-teste` deve ser
usada apenas na configuracao final escolhida pela validacao.

Depois das 12 validacoes, consolidar a comparacao e identificar o checkpoint sem
consultar o teste:

```bash
python scripts/consolidar_resultados.py
```

A grade completa pode ser executada de modo retomavel (configuracoes que ja tenham
JSON de validacao sao ignoradas) com:

```bash
PAI_DATASET_DIR=/mnt/d/PAI/dataset/BreastDCEDL_ISPY1_min_crop \
TORCH_HOME=/mnt/d/PAI/torch \
python scripts/executar_grade.py --device cuda --batch 64
```

Somente entao, recarregar o checkpoint vencedor em outro processo e avaliar uma
unica vez o teste reservado:

```bash
PAI_DATASET_DIR=/mnt/d/PAI/dataset/BreastDCEDL_ISPY1_min_crop \
python scripts/avaliar_teste_final.py \
  --checkpoint checkpoints/ARQUITETURA__FASES__CORTE__seed-42.pt --device cuda
```
