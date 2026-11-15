# Guia de estudo e implementação — letras A–E

Este guia é um roteiro de entendimento do grupo. O enunciado PDF original está versionado em `docs/enunciado.pdf`.

## Conceitos principais
- **pCR:** pathological complete response, resposta patológica completa ao tratamento. Classe 0 = ausente; classe 1 = presente.
- **DCE-MRI:** volumes 3D de ressonância adquiridos em diferentes momentos em relação ao contraste. Entradas dos classificadores.
- **Mask:** volume 3D com segmentação tumoral. Usado para visualização, para a estratégia de corte com maior área e para comparar com Grad-CAM. Não é o rótulo pCR.
- **CSV:** metadados; associa `pid` a `pCR`, `test`, índices das fases `pre`, `post_early`, `post_late`. Não usar campos clínicos/demográficos na entrada do modelo.

## Letra A — Interface
Abrir arquivos NIfTI, apresentar corte XY, navegar Z; máscara do mesmo paciente ao lado no mesmo Z. Validar orientação em MRIcro.

## Letra B — Preparação/aumento
Exibir distribuição pCR por partição. Aumentar imagens de treino/validação por rotações {-20,-10,0,10,20} graus × {original, espelhada} = 10 variações. Preservar coerência espacial entre canais e máscaras; nunca contaminar conjunto teste. Justificar método de balanceamento.

## Letra C — Extração de cortes
Escolher e comparar duas estratégias previstas: **corte central** e **corte com maior área tumoral**. A região da máscara ajuda a selecionar o segundo. Alternativamente, o enunciado aceita agregar probabilidades de todos os cortes pela média ou máximo. MIP de intensidades não é equivalente a essa agregação.

## Letra D — Redes
1. **CNN autoral:** no mínimo 5 camadas com parâmetros (entrada e pooling não contam).
2. **ResNet50:** iniciar com pesos pré-treinados ImageNet, adaptar entrada (1 ou 3 canais), trocar cabeça e retreinar parte totalmente conectada para saída binária.
3. **Outra rede:** MobileNetV2, adaptada para o mesmo problema.

Cada rede deve comparar **uma fase isolada** versus **três fases conjuntas**; as três fases podem ser empilhadas como 3 canais. Escolher melhores configurações com base na **validação**, avaliar no teste reservado e salvar pesos e metadados para a apresentação.

Métricas: **sensibilidade, especificidade, precisão, acurácia e F1**, além de Loss de treino/validação, tempo de treinamento, acertos/erros, ajustes de hiperparâmetros e discussão de overfitting/underfitting.

## Letra E — Grad-CAM
Carregar modelo realmente treinado, prever probabilidade pCR e sobrepor mapa de calor que evidencia regiões influentes para essa previsão. Comparar visualmente com a máscara — Grad-CAM não é segmentação.

## Passos de execução (a detalhar quando código estiver pronto)
1. Configurar ambiente e extrair 1,3 GB do [link exato do enunciado](https://zenodo.org/records/18114231/files/BreastDCEDL_ISPY1_min_crop.tar.gz?download=1) em `dataset/BreastDCEDL_ISPY1_min_crop/` com subpastas `dce/` e `mask/`. CSV oficial já versionado em `dataset/`.
2. Rodar inspeção CSV/pacientes e pré-processamento do dataset.
3. Validar a GUI e o corte/segmentação.
4. Testar formatos dos 3 modelos e treinamento em pequeno subconjunto de treino **apenas para depuração**.
5. Executar treinamento completo e avaliações comparativas, exportar métricas/checkpoints.
6. Demonstrar inferência + Grad-CAM em GUI sem retreinamento.
7. Integrar tudo no Python único, compilar relatório LaTeX e preparar entregas separadas.

## Teste prático do primeiro módulo (#2)

Executar na raiz do repositório:

```bash
python modulos/modulo_dataset.py
python modulos/modulo_dataset.py --verificar-arquivos
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase tres --estrategia maior_tumor
python modulos/modulo_dataset.py --inspecionar-paciente ISPY1_1001 --fase pre --estrategia central
python -m pytest -q
```

O primeiro valida os 172 pacientes e partições; o segundo verifica a presença dos 3 volumes DCE e da máscara por paciente; os dois seguintes abrem NIfTI reais e informam formato `[C,H,W]`, corte Z, classes e intensidade; o último executa 8 testes automatizados. Visualização gráfica é feita na Issue #1, não neste módulo.

## Pendências de validação
- Partição oficial **confirmada no artigo BreastDCEDL**: `test=0` (treino), `test=1` (teste), `test=2` (validação). Artigo: https://pmc.ncbi.nlm.nih.gov/articles/PMC12917191/ .
- Executar a inspeção real das fases temporais, verificar alinhamento (affines), orientação do plano XY e comparação no MRIcro antes de aprovar a leitura real; os índices não são necessariamente 0/1/2.
- Confirmar formato do envio de checkpoint e divergência de nome/estrutura do ZIP junto ao professor.

## Arquivos obrigatórios
Consultar `trab.pdf`: Python único, código com identificação dos integrantes, relatório SBC/LaTeX e PDF até 12 páginas, fontes/figuras, declaração de IA e ZIP até 10MB **sem dataset/pesos**. Os pesos são enviados na tarefa separada.

## Interface Tkinter

No Python padrão do Windows, verificar a instalação com `python -m tkinter`; não instalar por `pip install tkinter`, pois ele não é uma dependência publicada dessa forma no PyPI. Caso esteja ausente, corrigir a instalação do Python/Tcl-Tk.

## Orientação real dos volumes e documentação

Os volumes reais do I-SPY1 MinCrop são `[Z,256,256]`; exemplo observado: `[30,256,256]`. Portanto, o índice de corte varia de 0 a 29, e o corte XY é `volume[z,:,:]`, com 256×256 pixels. A correção do eixo foi feita na `feature/felipe-issue-2` após os primeiros testes práticos. Consulte [`modulos/documentacao/documentacao_dataset.md`](../modulos/documentacao/documentacao_dataset.md) para decisões, normalização, máscara, 1/3 canais e pendências para o relatório.
