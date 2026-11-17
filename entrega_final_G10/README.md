# Entrega integrada — Grupo G10 (PAI)

## Estado e limites

O arquivo trabalho_pai.py contem interface Tkinter, leitura NIfTI, tres
fases DCE-MRI, selecao de corte, carregamento de checkpoint, inferencia
por paciente, Grad-CAM e modo CLI opcional de treinamento.

**Funcionalidade implementada no codigo nao equivale a validacao com o dataset
e checkpoints reais.** O treinamento nunca e iniciado ao abrir a interface.

## Dependencias

Na raiz do repositorio:

~~~bash
pip install -r requirements.txt
~~~

O dataset I-SPY1 MinCrop deve ser extraido fora do ZIP, com pastas dce/ e
mask/. O CSV oficial de 172 pacientes deve estar disponivel.

## Interface e demonstracao

~~~bash
python entrega_final_G10/trabalho_pai.py
~~~

1. Selecione o checkpoint .pt compativel e clique **Validar e carregar**.
2. Selecione o CSV e um arquivo NIfTI dentro de dce/; a pasta do dataset
   sera identificada automaticamente, com mask/ como pasta irma.
3. Carregue pacientes e escolha um paciente na lista.
4. Use o controle **Corte Z** e **Fase exibida** para navegar pelo volume,
   verificando a imagem e a mascara sobreposta.
5. Clique **Classificar e gerar Grad-CAM**. O aplicativo classifica no corte
   exigido pelo protocolo do checkpoint e move automaticamente o controle Z.
6. Use **Salvar figura para o artigo** no corte classificado para exportar PNG,
   PDF ou SVG em alta resolucao.
7. Ao navegar para outro corte, o Grad-CAM anterior deixa de ser exibido para
   nao confundir cortes distintos. A classe explicada pode ser a classe
   prevista ou o logit da classe positiva (pcr1).

A imagem de visualizacao e normalizada separadamente para exibir contraste
em escala de cinza. **A entrada da rede continua seguindo exatamente o
pre-processamento do checkpoint.** Os tres canais temporais da rede nao
sao equivalentes a uma imagem RGB anatomica.

### Interpretacao do Grad-CAM

- O mapa e calculado sem utilizar a mascara como guia ou filtro.
- O contorno verde representa a mascara tumoral **fornecida pelo dataset**.
- O gradiente explica a classe positiva ou a classe prevista, conforme escolha.
- O diagnostico informa a fracao de ativacao na mascara, fracao de area tumoral,
  tamanho nativo do mapa e maximo da ativacao antes da normalizacao.
- Alta ativacao no fundo deve ser reportada, nao apagada artificialmente.
- Grad-CAM nao e segmentacao, atestado clinico nem prova causal.
- Mapas quase nulos podem indicar ausencia de evidencia positiva na camada.
  Comparar uma camada anterior e um diagnostico, nao correcao de desempenho.

O modo GUI aceita representacoes de fases, com cortes central ou maior_tumor.
Checkpoints que exigem 2,5D, mapas de realce, ROI ou mascara como filtro nao
podem ser carregados sem implementacao e validacao especificas.

## Treinamento opcional no mesmo arquivo Python

Uma configuracao, sem avaliar o teste:

~~~bash
python entrega_final_G10/trabalho_pai.py --treinar \
  --arquitetura cnn --fases tres --corte maior_tumor \
  --epocas 30 --batch 32 --paciencia 3
~~~

A grade completa das 12 combinacoes so roda explicitamente:

~~~bash
python entrega_final_G10/trabalho_pai.py --treinar --grade --epocas 30
~~~

O modo CLI:
- Separa pacientes por codigo oficial: test=0 treino (104), test=2 validacao
  (33), test=1 teste excluido de toda a selecao (35).
- Prepara 10 variacoes de treino e validacao (cinco angulos e espelhamento).
- Usa BCE ponderada e fine-tuning da layer4/cabeca na ResNet50.
- Seleciona epoca pela AUC de validacao e limiar por acuracia balanceada.
- Registra checkpoint, curvas Loss, historico CSV e metricas de validacao.
- Nao consulta o teste para selecionar pesos, epocas ou limiar.

**Esse modo nao reproduz automaticamente os resultados historicos das outras
branches.** Metodologias diferentes nao devem ser comparadas como equivalentes.

## Testes

~~~bash
python -m py_compile entrega_final_G10/trabalho_pai.py
python -m pytest tests/test_issue6_monolito_gradcam.py -q
~~~

Os testes sinteticos nao substituem verificacao com volumes NIfTI reais,
pesos efetivamente treinados e inspecao da orientacao no MRIcro.

## Entrega

O enunciado solicita G10.zip com tamanho maximo de 10 MB, incluindo codigo
Python unico, relatorio SBC em LaTeX e PDF e declaracao fiel de uso de IA.
Volumes e checkpoints nao devem entrar no ZIP. Pesos para avaliacao surpresa
sao entregues separadamente.

O prazo original era 16/11/2026 as 19h. Commits posteriores nao alteram
retroativamente a submissao feita no Canvas.
