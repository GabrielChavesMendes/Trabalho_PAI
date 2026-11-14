# Módulos de desenvolvimento (temporários)

Aqui serão construídos os componentes separadamente para testes e aprendizado:

- `modulo_interface.py` — GUI com NIfTI e máscara; letra A.
- `modulo_dataset.py` — CSV, leitura DCE/mask, 1/3 fases, cortes e augmentação; letras B/C.
- `modulo_redes.py` — CNN autoral, ResNet50 e MobileNetV2; letra D.
- `modulo_treinamento.py` — treinamento, validação, métricas e checkpoints; letra D.
- `modulo_gradcam.py` — explicabilidade e visualização; letra E.

**O módulo `modulo_dataset.py` já possui a primeira implementação de validação do CSV e construção de manifesto de arquivos.** Os demais módulos ainda precisam ser implementados: Na entrega, o código precisa estar integrado em **um único arquivo Python** funcional.
