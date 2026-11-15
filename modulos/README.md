# Módulos de desenvolvimento (temporários)

Aqui serão construídos os componentes separadamente para testes e aprendizado:

- `modulo_interface.py` — GUI com NIfTI e máscara; letra A.
- `modulo_dataset.py` — CSV, leitura DCE/mask, 1/3 fases, cortes e augmentação; letras B/C.
- `modulo_redes.py` — CNN autoral, ResNet50 e MobileNetV2; letra D.
- `modulo_treinamento.py` — treinamento, validação, métricas e checkpoints; letra D.
- `modulo_gradcam.py` — explicabilidade e visualização; letra E.

**`modulo_dataset.py` implementa** validação do CSV, seleção de partições por paciente, caminhos temporais, leitura de volumes NIfTI, seleção de dois cortes e saída PyTorch `[1,H,W]` ou `[3,H,W]`. Testes em `tests/test_modulo_dataset.py`. As demais etapas serão integradas gradualmente. Na entrega, os módulos precisam ser consolidados em **um único arquivo Python** funcional.
