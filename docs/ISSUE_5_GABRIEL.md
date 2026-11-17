# Issue #5 — Inferência e Grad-CAM na interface

## Escopo desta etapa independente da Issue #4

A branch `Gabriel-Issue5` parte de `develop` e implementa o núcleo de Grad-CAM sem assumir qual checkpoint será escolhido ao final da Issue #4.

### Implementado nesta branch

- `modulos/modulo_gradcam.py`: geração de Grad-CAM para logit positivo (pCR=1), com suporte a CPU/CUDA.
- Resolução da camada-alvo para as arquiteturas atuais CNN, ResNet50 e MobileNetV2.
- Registro temporário de hook de forward e captura do gradiente na ativação; remoção garantida do hook em bloco `finally`.
- Normalização do mapa e redimensionamento para a resolução da entrada.
- Função auxiliar para sobrepor o mapa a uma imagem 2D.
- Testes sintéticos independentes de dataset NIfTI real.

### Testes

Executar na raiz do repositório:

```powershell
python -m pytest tests/test_issue5_gradcam.py -q
```

Os testes usam tensores artificiais e não comprovam a qualidade explicativa em pacientes reais.

## Dependências da Issue #4 — ainda pendentes

- Confirmar arquitetura, checkpoint definitivo, fases, estratégia de corte, normalização e limiar selecionados.
- Instanciar o modelo correto e carregar os pesos definitivos sem retreinar na ação de classificação.
- Validar compatibilidade do checkpoint e seus metadados.
- Garantir que a inferência reproduza exatamente o pipeline de treinamento.

## Trabalho restante na Issue #5

- Integrar o núcleo de Grad-CAM à interface Tkinter existente.
- Permitir seleção de paciente/imagem e acionar inferência com o checkpoint final.
- Exibir imagem original, mapa sobreposto, probabilidade pCR=1 e decisão com limiar explícito.
- Exibir a máscara do mesmo paciente e do mesmo corte.
- Selecionar e documentar exemplos reais de TP, TN, FP e FN, após confirmar o checkpoint.
- Tratar arquivos ausentes, volumes inválidos, arquitetura incompatível e checkpoint inválido.
- Documentar que Grad-CAM não é segmentação tumoral nem prova de causalidade.

## Limites de interpretação

O mapa Grad-CAM representa relevância espacial aproximada para o logit da classe positiva. Não equivale a uma segmentação, não identifica necessariamente o tumor e não demonstra causalidade. A comparação com a máscara deve ser visual e interpretada com cautela.

## Estado

Núcleo genérico implementado e testes sintéticos adicionados. A integração completa permanece bloqueada pela escolha final do checkpoint e respectivos metadados na Issue #4.
