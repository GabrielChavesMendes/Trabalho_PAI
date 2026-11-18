# Entrega final — Grupo G10

Este diretório reúne os materiais da entrega ao Canvas. **Não confunda a pasta de entrega com todo o histórico experimental do GitHub.**

## Arquivos previstos para o ZIP

- `trabalho_pai.py`: código Python monolítico com preparação, três modelos, treino básico, GUI, inferência e Grad-CAM.
- `relatorio.pdf`: artigo SBC, máximo de 12 páginas.
- `relatorio.tex`, `referencias.bib` e **apenas as figuras referenciadas**: fontes necessárias para recompilar o artigo.
- `Declaracao_Uso_IA_Trabalho_PAI_G10.pdf`: declaração de utilização de IA.
- `Declaracao_Uso_IA_Trabalho_PAI_G10.docx`: versão editável adicional; confirmar se deve entrar no ZIP, pois o professor pode não exigir o Word.
- Identificação de integrantes (nomes, matrículas, curso e campus) nos arquivos pertinentes.

**O ZIP externo deve se chamar `G10.zip`, ter no máximo 10 MB e não incluir volumes DCE-MRI, máscaras NIfTI ou checkpoints `.pt`.** Os pesos foram destinados à tarefa separada *Modelo do Trabalho*. Comprimir apenas o conteúdo necessário da pasta, não as figuras históricas de `material_artigo/` que não aparecem no LaTeX.

## Execução

```bash
python entrega_final_G10/trabalho_pai.py
python entrega_final_G10/trabalho_pai.py --treinar --grade
```

O monolítico requer o CSV I-SPY1 oficial e os volumes extraídos localmente. Ele executa as 12 **combinações básicas**, mas não reproduz sozinho todos os experimentos históricos nem lê indiscriminadamente qualquer CSV novo.

**Antes de apresentar:** verificar abertura da GUI, carregamento estrito do checkpoint definitivo, inferência em pacientes conhecidos, alinhamento de máscara/Grad-CAM, treinamento com salvamento de um checkpoint e leitura do relatório. Nunca modificar o código entregue sem manter uma cópia da versão submetida.

O enunciado apresenta orientação divergente quanto a arquivos na raiz ou pasta interna por matrícula; confirmar com o professor quando possível.
