import pandas as pd
import numpy as np
from scipy import ndimage


def analisar_balanceamento(caminho_csv):
    """
    Lê o arquivo CSV do dataset para checar a proporção de pacientes
    com Resposta Patológica Completa (pCR = 1) e ausente (pCR = 0).

    O trabalho exige: 'Verifique se existe desbalanceamento entre as classes de pCR.
    Caso exista, proponha uma estratégia justificada'

    Parâmetros:
        caminho_csv (str): O caminho para a planilha CSV do dataset.

    Retorno:
        None (Apenas exibe os resultados no terminal)
    """
    try:
        # Carrega a planilha usando Pandas
        df = pd.read_csv(caminho_csv)

        # Assume-se que a coluna no CSV se chama 'pCR' ou 'class' (ajuste conforme o nome real)
        # Vamos usar 'pCR' como exemplo padrão.
        nome_coluna_classe = 'pCR'
        if nome_coluna_classe not in df.columns:
            print(f"Colunas disponíveis no CSV: {df.columns.tolist()}")
            nome_coluna_classe = input("Qual o nome correto da coluna de classe (pCR) acima? ")

        contagem = df[nome_coluna_classe].value_counts()
        total = len(df)

        ausente = contagem.get(0, 0)
        presente = contagem.get(1, 0)

        print("\n--- ANÁLISE DE DESBALANCEAMENTO ---")
        print(f"Total de pacientes: {total}")
        print(f"pCR Ausente (Classe 0): {ausente} ({(ausente / total) * 100:.2f}%)")
        print(f"pCR Presente (Classe 1): {presente} ({(presente / total) * 100:.2f}%)")

        if abs(ausente - presente) / total > 0.2:
            print("\n[ALERTA] Desbalanceamento relevante detectado (>20% de diferença).")
            print("Estratégia recomendada para o Relatório: Utilização de 'Focal Loss' ou "
                  "pesos balanceados (class_weights) na Função de Perda durante o treinamento "
                  "para penalizar mais fortemente os erros na classe minoritária.")
        else:
            print("\nAs classes estão razoavelmente balanceadas.")

    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")


def aumento_de_dados_regrado(volume_3d):
    """
    Aplica o Aumento de Dados estrito conforme a especificação do trabalho:
    'Espelhamento e pequenas rotações entre -20 e 20 graus, em intervalos de 10 graus.
    Gerando 10 variações: (-20, -10, 0, 10, 20) x (original e espelhada).'

    Parâmetros:
        volume_3d (numpy.ndarray): Matriz 3D da imagem (X, Y, Z)

    Retorno:
        variacoes (list): Uma lista contendo 10 volumes tridimensionais aumentados.
    """
    angulos = [-20, -10, 0, 10, 20]
    variacoes = []

    print("\n--- INICIANDO DATA AUGMENTATION ---")

    for angulo in angulos:
        # 1. Aplicando Rotação
        # Usamos reshape=False para manter a imagem do mesmo tamanho original
        # mode='nearest' preenche os cantos vazios gerados pela rotação com o pixel mais próximo
        volume_rotacionado = ndimage.rotate(volume_3d, angle=angulo, axes=(0, 1), reshape=False, mode='nearest')

        # 2. Var. 1: Versão Original (Apenas Rotacionada)
        variacoes.append(volume_rotacionado)

        # 3. Var. 2: Versão Espelhada
        # O espelhamento ocorre no plano horizontal (inversão do eixo X, que é o índice 0 na matriz)
        volume_espelhado = np.flip(volume_rotacionado, axis=0)
        variacoes.append(volume_espelhado)

        print(f"Gerado: Rotação {angulo}º Original | Rotação {angulo}º Espelhada")

    print(f"Total de volumes gerados para este paciente: {len(variacoes)}")
    return variacoes


# =====================================================================
# BLOCO DE TESTE
# =====================================================================
if __name__ == "__main__":
    # Teste da Leitura do CSV (Aponte para o CSV oficial que veio no Dataset)
    caminho_planilha = "dataset/BreastDCEDL_metadata_min_crop.csv"  # Mude para o caminho correto
    analisar_balanceamento(caminho_planilha)

    # Teste do Aumento de Dados (Usando matriz vazia apenas para ver a função rodar)
    # Na integração real, passaremos o volume lido pela nossa função 'carregar_volume_nifti'
    print("\nSimulando volume de entrada (10x10x10)...")
    volume_falso = np.zeros((10, 10, 10))
    lista_aumentada = aumento_de_dados_regrado(volume_falso)