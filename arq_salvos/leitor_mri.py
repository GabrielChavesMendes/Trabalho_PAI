import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def carregar_volume_nifti(caminho_arquivo):
    """
    Carrega um arquivo NIfTI (.nii ou .nii.gz) e extrai sua matriz de dados tridimensional.

    O formato NIfTI armazena tanto os metadados (cabeçalho com informações do paciente/exame)
    quanto os voxels (os pixels 3D da imagem). Esta função isola os dados da imagem
    para que possamos manipulá-los no formato de matrizes do NumPy.

    Parâmetros:
        caminho_arquivo (str): O caminho absoluto ou relativo para o arquivo .nii.

    Retorno:
        volume_3d (numpy.ndarray): Matriz multidimensional contendo os dados da imagem.
                                   Geralmente possui a estrutura (X, Y, Z).
    """

    # nib.load(): Função da biblioteca nibabel responsável por ler o arquivo no disco.
    # Ela não carrega toda a matriz de dados para a memória imediatamente,
    # apenas lê o cabeçalho e mapeia o arquivo, otimizando o uso de memória.
    imagem_nifti = nib.load(caminho_arquivo)

    # get_fdata(): Extrai os dados da imagem do objeto mapeado e converte para float (ponto flutuante).
    # Isso é crucial porque redes neurais e operações matemáticas (como médias e normalizações)
    # exigem que os dados estejam em formato float para manter a precisão.
    volume_3d = imagem_nifti.get_fdata()

    return volume_3d


def extrair_corte_z(volume_3d, indice_z):
    """
    Extrai um plano transversal bidimensional (plano xy) a partir do volume 3D da ressonância,
    utilizando a profundidade z especificada.

    Parâmetros:
        volume_3d (numpy.ndarray): A matriz 3D da ressonância carregada anteriormente.
        indice_z (int): O índice (profundidade) do corte que desejamos visualizar.

    Retorno:
        corte_xy (numpy.ndarray): Matriz 2D correspondente à fatia da imagem no eixo z escolhido.
    """

    # Validação de segurança: verifica se o índice escolhido existe dentro dos limites do volume.
    # O atributo .shape retorna uma tupla com as dimensões (X, Y, Z).
    # Portanto, .shape[2] representa a quantidade total de fatias no eixo Z.
    max_z = volume_3d.shape[2] - 1
    if indice_z < 0 or indice_z > max_z:
        raise ValueError(f"Índice Z fora dos limites. Escolha um valor entre 0 e {max_z}.")

    # Extração via "Slicing" do NumPy:
    # O formato [:, :, indice_z] significa:
    # 1º ':': "Pegue todos os pixels no eixo X (largura)".
    # 2º ':': "Pegue todos os pixels no eixo Y (altura)".
    # 3º 'indice_z': "Mas faça isso apenas na profundidade Z específica".
    corte_xy = volume_3d[:, :, indice_z]

    return corte_xy


# =====================================================================
# BLOCO DE TESTE (Para testarmos a lógica antes de criar a Interface)
# =====================================================================
if __name__ == "__main__":
    # AQUI VOCÊ PODE MUDAR PARA O CAMINHO DE UMA IMAGEM DO SEU COMPUTADOR
    # Como as imagens do dataset BreastDCEDL estarão no seu PC local, aponte o caminho correto.
    caminho_exemplo_imagem = "dataset\ISPY1_1142_spy1_vis1_dce_aqc_0.nii.gz"

    try:
        print("Iniciando carregamento do volume...")
        # 1. Carregamos o volume 3D
        volume = carregar_volume_nifti(caminho_exemplo_imagem)
        print(f"Volume carregado com sucesso! Dimensões originais: {volume.shape}")

        # 2. Descobrimos o corte central automaticamente
        # A divisão inteira (//) por 2 encontra a fatia exata no meio do volume.
        corte_central_z = volume.shape[2] // 2
        print(f"Extraindo o corte Z central (Índice: {corte_central_z})...")

        # 3. Extraímos o plano bidimensional (xy)
        fatia_imagem = extrair_corte_z(volume, corte_central_z)

        # 4. Usamos o Matplotlib para desenhar a imagem na tela e validar visualmente.
        # cmap="gray" garante que a ressonância seja mostrada em tons de cinza, o padrão médico.
        plt.imshow(fatia_imagem, cmap="gray")
        plt.title(f"Plano XY - Corte Z: {corte_central_z}")
        plt.axis('off')  # Esconde os eixos numéricos para focar apenas na imagem
        plt.show()

    except FileNotFoundError:
        print(
            "Arquivo de teste não encontrado. Altere a variável 'caminho_exemplo_imagem' para um arquivo .nii válido no seu PC.")