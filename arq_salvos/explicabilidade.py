import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2  # pip install opencv-python


class GradCAM:
    def __init__(self, modelo, camada_alvo):
        """
        Inicializa os 'ganchos' (hooks) para espionar a última camada convolucional.
        """
        self.modelo = modelo
        self.camada_alvo = camada_alvo
        self.gradientes = None
        self.ativacoes = None

        # Registra os hooks
        self.camada_alvo.register_forward_hook(self.salvar_ativacoes)
        self.camada_alvo.register_full_backward_hook(self.salvar_gradientes)

    def salvar_ativacoes(self, module, input, output):
        self.ativacoes = output.detach()

    def salvar_gradientes(self, module, grad_input, grad_output):
        self.gradientes = grad_output[0].detach()

    def gerar_mapa_calor(self, tensor_imagem):
        """Passa a imagem pela rede e calcula as áreas de maior peso (calor)."""
        self.modelo.eval()

        # 1. Forward Pass (Previsão)
        previsao = self.modelo(tensor_imagem)

        # Zera gradientes anteriores e força o backward a partir da previsão da classe 1 (pCR)
        self.modelo.zero_grad()
        previsao.backward(retain_graph=True)

        # 2. Calcula os pesos pela média global dos gradientes
        pesos = torch.mean(self.gradientes, dim=[2, 3], keepdim=True)

        # 3. Multiplica os pesos pelas ativações (Mapas de Características)
        cam = torch.sum(pesos * self.ativacoes, dim=1, keepdim=True)

        # 4. Aplica ReLU (queremos apenas as influências positivas) e normaliza
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)

        return cam.squeeze().cpu().numpy(), previsao.item()


def visualizar_grad_cam(imagem_original_2d, mapa_calor_cam):
    """
    Sobrepõe o mapa de calor vermelho/amarelo em cima da ressonância em tons de cinza.
    """
    # Redimensiona o mapa de calor da resolução da CNN (ex: 4x4 ou 16x16)
    # para o tamanho original da ressonância (256x256)
    mapa_redimensionado = cv2.resize(mapa_calor_cam, (imagem_original_2d.shape[1], imagem_original_2d.shape[0]))

    # Aplica a paleta de cores (Vermelho = alta importância, Azul = baixa importância)
    mapa_colorido = cv2.applyColorMap(np.uint8(255 * mapa_redimensionado), cv2.COLORMAP_JET)

    # Prepara a imagem original para receber a cor
    img_cinza = np.uint8(255 * (imagem_original_2d / np.max(imagem_original_2d)))
    img_cinza_rgb = cv2.cvtColor(img_cinza, cv2.COLOR_GRAY2RGB)

    # Mescla as duas (0.6 original, 0.4 calor)
    imagem_final = cv2.addWeighted(img_cinza_rgb, 0.6, mapa_colorido, 0.4, 0)

    # Plota o resultado final
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_cinza, cmap='gray')
    plt.title("Ressonância Magnética Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagem_final)
    plt.title("Grad-CAM: Onde a Rede Olhou?")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# =====================================================================
# EXEMPLO DE USO COM A REDE QUE CRIAMOS (CNNCancerMama)
# =====================================================================
if __name__ == "__main__":
    from cnn_customizada import CNNCancerMama

    print("Inicializando teste do Grad-CAM...")
    # 1. Carrega a nossa rede customizada
    rede = CNNCancerMama(canais_entrada=1)

    # 2. Inicializa o Grad-CAM apontando para a ÚLTIMA camada convolucional da rede (conv4)
    cam_explicador = GradCAM(modelo=rede, camada_alvo=rede.conv4)

    # 3. Simula uma fatia 2D real carregada pelo seu dataset
    fatia_simulada_256 = np.random.rand(256, 256)
    tensor_entrada = torch.FloatTensor(fatia_simulada_256).unsqueeze(0).unsqueeze(0)

    # 4. Gera o mapa de calor e a probabilidade
    mapa_gerado, score = cam_explicador.gerar_mapa_calor(tensor_entrada)
    print(f"Logit gerado pela rede para a classe pCR: {score:.4f}")

    # 5. Exibe a janela gráfica
    visualizar_grad_cam(fatia_simulada_256, mapa_gerado)