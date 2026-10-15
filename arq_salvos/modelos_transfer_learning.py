import torch
import torch.nn as nn
from torchvision import models


def construir_resnet50(canais_entrada=1):
    """
    Constrói a ResNet50 exigida no trabalho, aproveitando os pesos do ImageNet.
    """
    # 1. Carrega o modelo com os pesos pré-treinados
    modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 2. Adaptação da Camada de Entrada (Letra D)
    if canais_entrada != 3:
        # A ResNet original espera imagens RGB (3 canais).
        # Se usarmos apenas 1 fase da ressonância (1 canal), substituímos a primeira camada.
        # Somamos os pesos dos 3 canais originais para preservar o conhecimento de extração de bordas.
        peso_original = modelo.conv1.weight.data
        nova_conv = nn.Conv2d(canais_entrada, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nova_conv.weight.data = peso_original.sum(dim=1, keepdim=True)
        modelo.conv1 = nova_conv

    # 3. Adaptação da Camada de Saída
    # Congelamos as camadas base (opcional, para acelerar o treino inicial)
    # for param in modelo.parameters():
    #     param.requires_grad = False

    num_features = modelo.fc.in_features
    # Substitui a camada final de 1000 classes do ImageNet por 1 neurônio (pCR: 0 ou 1)
    modelo.fc = nn.Linear(num_features, 1)

    return modelo


def construir_mobilenet_v2(canais_entrada=1):
    """
    A Terceira Rede escolhida: MobileNetV2.
    Leve, rápida e excelente para processamento restrito a laboratórios.
    """
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if canais_entrada != 3:
        peso_original = modelo.features[0][0].weight.data
        nova_conv = nn.Conv2d(canais_entrada, 32, kernel_size=3, stride=2, padding=1, bias=False)
        nova_conv.weight.data = peso_original.sum(dim=1, keepdim=True)
        modelo.features[0][0] = nova_conv

    num_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(num_features, 1)

    return modelo