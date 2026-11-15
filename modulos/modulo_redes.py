"""Exemplo de arquiteturas para classificacao binaria de DCE-MRI.

Interface esperada: imagens como tensores [N, C, H, W], com C=1 ou 3.
As redes devolvem logits [N, 1]; aplique torch.sigmoid para probabilidades.
"""

import argparse

import torch
from torch import nn


class CNNSimples(nn.Module):
    """Quatro convolucoes e duas camadas lineares (6 com parametros)."""

    def __init__(self, canais_entrada=3):
        super().__init__()
        if canais_entrada not in (1, 3):
            raise ValueError("Use 1 ou 3 canais de entrada.")
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(canais_entrada, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classificador = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, imagens):
        return self.classificador(self.caracteristicas(imagens))


def _adaptar_primeira_convolucao(conv_original, canais_entrada):
    """Para 1 canal, soma os filtros RGB (equivalente a repetir o canal 3 vezes)."""
    if canais_entrada == 3:
        return conv_original
    conv_nova = nn.Conv2d(
        1, conv_original.out_channels, kernel_size=conv_original.kernel_size,
        stride=conv_original.stride, padding=conv_original.padding,
        dilation=conv_original.dilation, groups=conv_original.groups,
        bias=conv_original.bias is not None,
    )
    with torch.no_grad():
        conv_nova.weight.copy_(conv_original.weight.sum(dim=1, keepdim=True))
        if conv_original.bias is not None:
            conv_nova.bias.copy_(conv_original.bias)
    return conv_nova


def criar_resnet50(canais_entrada=3, pre_treinado=True):
    """Congela extrator de caracteristicas e substitui camada de saida."""
    from torchvision.models import ResNet50_Weights, resnet50

    pesos = ResNet50_Weights.DEFAULT if pre_treinado else None
    modelo = resnet50(weights=pesos)
    modelo.conv1 = _adaptar_primeira_convolucao(modelo.conv1, canais_entrada)
    for parametro in modelo.parameters():
        parametro.requires_grad = False
    modelo.fc = nn.Linear(modelo.fc.in_features, 1)
    return modelo


def criar_mobilenetv2(canais_entrada=3, pre_treinado=True):
    """Congela extrator de caracteristicas e substitui classificador."""
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    pesos = MobileNet_V2_Weights.DEFAULT if pre_treinado else None
    modelo = mobilenet_v2(weights=pesos)
    modelo.features[0][0] = _adaptar_primeira_convolucao(
        modelo.features[0][0], canais_entrada
    )
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False
    modelo.classifier[1] = nn.Linear(modelo.last_channel, 1)
    return modelo


def criar_modelo(nome, canais_entrada=3, pre_treinado=True):
    """Ponto unico de entrada para o futuro modulo_treinamento.py."""
    if canais_entrada not in (1, 3):
        raise ValueError("canais_entrada deve ser 1 ou 3.")
    nome = nome.lower()
    if nome == "cnn":
        return CNNSimples(canais_entrada)
    if nome == "resnet50":
        return criar_resnet50(canais_entrada, pre_treinado)
    if nome == "mobilenetv2":
        return criar_mobilenetv2(canais_entrada, pre_treinado)
    raise ValueError("Modelo desconhecido. Use cnn, resnet50 ou mobilenetv2.")


@torch.no_grad()
def prever_probabilidades(modelo, imagens):
    """Recebe tensor [N, C, H, W] e devolve probabilidades [N, 1]."""
    modelo.eval()
    return torch.sigmoid(modelo(imagens))


def testar_arquiteturas():
    """Smoke test com entrada artificial: NAO mede acuracia real."""
    for canais in (1, 3):
        imagens = torch.rand(1, canais, 64, 64)
        for nome in ("cnn", "resnet50", "mobilenetv2"):
            modelo = criar_modelo(nome, canais, pre_treinado=False)
            probs = prever_probabilidades(modelo, imagens)
            assert probs.shape == (1, 1), (nome, probs.shape)
            assert torch.isfinite(probs).all()
            assert ((probs >= 0) & (probs <= 1)).all()
            print(f"OK: {nome}, canais={canais}, saida={probs.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemplo didatico de redes DCE-MRI")
    parser.add_argument("--teste", action="store_true", help="Testar formas sem baixar pesos")
    argumentos = parser.parse_args()
    if argumentos.teste:
        testar_arquiteturas()
    else:
        parser.print_help()
