"""Nucleo reutilizavel de Grad-CAM para classificadores binarios DCE-MRI.

O modulo explica o logit positivo (pCR=1). Nao realiza segmentacao e nao
substitui a inferencia/pre-processamento definidos pelo checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ResultadoGradCAM:
    """Saida de uma explicacao para uma unica amostra."""

    mapa: torch.Tensor
    logit: float
    probabilidade_pcr: float


def resolver_camada_alvo(modelo: nn.Module, arquitetura: str) -> nn.Module:
    """Resolve a ultima camada convolucional adequada nas tres arquiteturas."""
    arquitetura = arquitetura.lower()

    if arquitetura == "resnet50":
        camada = getattr(modelo, "layer4", None)
        if camada is not None and len(camada):
            return camada[-1]

    elif arquitetura == "mobilenetv2":
        caracteristicas = getattr(modelo, "features", None)
        if caracteristicas is not None and len(caracteristicas):
            return caracteristicas[-1]

    elif arquitetura == "cnn":
        caracteristicas = getattr(modelo, "caracteristicas", None)
        if caracteristicas is not None:
            camadas = list(caracteristicas.children())
            for indice in range(len(camadas) - 1, -1, -1):
                if isinstance(camadas[indice], nn.Conv2d):
                    # Preferir a ativacao imediatamente posterior, se existir.
                    if indice + 1 < len(camadas) and isinstance(
                        camadas[indice + 1], (nn.ReLU, nn.LeakyReLU, nn.GELU)
                    ):
                        return camadas[indice + 1]
                    return camadas[indice]

    raise ValueError(
        f"Nao foi possivel resolver a camada Grad-CAM para arquitetura '{arquitetura}'."
    )


def gerar_gradcam(
    modelo: nn.Module,
    imagem: torch.Tensor,
    arquitetura: Optional[str] = None,
    camada_alvo: Optional[nn.Module] = None,
    dispositivo: Optional[str | torch.device] = None,
) -> ResultadoGradCAM:
    """Gera mapa Grad-CAM para o logit pCR=1 de uma imagem [1,C,H,W].

    Os hooks sao registrados apenas durante esta chamada e sempre removidos,
    inclusive quando ocorre uma excecao. Funciona em CPU e CUDA.
    """
    if not isinstance(imagem, torch.Tensor) or imagem.ndim != 4:
        raise ValueError("imagem deve ser um tensor [1,C,H,W].")
    if imagem.shape[0] != 1:
        raise ValueError("Grad-CAM aceita exatamente uma imagem por chamada.")
    if not torch.isfinite(imagem).all():
        raise ValueError("imagem contem NaN ou infinito.")

    if camada_alvo is None:
        if not arquitetura:
            raise ValueError("Informe arquitetura ou camada_alvo.")
        camada_alvo = resolver_camada_alvo(modelo, arquitetura)

    dispositivo_modelo = (
        torch.device(dispositivo)
        if dispositivo is not None
        else next(modelo.parameters()).device
    )
    imagem = imagem.to(dispositivo_modelo)
    modelo.to(dispositivo_modelo)

    ativacoes: list[torch.Tensor] = []
    gradientes: list[torch.Tensor] = []

    def capturar_ativacao(_modulo, _entradas, saida):
        if not isinstance(saida, torch.Tensor) or saida.ndim != 4:
            raise RuntimeError("A camada Grad-CAM precisa produzir [N,C,H,W].")
        ativacoes.append(saida)
        if saida.requires_grad:
            saida.register_hook(lambda grad: gradientes.append(grad))

    estava_treinando = modelo.training
    handle = camada_alvo.register_forward_hook(capturar_ativacao)
    try:
        modelo.eval()
        modelo.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = modelo(imagem)
            if not isinstance(logits, torch.Tensor) or logits.numel() != 1:
                raise ValueError("O modelo deve retornar um unico logit binario.")
            logit = logits.reshape(-1)[0]
            logit.backward()

        if not ativacoes or not gradientes:
            raise RuntimeError("Nao foi possivel capturar ativacoes e gradientes.")

        ativacao = ativacoes[-1]
        gradiente = gradientes[-1]
        pesos = gradiente.mean(dim=(2, 3), keepdim=True)
        mapa = F.relu((pesos * ativacao).sum(dim=1, keepdim=True))
        mapa = F.interpolate(
            mapa, size=imagem.shape[-2:], mode="bilinear", align_corners=False
        )
        minimo = mapa.amin(dim=(2, 3), keepdim=True)
        maximo = mapa.amax(dim=(2, 3), keepdim=True)
        mapa = (mapa - minimo) / (maximo - minimo).clamp_min(1e-8)

        probabilidade = torch.sigmoid(logit.detach())
        return ResultadoGradCAM(
            mapa=mapa[0, 0].detach().cpu(),
            logit=float(logit.detach().cpu()),
            probabilidade_pcr=float(probabilidade.cpu()),
        )
    finally:
        handle.remove()
        modelo.zero_grad(set_to_none=True)
        modelo.train(estava_treinando)


def sobrepor_gradcam(
    imagem_2d: torch.Tensor,
    mapa: torch.Tensor,
    alpha: float = 0.4,
) -> torch.Tensor:
    """Combina imagem normalizada e mapa em RGB [H,W,3], sem escolher paleta."""
    if imagem_2d.ndim != 2 or mapa.ndim != 2:
        raise ValueError("imagem_2d e mapa devem ser matrizes [H,W].")
    if imagem_2d.shape != mapa.shape:
        raise ValueError("imagem e mapa Grad-CAM devem ter o mesmo tamanho.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha deve estar entre 0 e 1.")
    base = imagem_2d.float().clamp(0, 1)
    base_rgb = base.unsqueeze(-1).repeat(1, 1, 3)
    calor_rgb = torch.stack((mapa, torch.zeros_like(mapa), 1.0 - mapa), dim=-1)
    return ((1.0 - alpha) * base_rgb + alpha * calor_rgb).clamp(0, 1)
