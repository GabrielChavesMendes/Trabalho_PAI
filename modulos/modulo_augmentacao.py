"""Aumento geométrico sincronizado para imagens DCE-MRI e máscaras."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

ANGULOS_PADRAO = (-20, -10, 0, 10, 20)


@dataclass(frozen=True)
class Variacao:
    angulo: int
    espelhada: bool

    @property
    def nome(self) -> str:
        tipo = "espelhada" if self.espelhada else "original"
        return f"{self.angulo:+d}deg_{tipo}"


def listar_variacoes(angulos: Iterable[int] = ANGULOS_PADRAO) -> tuple[Variacao, ...]:
    """Combina cada ângulo com as versões original e espelhada."""
    valores = tuple(int(angulo) for angulo in angulos)
    if not valores or len(set(valores)) != len(valores):
        raise ValueError("Informe ângulos não vazios e sem repetição.")
    return tuple(
        Variacao(angulo, espelhada)
        for angulo in valores
        for espelhada in (False, True)
    )


def transformar_par(
    imagem: np.ndarray,
    mascara: np.ndarray | None,
    angulo: int,
    espelhada: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Aplica a mesma rotação/espelhamento a todos os canais e à máscara."""
    imagem = np.asarray(imagem, dtype=np.float32)
    if imagem.ndim != 3 or not np.isfinite(imagem).all():
        raise ValueError("Imagem deve ser [C,H,W] e conter valores finitos.")

    mask = None if mascara is None else np.asarray(mascara)
    if mask is not None and (mask.ndim != 3 or mask.shape != (1, *imagem.shape[-2:])):
        raise ValueError("Máscara deve ter formato [1,H,W] e o mesmo plano XY da imagem.")

    imagem = ndimage.rotate(
        imagem, angulo, axes=(-2, -1), reshape=False,
        order=1, mode="nearest", prefilter=False,
    ).astype(np.float32, copy=False)

    if mask is not None:
        mask = ndimage.rotate(
            mask, angulo, axes=(-2, -1), reshape=False,
            order=0, mode="nearest", prefilter=False,
        ).astype(np.float32, copy=False)

    if espelhada:
        imagem = np.flip(imagem, axis=-1)
        if mask is not None:
            mask = np.flip(mask, axis=-1)

    imagem = np.ascontiguousarray(imagem)
    mascara_saida = None if mask is None else np.ascontiguousarray(mask)
    return imagem, mascara_saida


def gerar_variacoes(
    imagem: np.ndarray,
    mascara: np.ndarray | None = None,
    angulos: Iterable[int] = ANGULOS_PADRAO,
) -> list[tuple[Variacao, np.ndarray, np.ndarray | None]]:
    """Gera as dez combinações previstas no trabalho."""
    return [
        (variacao, *transformar_par(imagem, mascara, variacao.angulo, variacao.espelhada))
        for variacao in listar_variacoes(angulos)
    ]
