"""Aumento geométrico sincronizado para DCE-MRI e máscara (Issue #1).

Volumes do MinCrop são [Z,H,W]; as operações abaixo atuam apenas no plano XY.
A máscara usa interpolação nearest-neighbor e nunca é concatenada à entrada da rede.
"""
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
        return f"{self.angulo:+d}deg_{'espelhada' if self.espelhada else 'original'}"


def listar_variacoes(angulos: Iterable[int] = ANGULOS_PADRAO) -> tuple[Variacao, ...]:
    """Retorna exatamente 2N combinações: ângulos × original/espelhada."""
    valores = tuple(int(a) for a in angulos)
    if not valores or len(set(valores)) != len(valores):
        raise ValueError("Informe ângulos não vazios e sem repetição.")
    return tuple(Variacao(a, espelho) for a in valores for espelho in (False, True))


def transformar_par(
    imagem: np.ndarray,
    mascara: np.ndarray | None,
    angulo: int,
    espelhada: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Transforma [C,H,W] e [1,H,W] com a mesma geometria e sem deformar rótulos."""
    imagem = np.asarray(imagem, dtype=np.float32)
    if imagem.ndim != 3 or not np.isfinite(imagem).all():
        raise ValueError("Imagem deve ser [C,H,W] e conter apenas valores finitos.")
    mask = None if mascara is None else np.asarray(mascara)
    if mask is not None and (mask.ndim != 3 or mask.shape[-2:] != imagem.shape[-2:]):
        raise ValueError("Máscara deve ser [1,H,W] e ter o mesmo plano XY da imagem.")
    if mask is not None and mask.shape[0] != 1:
        raise ValueError("A máscara deve possuir exatamente um canal.")

    img_out = ndimage.rotate(
        imagem, int(angulo), axes=(-2, -1), reshape=False,
        order=1, mode="nearest", prefilter=False,
    ).astype(np.float32, copy=False)
    mask_out = None
    if mask is not None:
        mask_out = ndimage.rotate(
            mask, int(angulo), axes=(-2, -1), reshape=False,
            order=0, mode="nearest", prefilter=False,
        ).astype(np.float32, copy=False)
    if espelhada:
        img_out = np.flip(img_out, axis=-1).copy()
        if mask_out is not None:
            mask_out = np.flip(mask_out, axis=-1).copy()
    return np.ascontiguousarray(img_out), None if mask_out is None else np.ascontiguousarray(mask_out)


def gerar_variacoes(
    imagem: np.ndarray,
    mascara: np.ndarray | None = None,
    angulos: Iterable[int] = ANGULOS_PADRAO,
) -> list[tuple[Variacao, np.ndarray, np.ndarray | None]]:
    """Gera 10 exemplos por padrão, incluindo original e espelhado sem rotação."""
    return [
        (v, *transformar_par(imagem, mascara, v.angulo, v.espelhada))
        for v in listar_variacoes(angulos)
    ]


def validar_particao_aumentada(particao: str) -> None:
    """Impede explicitamente aumentar validação/teste para avaliação."""
    if particao != "treino":
        raise ValueError(
            "Aumento de dados só é permitido no treino. "
            "Validação e teste devem permanecer sem duplicatas/transformações."
        )
