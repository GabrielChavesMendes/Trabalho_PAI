"""Adaptador da Issue #1 sobre o DatasetDCE da Issue #2."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from modulos.modulo_augmentacao import ANGULOS_PADRAO, listar_variacoes, transformar_par, validar_particao_aumentada


class DatasetDCEAumentado(Dataset):
    """Expande somente treino em 10 variantes por paciente, sob demanda.

    O DatasetDCE original é injetado para evitar cópia de lógica da Issue #2.
    Cada item mantém a máscara separada e o mesmo rótulo/paciente/corte.
    """
    def __init__(self, dataset_base: Dataset, particao: str, angulos=ANGULOS_PADRAO):
        validar_particao_aumentada(particao)
        self.dataset_base = dataset_base
        self.variacoes = listar_variacoes(angulos)
        self.particao = particao

    def __len__(self):
        return len(self.dataset_base) * len(self.variacoes)

    def __getitem__(self, indice):
        base_indice, variacao_indice = divmod(indice, len(self.variacoes))
        amostra = dict(self.dataset_base[base_indice])
        variacao = self.variacoes[variacao_indice]
        imagem, mascara = transformar_par(
            amostra["imagem"].detach().cpu().numpy(),
            amostra["mascara"].detach().cpu().numpy(),
            variacao.angulo,
            variacao.espelhada,
        )
        amostra["imagem"] = torch.from_numpy(imagem)
        amostra["mascara"] = torch.from_numpy(mascara)
        amostra["augmentacao"] = variacao.nome
        return amostra
