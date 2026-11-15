"""Aumento de dados aplicado sobre o DatasetDCE da Issue #2."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from modulos.modulo_augmentacao import (
    ANGULOS_PADRAO,
    listar_variacoes,
    transformar_par,
)


class DatasetDCEAumentado(Dataset):
    """Cria 10 variantes por paciente, sem carregar o dataset inteiro na memória.

    O dataset-base deve ser um DatasetDCE construído após selecionar treino ou
    validação. A partição é conferida nos próprios objetos Paciente para impedir
    que amostras do conjunto de teste sejam aumentadas por engano.
    """

    PARTICOES_PERMITIDAS = {"treino": 0, "validacao": 2}

    def __init__(self, dataset_base: Dataset, particao: str, angulos=ANGULOS_PADRAO):
        if particao not in self.PARTICOES_PERMITIDAS:
            raise ValueError("Aumento permitido somente em treino ou validação; teste nunca é aumentado.")

        pacientes = getattr(dataset_base, "pacientes", None)
        if pacientes is None:
            raise TypeError("Use um DatasetDCE com a lista de pacientes disponível para validar a partição.")
        codigos = {paciente.particao_codigo for paciente in pacientes}
        codigo_esperado = self.PARTICOES_PERMITIDAS[particao]
        if codigos and codigos != {codigo_esperado}:
            raise ValueError(f"O dataset informado não pertence exclusivamente à partição {particao}.")

        self.dataset_base = dataset_base
        self.variacoes = listar_variacoes(angulos)

    def __len__(self):
        return len(self.dataset_base) * len(self.variacoes)

    def __getitem__(self, indice):
        indice_base, indice_variacao = divmod(indice, len(self.variacoes))
        amostra = dict(self.dataset_base[indice_base])
        variacao = self.variacoes[indice_variacao]

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
