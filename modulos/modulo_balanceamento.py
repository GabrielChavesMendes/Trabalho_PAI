"""Distribuição pCR e peso de classe calculado exclusivamente no treino."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch


def contar_classes(pacientes: Iterable) -> dict[int, int]:
    contagem = Counter(int(p.pcr) for p in pacientes)
    if not set(contagem).issubset({0, 1}):
        raise ValueError("Rótulos pCR devem ser binários (0/1).")
    return {0: contagem.get(0, 0), 1: contagem.get(1, 0)}


def calcular_pos_weight_treino(pacientes_treino) -> torch.Tensor:
    """Retorna negativos/positivos do treino; não aceita partições misturadas."""
    pacientes = list(pacientes_treino)
    if not pacientes:
        raise ValueError("A partição de treino está vazia.")
    codigos = {int(p.particao_codigo) for p in pacientes}
    if codigos != {0}:
        raise ValueError("pos_weight deve ser calculado somente com pacientes test=0 (treino).")
    classes = contar_classes(pacientes)
    if classes[1] == 0 or classes[0] == 0:
        raise ValueError("Treino precisa conter exemplos das duas classes pCR.")
    return torch.tensor([classes[0] / classes[1]], dtype=torch.float32)


def resumo_particoes(tabela) -> list[dict]:
    """Resumo por partição oficial: 0 treino, 1 teste, 2 validação."""
    saida = []
    for nome, codigo in (("Treino", 0), ("Teste", 1), ("Validação", 2)):
        fatia = tabela.loc[tabela["test"].astype(int) == codigo]
        saida.append({
            "particao": nome, "total": int(len(fatia)),
            "pCR_0": int((fatia["pCR"] == 0).sum()),
            "pCR_1": int((fatia["pCR"] == 1).sum()),
        })
    return saida
