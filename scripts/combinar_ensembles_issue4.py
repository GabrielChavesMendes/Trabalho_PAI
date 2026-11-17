"""Combina por ranks dois preditores escolhidos por protocolos de validacao."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.modulo_treinamento import (  # noqa: E402
    RegistroPredicao,
    calcular_metricas,
    salvar_matriz_confusao,
    salvar_resultado_json,
)


def _ranks(caminho: Path) -> tuple[dict[str, float], dict[str, int]]:
    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    valores = {
        item["paciente"]: float(item["probabilidade"])
        for item in conteudo["predicoes_por_paciente"]
    }
    rotulos = {
        item["paciente"]: int(item["rotulo"])
        for item in conteudo["predicoes_por_paciente"]
    }
    ordenados = sorted(valores, key=valores.get)
    divisor = max(len(ordenados) - 1, 1)
    return {paciente: indice / divisor for indice, paciente in enumerate(ordenados)}, rotulos


def main() -> int:
    fontes = [
        RAIZ / "resultados" / "busca_2p5d_issue4"
        / "resnet50__post-early__maior-tumor__seed-84__vizinhanca-2p5d-s84__test.json",
        RAIZ / "resultados" / "ensemble_cv_issue4" / "ensemble__test.json",
    ]
    series = []
    rotulos_referencia = None
    for fonte in fontes:
        ranks, rotulos = _ranks(fonte)
        if rotulos_referencia is not None and rotulos != rotulos_referencia:
            raise ValueError("fontes possuem pacientes ou rotulos diferentes")
        rotulos_referencia = rotulos
        series.append(ranks)
    assert rotulos_referencia is not None
    predicoes = [
        RegistroPredicao(
            paciente,
            rotulos_referencia[paciente],
            sum(ranks[paciente] for ranks in series) / len(series),
        )
        for paciente in sorted(rotulos_referencia)
    ]
    # Ranks ficam em [0,1]; 0,5 e o ponto central predefinido, nao otimizado.
    metricas = calcular_metricas(predicoes, 0.5)
    saida = RAIZ / "resultados" / "ensemble_ranks_issue4"
    saida.mkdir(parents=True, exist_ok=True)
    salvar_resultado_json(saida / "ensemble-ranks-50-50__test.json", metricas, predicoes)
    salvar_matriz_confusao(
        saida / "ensemble-ranks-50-50__test__confusion-matrix.png", metricas
    )
    (saida / "protocolo.json").write_text(
        json.dumps(
            {
                "metodo": "media 50/50 dos ranks; sem pesos ajustados",
                "fontes": [str(path.relative_to(RAIZ)) for path in fontes],
                "limiar": 0.5,
                "observacao": (
                    "Resultado exploratorio: o teste ja havia sido consultado em "
                    "rodadas anteriores e nao deve ser descrito como estimativa intocada."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"AUC teste={metricas['auc_roc']:.4f}; "
        f"BAcc={metricas['acuracia_balanceada']:.4f}; F1={metricas['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
