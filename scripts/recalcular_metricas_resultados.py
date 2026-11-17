"""Acrescenta AUC-ROC/BAcc a JSONs existentes sem reavaliar os modelos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.modulo_treinamento import RegistroPredicao, calcular_metricas  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalcular metricas das predicoes salvas")
    parser.add_argument("--resultados", type=Path, default=RAIZ / "resultados")
    args = parser.parse_args()
    caminhos = sorted(args.resultados.glob("*__validation.json"))
    caminhos += sorted(args.resultados.glob("*__test.json"))
    for caminho in caminhos:
        conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        anteriores = conteudo["metricas"]
        registros = [
            RegistroPredicao(
                paciente=item["paciente"],
                rotulo=int(item["rotulo"]),
                probabilidade=float(item["probabilidade"]),
            )
            for item in conteudo["predicoes_por_paciente"]
        ]
        metricas = calcular_metricas(registros, float(anteriores["limiar"]))
        if "loss" in anteriores:
            metricas["loss"] = anteriores["loss"]
        conteudo["metricas"] = metricas
        caminho.write_text(
            json.dumps(conteudo, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Atualizado: {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
