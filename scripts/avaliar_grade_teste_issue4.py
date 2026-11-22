"""Avalia no teste reservado todos os checkpoints de uma grade ja concluida.

Os limiares permanecem os escolhidos na validacao e nenhum modelo e retreinado.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliar os 12 checkpoints da Issue #4 no teste")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--tag", default="final-v2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--refazer", action="store_true")
    args = parser.parse_args()

    padrao = f"*__{args.tag}.pt" if args.tag else "*.pt"
    checkpoints = sorted(args.checkpoints.glob(padrao))
    if len(checkpoints) != 12:
        raise ValueError(
            f"esperados 12 checkpoints para a grade, encontrados {len(checkpoints)} em "
            f"{args.checkpoints} com o padrao {padrao!r}"
        )

    args.saida.mkdir(parents=True, exist_ok=True)
    for numero, checkpoint in enumerate(checkpoints, start=1):
        resultado = args.saida / f"{checkpoint.stem}__test.json"
        if resultado.is_file() and not args.refazer:
            print(f"[{numero:02d}/12] Ja avaliado: {checkpoint.stem}", flush=True)
            continue
        print(f"[{numero:02d}/12] Testando: {checkpoint.stem}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(RAIZ / "scripts" / "avaliar_teste_final.py"),
                "--checkpoint", str(checkpoint),
                "--device", args.device,
                "--saida", str(args.saida),
            ],
            cwd=RAIZ,
            check=True,
        )

    linhas: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        caminho = args.saida / f"{checkpoint.stem}__test.json"
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
        metricas = conteudo["metricas"]
        partes = checkpoint.stem.split("__")
        linhas.append({
            "experimento": checkpoint.stem,
            "arquitetura": partes[0],
            "fases": partes[1],
            "corte": partes[2],
            "auc_roc": metricas["auc_roc"],
            "acuracia_balanceada": metricas["acuracia_balanceada"],
            "acuracia": metricas["acuracia"],
            "precisao": metricas["precisao"],
            "sensibilidade": metricas["sensibilidade"],
            "especificidade": metricas["especificidade"],
            "f1": metricas["f1"],
            "loss": metricas["loss"],
            "limiar_validacao": metricas["limiar"],
        })
    linhas.sort(
        key=lambda item: (
            -float(item["auc_roc"]),
            -float(item["acuracia_balanceada"]),
            -float(item["f1"]),
        )
    )
    comparacao = args.saida / "comparacao_teste.csv"
    with comparacao.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"Grade de teste concluida: 12/12; comparacao em {comparacao}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
