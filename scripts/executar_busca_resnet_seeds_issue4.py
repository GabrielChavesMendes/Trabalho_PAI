"""Avalia a estabilidade da ResNet promissora em cinco seeds, sem usar teste."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "busca_resnet_seeds_issue4"
CHECKPOINTS = RAIZ / "checkpoints" / "busca_resnet_seeds_issue4"
SEEDS = (7, 21, 42, 84, 123)


def main() -> int:
    ambiente = os.environ.copy()
    ambiente["PYTHONUNBUFFERED"] = "1"
    for numero, seed in enumerate(SEEDS, start=1):
        tag = "busca-resnet-seeds"
        nome = f"resnet50__post-early__maior-tumor__seed-{seed}__{tag}"
        if (SAIDA / f"{nome}__validation.json").is_file():
            print(f"[{numero:02d}/{len(SEEDS):02d}] Ja concluido: seed {seed}", flush=True)
            continue
        comando = [
            sys.executable, str(RAIZ / "scripts" / "executar_experimento.py"),
            "--loader-factory", "modulos.fabrica_carregadores:criar_carregadores",
            "--arquitetura", "resnet50", "--canais", "1",
            "--fases", "post_early", "--corte", "maior_tumor",
            "--epocas", "30", "--batch", "32", "--paciencia", "6",
            "--seed", str(seed), "--device", "cuda",
            "--lr", "0.0001", "--lr-backbone", "0.00001",
            "--weight-decay", "0.0001", "--pos-weight", "1",
            "--fine-tuning", "layer4", "--normalizacao-entrada", "percentil",
            "--criterio-limiar", "acuracia_balanceada",
            "--criterio-checkpoint", "auc_roc", "--tag", tag,
            "--saida", str(SAIDA), "--checkpoints", str(CHECKPOINTS),
        ]
        print(f"[{numero:02d}/{len(SEEDS):02d}] Iniciando seed {seed}", flush=True)
        subprocess.run(comando, cwd=RAIZ, env=ambiente, check=True)
    subprocess.run(
        [
            sys.executable, str(RAIZ / "scripts" / "consolidar_resultados.py"),
            "--resultados", str(SAIDA),
            "--saida", str(SAIDA / "comparacao_validacao.csv"),
            "--criterio-checkpoint", "auc_roc",
            "--diretorio-checkpoints", "checkpoints/busca_resnet_seeds_issue4",
        ], cwd=RAIZ, check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
