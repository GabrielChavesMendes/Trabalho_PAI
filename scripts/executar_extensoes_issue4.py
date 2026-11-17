"""Experimentos adicionais contra sobreajuste, definidos apos a grade principal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "ajustes_issue4_extensoes"
CHECKPOINTS = RAIZ / "checkpoints" / "ajustes_issue4_extensoes"

EXPERIMENTOS = (
    # Isola a quantidade de camadas liberadas.
    ("ultimo-bloco", "tres", "maior_tumor", 3, "layer4_ultimo_bloco", 1e-4, 1e-5, 1e-4),
    # Mantem a cabeca ativa, mas reduz/regulariza fortemente o backbone.
    ("layer4-regularizada", "tres", "maior_tumor", 3, "layer4", 1e-4, 1e-6, 1e-3),
    # Alternativa sem depender da mascara na inferencia.
    ("tres-central", "tres", "central", 3, "layer4", 1e-4, 1e-5, 1e-4),
    # Fase isolada no corte que teve a melhor AUC do baseline.
    ("post-early-central", "post_early", "central", 1, "layer4", 1e-4, 1e-5, 1e-4),
)


def main() -> int:
    ambiente = os.environ.copy()
    ambiente["PYTHONUNBUFFERED"] = "1"
    for numero, (tag, fases, corte, canais, modo, lr, lr_backbone, wd) in enumerate(
        EXPERIMENTOS, start=1
    ):
        nome = f"resnet50__{fases.replace('_', '-')}__{corte.replace('_', '-')}__seed-42__{tag}"
        if (SAIDA / f"{nome}__validation.json").is_file():
            print(f"[{numero:02d}/{len(EXPERIMENTOS):02d}] Ja concluido: {tag}", flush=True)
            continue
        comando = [
            sys.executable, str(RAIZ / "scripts" / "executar_experimento.py"),
            "--loader-factory", "modulos.fabrica_carregadores:criar_carregadores",
            "--arquitetura", "resnet50", "--canais", str(canais),
            "--fases", fases, "--corte", corte,
            "--epocas", "30", "--batch", "32", "--paciencia", "3",
            "--seed", "42", "--device", "cuda",
            "--lr", str(lr), "--lr-backbone", str(lr_backbone),
            "--weight-decay", str(wd), "--pos-weight", "1.0",
            "--fine-tuning", modo, "--normalizacao-entrada", "percentil",
            "--criterio-limiar", "acuracia_balanceada",
            "--criterio-checkpoint", "auc_roc", "--tag", tag,
            "--saida", str(SAIDA), "--checkpoints", str(CHECKPOINTS),
        ]
        print(f"[{numero:02d}/{len(EXPERIMENTOS):02d}] Iniciando: {tag}", flush=True)
        subprocess.run(comando, cwd=RAIZ, env=ambiente, check=True)
    subprocess.run(
        [
            sys.executable, str(RAIZ / "scripts" / "consolidar_resultados.py"),
            "--resultados", str(SAIDA),
            "--saida", str(SAIDA / "comparacao_extensoes.csv"),
            "--criterio-checkpoint", "auc_roc",
            "--diretorio-checkpoints", "checkpoints/ajustes_issue4_extensoes",
        ], cwd=RAIZ, check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
