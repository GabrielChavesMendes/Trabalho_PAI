"""Grade controlada de ajustes, usando somente treino e validacao."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "ajustes_issue4"
CHECKPOINTS = RAIZ / "checkpoints" / "ajustes_issue4"

# Uma unica configuracao de imagem isola o efeito de cada hiperparametro.
EXPERIMENTOS = (
    ("pw-1", 1.0, "congelado", 1e-4, None, "percentil"),
    ("pw-1p73", 3 ** 0.5, "congelado", 1e-4, None, "percentil"),
    ("pw-3", 3.0, "congelado", 1e-4, None, "percentil"),
    ("ft-pw-1", 1.0, "layer4", 1e-4, 1e-5, "percentil"),
    ("ft-pw-1p73", 3 ** 0.5, "layer4", 1e-4, 1e-5, "percentil"),
    ("ft-pw-3", 3.0, "layer4", 1e-4, 1e-5, "percentil"),
    ("ft-lr-baixa", 3 ** 0.5, "layer4", 1e-5, 1e-6, "percentil"),
    ("ft-zscore", 3 ** 0.5, "layer4", 1e-4, 1e-5, "zscore_treino"),
)


def main() -> int:
    ambiente = os.environ.copy()
    ambiente["PYTHONUNBUFFERED"] = "1"
    for numero, (tag, peso, fine_tuning, lr, lr_backbone, normalizacao) in enumerate(
        EXPERIMENTOS, start=1
    ):
        nome = f"resnet50__tres__maior-tumor__seed-42__{tag}"
        validacao = SAIDA / f"{nome}__validation.json"
        if validacao.is_file():
            print(f"[{numero:02d}/{len(EXPERIMENTOS):02d}] Ja concluido: {tag}", flush=True)
            continue
        comando = [
            sys.executable,
            str(RAIZ / "scripts" / "executar_experimento.py"),
            "--loader-factory", "modulos.fabrica_carregadores:criar_carregadores",
            "--arquitetura", "resnet50",
            "--canais", "3",
            "--fases", "tres",
            "--corte", "maior_tumor",
            "--epocas", "30",
            "--batch", "32" if fine_tuning == "layer4" else "64",
            "--paciencia", "3",
            "--seed", "42",
            "--device", "cuda",
            "--lr", str(lr),
            "--pos-weight", str(peso),
            "--fine-tuning", fine_tuning,
            "--normalizacao-entrada", normalizacao,
            "--criterio-limiar", "acuracia_balanceada",
            "--criterio-checkpoint", "auc_roc",
            "--tag", tag,
            "--saida", str(SAIDA),
            "--checkpoints", str(CHECKPOINTS),
        ]
        if lr_backbone is not None:
            comando.extend(("--lr-backbone", str(lr_backbone)))
        print(f"[{numero:02d}/{len(EXPERIMENTOS):02d}] Iniciando: {tag}", flush=True)
        subprocess.run(comando, cwd=RAIZ, env=ambiente, check=True)
    subprocess.run(
        [
            sys.executable,
            str(RAIZ / "scripts" / "consolidar_resultados.py"),
            "--resultados", str(SAIDA),
            "--saida", str(SAIDA / "comparacao_ajustes.csv"),
            "--criterio-checkpoint", "auc_roc",
            "--diretorio-checkpoints", "checkpoints/ajustes_issue4",
        ],
        cwd=RAIZ,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
