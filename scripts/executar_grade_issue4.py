"""Executa as 12 configuracoes da Issue #4 sem consultar o teste reservado."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Executar grade 3 redes x 2 fases x 2 cortes")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "grade_issue4_revisada")
    parser.add_argument("--checkpoints", type=Path, default=RAIZ / "checkpoints" / "grade_issue4_revisada")
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--finetune-resnet", action="store_true")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-cnn", type=float, default=1e-3)
    parser.add_argument("--lr-transfer", type=float, default=1e-4)
    parser.add_argument("--criterio-limiar", choices=("f1", "acuracia_balanceada"), default="acuracia_balanceada")
    parser.add_argument("--criterio-modelo", choices=("f1", "acuracia_balanceada", "auc_roc"), default="auc_roc")
    parser.add_argument("--refazer", action="store_true")
    args = parser.parse_args()

    configuracoes = [
        (arquitetura, fases, corte)
        for corte in ("central", "maior_tumor")
        for fases in ("post_early", "tres")
        for arquitetura in ("cnn", "mobilenetv2", "resnet50")
    ]
    ambiente = os.environ.copy()
    ambiente["PYTHONUNBUFFERED"] = "1"
    concluidas = 0
    for numero, (arquitetura, fases, corte) in enumerate(configuracoes, start=1):
        temporal_nome = fases.replace("_", "-")
        nome = f"{arquitetura}__{temporal_nome}__{corte.replace('_', '-')}__seed-{args.seed}"
        validacao = args.saida / f"{nome}__validation.json"
        checkpoint = args.checkpoints / f"{nome}.pt"
        if validacao.is_file() and checkpoint.is_file() and not args.refazer:
            print(f"[{numero:02d}/12] Ja concluido e checkpoint confirmado: {nome}", flush=True)
            concluidas += 1
            continue
        if (validacao.exists() or checkpoint.exists()) and not args.refazer:
            raise FileNotFoundError(
                f"Execucao incompleta/inconsistente para {nome}. Confira {validacao} e {checkpoint}; "
                "use --refazer apenas se quiser substituir esta configuracao."
            )
        comando = [
            sys.executable,
            str(RAIZ / "scripts" / "executar_experimento.py"),
            "--loader-factory", "modulos.fabrica_carregadores:criar_carregadores",
            "--arquitetura", arquitetura,
            "--canais", "1" if fases == "post_early" else "3",
            "--fases", fases,
            "--corte", corte,
            "--epocas", str(args.epocas),
            "--batch", str(args.batch),
            "--paciencia", str(args.paciencia),
            "--seed", str(args.seed),
            "--device", args.device,
            "--pos-weight", str(args.pos_weight),
            "--lr", str(args.lr_cnn if arquitetura == "cnn" else args.lr_transfer),
            "--lr-backbone", str(args.lr_backbone),
            "--criterio-limiar", args.criterio_limiar,
            "--saida", str(args.saida),
            "--checkpoints", str(args.checkpoints),
        ]
        if args.finetune_resnet and arquitetura == "resnet50":
            comando.append("--finetune-resnet")
        print(f"[{numero:02d}/12] Iniciando {nome}", flush=True)
        subprocess.run(comando, cwd=RAIZ, env=ambiente, check=True)
        concluidas += 1
    print(f"Grade concluida: {concluidas}/12 configuracoes", flush=True)
    subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "consolidar_resultados.py"),
         "--resultados", str(args.saida),
         "--saida", str(args.saida / "comparacao_validacao.csv"),
         "--criterio", args.criterio_modelo],
        cwd=RAIZ,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
