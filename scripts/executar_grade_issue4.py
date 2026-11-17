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
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--pos-weight-resnet", type=float)
    parser.add_argument("--batch-resnet", type=int)
    parser.add_argument("--lr-cnn", type=float, default=1e-3)
    parser.add_argument("--lr-transferencia", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float)
    parser.add_argument("--fine-tuning-resnet", choices=("congelado", "layer4", "layer4_ultimo_bloco"), default="congelado")
    parser.add_argument("--normalizacao-entrada", choices=("percentil", "zscore_treino", "imagenet"), default="percentil")
    parser.add_argument("--criterio-limiar", choices=("f1", "acuracia_balanceada"), default="acuracia_balanceada")
    parser.add_argument("--criterio-checkpoint", choices=("loss", "auc_roc", "acuracia_balanceada"), default="auc_roc")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    parser.add_argument("--checkpoints", type=Path, default=RAIZ / "checkpoints")
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
        if args.tag:
            nome += f"__{args.tag}"
        validacao = args.saida / f"{nome}__validation.json"
        if validacao.is_file() and not args.refazer:
            print(f"[{numero:02d}/12] Ja concluido: {nome}", flush=True)
            concluidas += 1
            continue
        comando = [
            sys.executable,
            str(RAIZ / "scripts" / "executar_experimento.py"),
            "--loader-factory", "modulos.fabrica_carregadores:criar_carregadores",
            "--arquitetura", arquitetura,
            "--canais", "1" if fases == "post_early" else "3",
            "--fases", fases,
            "--corte", corte,
            "--epocas", str(args.epocas),
            "--batch", str(args.batch_resnet if arquitetura == "resnet50" and args.batch_resnet else args.batch),
            "--paciencia", str(args.paciencia),
            "--seed", str(args.seed),
            "--device", args.device,
            "--lr", str(args.lr_cnn if arquitetura == "cnn" else args.lr_transferencia),
            "--fine-tuning", (
                "completo" if arquitetura == "cnn" else
                args.fine_tuning_resnet if arquitetura == "resnet50" else "congelado"
            ),
            "--normalizacao-entrada", args.normalizacao_entrada,
            "--criterio-limiar", args.criterio_limiar,
            "--criterio-checkpoint", args.criterio_checkpoint,
            "--saida", str(args.saida),
            "--checkpoints", str(args.checkpoints),
        ]
        peso = args.pos_weight_resnet if arquitetura == "resnet50" and args.pos_weight_resnet is not None else args.pos_weight
        if peso is not None:
            comando.extend(("--pos-weight", str(peso)))
        if arquitetura == "resnet50" and args.lr_backbone is not None:
            comando.extend(("--lr-backbone", str(args.lr_backbone)))
        if args.tag:
            comando.extend(("--tag", args.tag))
        print(f"[{numero:02d}/12] Iniciando {nome}", flush=True)
        subprocess.run(comando, cwd=RAIZ, env=ambiente, check=True)
        concluidas += 1
    print(f"Grade concluida: {concluidas}/12 configuracoes", flush=True)
    subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "consolidar_resultados.py"),
         "--resultados", str(args.saida),
         "--saida", str(args.saida / "comparacao_validacao.csv"),
         "--criterio-checkpoint", args.criterio_checkpoint,
         "--diretorio-checkpoints", str(args.checkpoints)],
        cwd=RAIZ,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
