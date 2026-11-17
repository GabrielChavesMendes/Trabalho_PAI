"""Ajuste controlado da ResNet50 usando somente treino e validacao.

Cada candidato usa pastas separadas. Nunca abre o teste reservado.
Usar resultados deste script para escolher hiperparametros antes da grade final.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Quatro alternativas simples, suficientes para documentar ajustes reais.
# Usar sempre os mesmos pacientes, seed, fase, corte e limite de epocas.
CANDIDATOS = (
    ("peso_3_congelada", 3.0, False),
    ("peso_1_congelada", 1.0, False),
    ("peso_1_73_congelada", 1.73, False),
    ("peso_3_finetune", 3.0, True),
)


def resumir(base: Path, fase: str, corte: str, seed: int) -> list[dict]:
    nome = f"resnet50__{fase.replace('_', '-')}__{corte.replace('_', '-')}__seed-{seed}"
    resultados = []
    for rotulo, peso, fine in CANDIDATOS:
        arquivo = base / rotulo / "resultados" / f"{nome}__validation.json"
        if not arquivo.is_file():
            raise FileNotFoundError(f"Validacao ausente: {arquivo}")
        metricas = json.loads(arquivo.read_text(encoding="utf-8"))["metricas"]
        resultados.append({
            "candidato": rotulo,
            "pos_weight": peso,
            "fine_tuning_layer4": fine,
            **{k: metricas.get(k) for k in (
                "auc_roc", "acuracia_balanceada", "f1",
                "sensibilidade", "especificidade", "acuracia", "loss", "limiar"
            )},
        })
    resultados.sort(key=lambda x: (
        -(x["auc_roc"] if x["auc_roc"] is not None else -1),
        -x["acuracia_balanceada"],
        -x["f1"],
        x["candidato"],
    ))
    saida = base / "comparacao_hiperparametros_validacao.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(resultados[0]))
        escritor.writeheader()
        escritor.writerows(resultados)
    print(f"Comparacao salva: {saida}")
    print("Candidato melhor pela VALIDACAO:", resultados[0]["candidato"])
    print("Nao consultar o conjunto de teste para mudar esta escolha.")
    return resultados


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparar quatro ajustes de ResNet50 sem usar teste")
    parser.add_argument("--fase", choices=("pre", "post_early", "post_late", "tres"), default="tres")
    parser.add_argument("--corte", choices=("central", "maior_tumor"), default="maior_tumor")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "ajustes_issue4")
    parser.add_argument("--somente-resumo", action="store_true")
    args = parser.parse_args()
    if not args.somente_resumo:
        for i, (rotulo, peso, fine) in enumerate(CANDIDATOS, start=1):
            pasta = args.saida / rotulo
            resultados = pasta / "resultados"
            checkpoints = pasta / "checkpoints"
            nome = f"resnet50__{args.fase.replace('_', '-')}__{args.corte.replace('_', '-')}__seed-{args.seed}"
            validacao = resultados / f"{nome}__validation.json"
            checkpoint = checkpoints / f"{nome}.pt"
            if validacao.is_file() and checkpoint.is_file():
                print(f"[{i}/4] Ja treinado: {rotulo}", flush=True)
                continue
            if validacao.exists() or checkpoint.exists():
                raise RuntimeError(
                    f"Arquivos parciais de {rotulo}; confira antes de retreinar: {validacao}, {checkpoint}"
                )
            cmd = [
                sys.executable, str(RAIZ / "scripts" / "executar_experimento.py"),
                "--loader-factory", "modulos.fabrica_carregadores:criar_carregadores",
                "--arquitetura", "resnet50",
                "--canais", "3" if args.fase == "tres" else "1",
                "--fases", args.fase,
                "--corte", args.corte,
                "--pos-weight", str(peso),
                "--epocas", str(args.epocas),
                "--batch", str(args.batch),
                "--paciencia", str(args.paciencia),
                "--seed", str(args.seed),
                "--lr", "1e-4",
                "--lr-backbone", str(args.lr_backbone),
                "--criterio-limiar", "acuracia_balanceada",
                "--criterio-checkpoint", "auc_roc",
                "--device", args.device,
                "--saida", str(resultados),
                "--checkpoints", str(checkpoints),
            ]
            if fine:
                cmd.append("--finetune-resnet")
            print(f"[{i}/4] Treinando {rotulo}", flush=True)
            subprocess.run(cmd, cwd=RAIZ, check=True)
    resumir(args.saida, args.fase, args.corte, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
