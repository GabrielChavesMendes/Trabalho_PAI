"""Consolida somente resultados de validacao da Issue #4 em uma tabela."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _ler_historico(caminho: Path, criterio_checkpoint: str) -> tuple[float, int, int]:
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if criterio_checkpoint == "loss":
        melhor = min(linhas, key=lambda linha: float(linha["loss_validacao"]))
    else:
        melhor = max(linhas, key=lambda linha: float(linha[criterio_checkpoint]))
    return (
        sum(float(linha["tempo_segundos"]) for linha in linhas),
        len(linhas),
        int(melhor["epoca"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidar as 12 validacoes da Issue #4")
    parser.add_argument("--resultados", type=Path, default=Path("resultados"))
    parser.add_argument("--saida", type=Path, default=Path("resultados/comparacao_validacao.csv"))
    parser.add_argument(
        "--criterio-checkpoint",
        choices=("loss", "auc_roc", "acuracia_balanceada"),
        default="loss",
    )
    parser.add_argument("--diretorio-checkpoints", default="checkpoints")
    args = parser.parse_args()

    linhas = []
    for caminho in sorted(args.resultados.glob("*__validation.json")):
        nome = caminho.name.removesuffix("__validation.json")
        partes = nome.split("__")
        if len(partes) < 4:
            print(f"Ignorado (nome inesperado): {caminho}")
            continue
        arquitetura, fases, corte, seed = partes[:4]
        tag = "__".join(partes[4:])
        conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        metricas = conteudo["metricas"]
        historico = args.resultados / f"{nome}__history.csv"
        tempo, epocas, melhor_epoca = _ler_historico(
            historico, args.criterio_checkpoint
        )
        linha = {
            "experimento": nome,
            "arquitetura": arquitetura,
            "fases": fases,
            "corte": corte,
            "seed": seed.removeprefix("seed-"),
            "tag": tag,
            "epocas_executadas": epocas,
            "melhor_epoca": melhor_epoca,
            "tempo_treinamento_segundos": tempo,
            **{chave: metricas[chave] for chave in (
                "loss", "limiar", "sensibilidade", "especificidade", "precisao",
                "acuracia", "f1", "auc_roc", "acuracia_balanceada",
                "tn", "fp", "fn", "tp", "numero_pacientes",
            )},
        }
        linhas.append(linha)

    if not linhas:
        raise SystemExit("Nenhum arquivo *__validation.json encontrado")
    linhas.sort(
        key=lambda item: (
            -float(item["auc_roc"]),
            -float(item["acuracia_balanceada"]),
            -float(item["f1"]),
            float(item["loss"]),
            str(item["experimento"]),
        )
    )
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo, fieldnames=list(linhas[0]), lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(linhas)
    vencedor = linhas[0]
    manifesto = {
        "criterio_selecao": (
            "maior AUC-ROC de validacao; desempate por acuracia balanceada, F1, loss e nome"
        ),
        "experimento": vencedor["experimento"],
        "checkpoint_local": f"{args.diretorio_checkpoints}/{vencedor['experimento']}.pt",
        "limiar_validacao": vencedor["limiar"],
        "metricas_validacao": {
            chave: vencedor[chave]
            for chave in (
                "loss", "sensibilidade", "especificidade", "precisao",
                "acuracia", "f1", "auc_roc", "acuracia_balanceada",
                "tn", "fp", "fn", "tp",
            )
        },
        "configuracao": {
            "arquitetura": vencedor["arquitetura"],
            "fases": vencedor["fases"],
            "corte": vencedor["corte"],
            "seed": vencedor["seed"],
            "melhor_epoca": vencedor["melhor_epoca"],
        },
        "observacao": "O checkpoint e local e ignorado pelo Git; o teste nao participa desta selecao.",
    }
    caminho_manifesto = args.saida.parent / "modelo_selecionado.json"
    caminho_manifesto.write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{len(linhas)} configuracoes consolidadas em {args.saida}")
    print(
        f"Melhor pela validacao: {linhas[0]['experimento']} "
        f"(AUC={float(linhas[0]['auc_roc']):.4f}, "
        f"BAcc={float(linhas[0]['acuracia_balanceada']):.4f})"
    )
    print(f"Manifesto para as proximas issues: {caminho_manifesto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
