"""Consolida somente resultados de validacao da Issue #4 em uma tabela."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def _ler_historico(caminho: Path) -> tuple[float, int, int]:
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    melhor = min(linhas, key=lambda linha: float(linha["loss_validacao"]))
    return (
        sum(float(linha["tempo_segundos"]) for linha in linhas),
        len(linhas),
        int(melhor["epoca"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidar as 12 validacoes da Issue #4")
    parser.add_argument("--resultados", type=Path, default=Path("resultados"))
    parser.add_argument("--criterio", choices=("f1", "acuracia_balanceada", "auc_roc"), default="f1")
    parser.add_argument("--saida", type=Path, default=Path("resultados/comparacao_validacao.csv"))
    args = parser.parse_args()

    linhas = []
    for caminho in sorted(args.resultados.glob("*__validation.json")):
        nome = caminho.name.removesuffix("__validation.json")
        partes = nome.split("__")
        if len(partes) != 4:
            print(f"Ignorado (nome inesperado): {caminho}")
            continue
        arquitetura, fases, corte, seed = partes
        conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        metricas = conteudo["metricas"]
        historico = args.resultados / f"{nome}__history.csv"
        tempo, epocas, melhor_epoca = _ler_historico(historico)
        # Compatibilidade com resultados antigos que nao registraram AUC/balanced.
        predicoes = conteudo.get("predicoes_por_paciente", [])
        if "auc_roc" not in metricas or "acuracia_balanceada" not in metricas:
            from modulos.modulo_treinamento import RegistroPredicao, calcular_metricas
            registros = [RegistroPredicao(p["paciente"], p["rotulo"], p["probabilidade"]) for p in predicoes]
            if registros:
                extras = calcular_metricas(registros, float(metricas["limiar"]))
                metricas = {**extras, **metricas}
        linha = {
            "experimento": nome,
            "arquitetura": arquitetura,
            "fases": fases,
            "corte": corte,
            "seed": seed.removeprefix("seed-"),
            "epocas_executadas": epocas,
            "melhor_epoca": melhor_epoca,
            "tempo_treinamento_segundos": tempo,
            **{chave: metricas.get(chave) for chave in (
                "loss", "limiar", "sensibilidade", "especificidade", "precisao",
                "acuracia", "f1", "acuracia_balanceada", "auc_roc",
                "tn", "fp", "fn", "tp", "numero_pacientes",
            )},
        }
        linhas.append(linha)

    if not linhas:
        raise SystemExit("Nenhum arquivo *__validation.json encontrado")
    linhas.sort(
        key=lambda item: (
            -float(item[args.criterio]) if item[args.criterio] is not None else float("inf"),
            -float(item["acuracia_balanceada"] or 0),
            -float(item["sensibilidade"]),
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
            f"maior {args.criterio} de validacao; desempate por acuracia balanceada, sensibilidade e loss"
        ),
        "experimento": vencedor["experimento"],
        "checkpoint_local": f"checkpoints/{vencedor['experimento']}.pt",
        "limiar_validacao": vencedor["limiar"],
        "metricas_validacao": {
            chave: vencedor[chave]
            for chave in (
                "loss", "sensibilidade", "especificidade", "precisao",
                "acuracia", "f1", "acuracia_balanceada", "auc_roc", "tn", "fp", "fn", "tp",
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
    print(f"Melhor pela validacao: {linhas[0]['experimento']} (F1={float(linhas[0]['f1']):.4f})")
    print(f"Manifesto para as proximas issues: {caminho_manifesto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
