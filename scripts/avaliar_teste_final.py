"""Recarrega o melhor checkpoint e avalia uma unica vez o teste reservado."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.fabrica_carregadores import criar_carregadores  # noqa: E402
from modulos.modulo_redes import criar_modelo  # noqa: E402
from modulos.modulo_treinamento import (  # noqa: E402
    avaliar_modelo,
    carregar_checkpoint,
    escolher_dispositivo,
    salvar_matriz_confusao,
    salvar_resultado_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliar o teste do modelo escolhido na validacao")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    args = parser.parse_args()

    dispositivo = escolher_dispositivo(args.device)
    bruto = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuracao = bruto.get("configuracao", {})
    selecao = bruto.get("selecao_validacao")
    if not selecao or "limiar" not in selecao:
        raise ValueError("checkpoint nao possui limiar selecionado exclusivamente na validacao")

    modelo = criar_modelo(
        configuracao["arquitetura"],
        int(configuracao["canais_entrada"]),
        pre_treinado=False,
    )
    carregar_checkpoint(args.checkpoint, modelo, dispositivo)
    modelo.to(dispositivo)
    carregadores = criar_carregadores(
        configuracao_temporal=configuracao["configuracao_temporal"],
        estrategia_corte=configuracao["estrategia_corte"],
        tamanho_lote=int(configuracao["tamanho_lote"]),
        seed=int(configuracao["seed"]),
        normalizacao_entrada=configuracao.get("normalizacao_entrada", "percentil"),
        usar_mascara=bool(bruto.get("preprocessamento", {}).get("mascara_como_filtro", False)),
        recorte_roi=bool(bruto.get("preprocessamento", {}).get("recorte_roi", False)),
        representacao_entrada=str(
            bruto.get("preprocessamento", {}).get("representacao_entrada", "fases")
        ),
    )
    criterio = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(configuracao["pos_weight"])], device=dispositivo)
    )
    limiar = float(selecao["limiar"])
    loss, metricas, predicoes = avaliar_modelo(
        modelo, carregadores["teste"], criterio, dispositivo, limiar
    )
    metricas = {**metricas, "loss": loss}
    nome = args.checkpoint.stem
    prefixo = args.saida / f"{nome}__test"
    salvar_resultado_json(prefixo.with_suffix(".json"), metricas, predicoes)
    salvar_matriz_confusao(
        prefixo.with_name(prefixo.name + "__confusion-matrix.png"), metricas
    )

    exemplos = []
    for item in predicoes:
        previsto = int(item.probabilidade >= limiar)
        exemplos.append({
            "paciente": item.paciente,
            "rotulo": item.rotulo,
            "probabilidade": item.probabilidade,
            "previsto": previsto,
            "correto": int(previsto == item.rotulo),
            "tipo": "acerto" if previsto == item.rotulo else ("falso_positivo" if previsto else "falso_negativo"),
        })
    exemplos.sort(key=lambda item: (item["correto"], -abs(float(item["probabilidade"]) - limiar)))
    with prefixo.with_name(prefixo.name + "__examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.DictWriter(
            arquivo, fieldnames=list(exemplos[0]), lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(exemplos)
    print(f"Teste avaliado sem novo treino: {len(predicoes)} pacientes; F1={float(metricas['f1']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
