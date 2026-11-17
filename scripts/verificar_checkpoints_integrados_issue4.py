"""Confere recarga dos quatro pesos externos e, opcionalmente, validacao real.

Copiar os quatro arquivos selecionados para checkpoints/integrados_issue4/.
Sem --validar-dataset faz teste sintético de carregamento; nao reproduz metricas.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.checkpoints_integrados_issue4 import (
    carregar_modelo_checkpoint, normalizacao_checkpoint,
)
from modulos.fabrica_carregadores import criar_carregadores
from modulos.modulo_treinamento import avaliar_modelo, calcular_metricas

PESOS = {
    "cnn__post-early__central__seed-42__final-v2.pt": ("cnn", "post_early", "central"),
    "resnet50__post-early__central__seed-42__final-v2.pt": ("resnet50", "post_early", "central"),
    "resnet50__post-early__maior-tumor__seed-84__imagenet-resnet-s84.pt": ("resnet50", "post_early", "maior_tumor"),
    "resnet50__tres__maior-tumor__seed-42__final-v2.pt": ("resnet50", "tres", "maior_tumor"),
}
METRICAS = ("sensibilidade", "especificidade", "precisao", "acuracia", "f1", "auc_roc", "acuracia_balanceada")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verificar pesos integrados da Issue 4")
    parser.add_argument("--pasta", type=Path, default=RAIZ / "checkpoints" / "integrados_issue4")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--validar-dataset", action="store_true", help="Reexecutar validacao nos 33 pacientes reais")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponivel")
    faltantes = [nome for nome in PESOS if not (args.pasta / nome).is_file()]
    if faltantes:
        raise FileNotFoundError("Checkpoints ausentes: " + ", ".join(faltantes))

    device = torch.device(args.device)
    for i, (nome, esperado) in enumerate(PESOS.items(), 1):
        modelo, dados = carregar_modelo_checkpoint(args.pasta / nome, device)
        conf = dados["configuracao"]
        real = (conf["arquitetura"], conf["configuracao_temporal"], conf["estrategia_corte"])
        if real != esperado:
            raise ValueError(f"Configuracao incorreta para {nome}: {real}")
        norma = normalizacao_checkpoint(dados)
        with torch.inference_mode():
            x = torch.rand(1, int(conf["canais_entrada"]), 64, 64, device=device)
            y = modelo(x)
        if y.shape != (1, 1) or not bool(torch.isfinite(y).all()):
            raise AssertionError(f"Saida invalida para {nome}")
        print(f"[{i}/4] Recarga estrita OK: {nome}; normalizacao={norma}", flush=True)
        if args.validar_dataset:
            carregadores = criar_carregadores(
                configuracao_temporal=conf["configuracao_temporal"],
                estrategia_corte=conf["estrategia_corte"],
                tamanho_lote=int(conf["tamanho_lote"]),
                seed=int(conf["seed"]),
                normalizacao_entrada=norma,
            )
            pos_weight = float(conf["pos_weight"])
            criterio = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device)
            )
            _, _, predicoes = avaliar_modelo(
                modelo, carregadores["validacao"], criterio, device,
                limiar=float(dados["selecao_validacao"]["limiar"]),
            )
            metricas = calcular_metricas(
                predicoes, float(dados["selecao_validacao"]["limiar"])
            )
            reportadas = dados["selecao_validacao"]["metricas"]
            erros = [
                f"{chave}: obtido={metricas[chave]:.5f}, salvo={reportadas[chave]:.5f}"
                for chave in METRICAS
                if abs(float(metricas[chave]) - float(reportadas[chave])) > 0.001
            ]
            if metricas["numero_pacientes"] != 33 or erros:
                raise AssertionError(
                    f"Validacao nao reproduzida para {nome}: "
                    + "; ".join(erros)
                    + f" (pacientes={metricas['numero_pacientes']})"
                )
            print(f"    Validacao reproduzida: 33 pacientes; F1={metricas['f1']:.4f}; AUC={metricas['auc_roc']:.4f}",flush=True)
        del modelo, dados
    print("Quatro checkpoints externos verificados." + (" Validacao real OK." if args.validar_dataset else " Falta testar no dataset real."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
