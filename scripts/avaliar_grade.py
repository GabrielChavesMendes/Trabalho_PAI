"""Avalia todas as 12 configuracoes FINALIZADAS no teste, sem retreinar.

Execute somente quando pesos, hiperparametros e limiares estiverem congelados
pela VALIDACAO. Os dados do teste nunca participam da escolha dos modelos.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.fabrica_carregadores import criar_carregadores
from modulos.modulo_redes import criar_modelo
from modulos.modulo_treinamento import (
    avaliar_modelo, carregar_checkpoint, escolher_dispositivo,
    salvar_matriz_confusao, salvar_resultado_json,
)

ARQUITETURAS = ("cnn", "mobilenetv2", "resnet50")
FASES = ("post_early", "tres")
CORTES = ("central", "maior_tumor")


def configuracoes(seed: int) -> list[tuple[str, str, str, str]]:
    return [
        (modelo, fase, corte, f"{modelo}__{fase.replace('_','-')}__{corte.replace('_','-')}__seed-{seed}")
        for corte in CORTES for fase in FASES for modelo in ARQUITETURAS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Teste final dos 12 checkpoints fixados pela validacao")
    parser.add_argument("--checkpoints", type=Path, default=RAIZ / "checkpoints" / "grade_issue4_revisada")
    parser.add_argument("--resultados-validacao", type=Path, default=RAIZ / "resultados" / "grade_issue4_revisada")
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "grade_issue4_revisada" / "teste_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--confirmar-validacao-fechada", action="store_true", help="Obrigatorio: nao sera permitido ajustar apos consultar o teste")
    args = parser.parse_args()
    if not args.confirmar_validacao_fechada:
        parser.error("Finalize ajustes na VALIDACAO antes; use --confirmar-validacao-fechada apenas entao.")

    tarefas = configuracoes(args.seed)
    # Preflight: recusar avaliacao parcial se faltar checkpoint ou validacao.
    for _, _, _, nome in tarefas:
        checkpoint = args.checkpoints / f"{nome}.pt"
        validacao = args.resultados_validacao / f"{nome}__validation.json"
        if not checkpoint.is_file() or not validacao.is_file():
            raise FileNotFoundError(f"Falta checkpoint ou validacao para {nome}: {checkpoint} | {validacao}")
    args.saida.mkdir(parents=True, exist_ok=True)
    device = escolher_dispositivo(args.device)
    linhas = []
    for i, (arquitetura, fase, corte, nome) in enumerate(tarefas, 1):
        prefixo = args.saida / f"{nome}__test"
        destino_json = args.saida / f"{nome}__test.json"
        if destino_json.exists():
            dados = json.loads(destino_json.read_text(encoding="utf-8"))
            metricas = dados["metricas"]
            print(f"[{i}/12] Teste ja registrado: {nome}", flush=True)
        else:
            caminho = args.checkpoints / f"{nome}.pt"
            bruto = torch.load(caminho, map_location="cpu", weights_only=False)
            conf = bruto["configuracao"]
            selecao = bruto.get("selecao_validacao")
            if not selecao or "limiar" not in selecao:
                raise ValueError(f"Checkpoint sem limiar de validacao: {nome}")
            if (conf["arquitetura"] != arquitetura or
                conf["configuracao_temporal"] != fase or
                conf["estrategia_corte"] != corte or
                conf["canais_entrada"] != (3 if fase == "tres" else 1)):
                raise ValueError(f"Checkpoint com configuracao divergente: {nome}")
            modelo = criar_modelo(arquitetura, conf["canais_entrada"], pre_treinado=False)
            carregar_checkpoint(caminho, modelo, device)
            modelo.to(device)
            carregadores = criar_carregadores(
                configuracao_temporal=fase, estrategia_corte=corte,
                tamanho_lote=int(conf["tamanho_lote"]), seed=int(conf["seed"])
            )
            criterio = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([float(conf["pos_weight"])], device=device)
            )
            inicio = time.perf_counter()
            loss, metricas, predicoes = avaliar_modelo(
                modelo, carregadores["teste"], criterio, device, float(selecao["limiar"])
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metricas = {**metricas, "loss": loss, "tempo_teste_segundos": time.perf_counter() - inicio}
            if int(metricas["numero_pacientes"]) != 35:
                raise ValueError(f"Teste deve ter 35 pacientes, obtido {metricas['numero_pacientes']}")
            salvar_resultado_json(destino_json, metricas, predicoes)
            salvar_matriz_confusao(
                args.saida / f"{nome}__test__confusion-matrix.png", metricas
            )
            exemplos = args.saida / f"{nome}__test__examples.csv"
            with exemplos.open("w", newline="", encoding="utf-8") as arq:
                writer = csv.DictWriter(arq, fieldnames=("paciente", "rotulo", "probabilidade", "previsto", "correto"))
                writer.writeheader()
                for p in predicoes:
                    previsto = int(p.probabilidade >= selecao["limiar"])
                    writer.writerow({
                        "paciente": p.paciente, "rotulo": p.rotulo,
                        "probabilidade": p.probabilidade,
                        "previsto": previsto, "correto": int(previsto == p.rotulo)
                    })
            print(f"[{i}/12] {nome}: F1={metricas['f1']:.3f}, AUC={metricas['auc_roc']:.3f}", flush=True)
            del modelo, carregadores
            if device.type == "cuda":
                torch.cuda.empty_cache()
        linhas.append({"experimento": nome, **metricas})
    saida = args.saida / "comparacao_teste.csv"
    with saida.open("w", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"Concluido: 12/12 avaliacoes de teste. Tabela: {saida}")
    print("IMPORTANTE: resultados de teste nao podem orientar novos ajustes de treinamento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
