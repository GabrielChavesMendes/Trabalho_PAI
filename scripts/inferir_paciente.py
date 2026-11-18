"""Demonstração de um paciente com pesos salvos; interface/Grad-CAM são Issue #5."""
import argparse
import csv
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.ambiente import usar_venv_projeto, escolher_device

CATALOGO = RAIZ / "resultados/grade_definitiva_issue4_20261117_01/artigo/shortlist_integrada_validacao_12.csv"


def modelos_selecionados():
    with CATALOGO.open(encoding="utf-8", newline="") as f:
        modelos = list(csv.DictReader(f))
    esperadas = {(a,f,c) for a in ("cnn", "resnet50", "mobilenetv2")
                 for f in ("post_early", "tres") for c in ("central", "maior_tumor")}
    if len(modelos) != 12 or {(m["arquitetura"],m["fases"],m["corte"]) for m in modelos} != esperadas:
        raise ValueError("Catalogo deve preservar exatamente as 12 combinacoes finais")
    return modelos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    selecao = p.add_mutually_exclusive_group()
    selecao.add_argument("--modelo", type=int, choices=range(1,13), metavar="1..12")
    selecao.add_argument("--checkpoint", type=Path, help="Caminho direto de um checkpoint compativel")
    p.add_argument("--paciente", help="ID do CSV, por exemplo ISPY1_1009")
    p.add_argument("--listar", action="store_true", help="Mostrar os 12 selecionados e a disponibilidade dos pesos")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    a = p.parse_args()
    modelos = modelos_selecionados()
    if a.listar:
        for i,m in enumerate(modelos,1):
            disponivel = "disponivel" if (RAIZ / m["caminho_checkpoint"]).is_file() else "peso ausente"
            print(f"{i:2}: {m['arquitetura']:11} | {m['fases']:10} | {m['corte']:11} | {disponivel}")
        return 0
    if not a.paciente or (a.modelo is None and a.checkpoint is None):
        p.error("Informe --modelo 1..12 (ou --checkpoint) e --paciente; use --listar para ver as opcoes")
    from modulos.checkpoints import carregar_modelo_checkpoint, pacientes_dataset, prever_paciente
    device = escolher_device(a.device)
    cp = a.checkpoint.resolve() if a.checkpoint else RAIZ / modelos[a.modelo-1]["caminho_checkpoint"]
    pacientes = [paciente for paciente in pacientes_dataset() if paciente.identificador == a.paciente]
    if len(pacientes) != 1:
        raise ValueError(f"Paciente nao encontrado no CSV: {a.paciente}")
    modelo,dados = carregar_modelo_checkpoint(cp, device)
    if a.modelo:
        esperado = modelos[a.modelo-1]
        conf = dados["configuracao"]
        if (conf["arquitetura"], conf["configuracao_temporal"], conf["estrategia_corte"]) != (
                esperado["arquitetura"], esperado["fases"], esperado["corte"]):
            raise ValueError("Checkpoint diverge da combinacao selecionada no catalogo")
    resultado = prever_paciente(modelo,dados,pacientes[0],device)
    print(json.dumps({"checkpoint":str(cp), "device":device, **resultado}, ensure_ascii=False, indent=2))
    print(f"pCR=1: {resultado['probabilidade_pcr']:.6%} | limiar: {resultado['limiar']!r} | "
          f"classificacao: pCR={resultado['classificacao_pcr']}")
    return 0


if __name__ == "__main__":
    try:
        usar_venv_projeto()
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        raise SystemExit(1)
