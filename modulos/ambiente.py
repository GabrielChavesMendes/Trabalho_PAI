"""Pequenos utilitários de execução dos CLIs; não importa modelos ou treina."""
import os
from pathlib import Path
import sys
import subprocess

RAIZ = Path(__file__).resolve().parents[1]


def usar_venv_projeto():
    """Permite `python scripts/...` mesmo se o Python global for CPU-only."""
    python = RAIZ / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError("Ambiente .venv ausente; use o ambiente existente do projeto")
    if not os.path.samefile(sys.executable, python):
        # subprocess faz a citação correta dos caminhos com espaços no Windows.
        resultado = subprocess.run([str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
        raise SystemExit(resultado.returncode)


def escolher_device(nome="auto"):
    import torch
    if nome == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if nome == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponivel na .venv")
    if nome not in ("cpu", "cuda"):
        raise ValueError("Device deve ser auto, cpu ou cuda")
    return nome
