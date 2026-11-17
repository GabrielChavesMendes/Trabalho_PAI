"""Comando único: testes, recarga CPU e validação real dos quatro recebidos."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.execucao_issue4 import usar_venv_projeto, escolher_device


def executar_etapa(titulo, argumentos, log, ambiente):
    print(f"\n=== {titulo} ===", flush=True)
    with log.open("x", encoding="utf-8") as arquivo:
        with subprocess.Popen([sys.executable, *map(str, argumentos)], cwd=RAIZ, env=ambiente,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as processo:
            for linha in processo.stdout:
                print(linha, end="", flush=True)
                arquivo.write(linha)
            codigo = processo.wait()
    if codigo:
        raise RuntimeError(f"{titulo} falhou (codigo {codigo}). Consulte {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--pasta", type=Path, default=RAIZ / "checkpoints/integrados_issue4")
    p.add_argument("--saida", type=Path, help="Pasta nova; por padrao usa data/hora em resultados/verificacoes_issue4")
    a = p.parse_args()
    device = escolher_device(a.device)
    pasta = a.pasta.resolve()
    saida = (a.saida or RAIZ / "resultados/verificacoes_issue4" /
             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    saida.mkdir(parents=True, exist_ok=False)
    ambiente = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # Diretório exclusivo: evita ACL/cache de outro usuário e não apaga temporários anteriores.
    temporarios = RAIZ / ".venv" / ("verificacao-" + uuid4().hex)
    temporarios.mkdir()
    print(f"Python: {sys.executable}\nValidacao: {device}\nRelatorios: {saida}", flush=True)
    resumo = {"device_validacao": device, "python": sys.executable,
              "treinamento_executado": False, "teste_avaliado": False}
    try:
        executar_etapa("1/3 Testes automatizados", ["-m", "pytest", str(RAIZ / "tests"), "-q",
                       "-c", str(RAIZ / "pytest.ini"), "--rootdir", str(RAIZ), "--confcutdir", str(RAIZ),
                       "--basetemp", str(temporarios / "tmp"), "-o", f"cache_dir={temporarios / 'cache'}"],
                       saida / "testes.txt", ambiente)
        script = RAIZ / "scripts/verificar_checkpoints_integrados_issue4.py"
        executar_etapa("2/3 Recarga dos quatro em CPU", [script, "--pasta", pasta, "--device", "cpu",
                       "--saida", saida / "recarga_cpu"], saida / "recarga_cpu.txt", ambiente)
        executar_etapa("3/3 Validacao real: 33 pacientes", [script, "--pasta", pasta, "--device", device,
                       "--validar-dataset", "--saida", saida / "validacao"], saida / "validacao.txt", ambiente)
        resumo["status"] = "aprovado"
    except (OSError, RuntimeError) as exc:
        resumo.update(status="erro", erro=str(exc))
        print(f"ERRO: {exc}", file=sys.stderr)
    with (saida / "resumo.json").open("x", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    if resumo["status"] == "erro":
        return 1
    print("\nAPROVADO: testes + 4 recargas CPU + validacao real dos 33. Nenhum treino ou teste final executado.")
    return 0


if __name__ == "__main__":
    try:
        usar_venv_projeto()
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        raise SystemExit(1)
