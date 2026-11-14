"""Issue #2 — Inspeção e validação do BreastDCEDL I-SPY1 MinCrop.

Este módulo não faz treinamento. Verifica a correspondência entre o CSV oficial,
volumes DCE e máscaras. Não carrega dados clínicos como atributos das redes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUNAS_OBRIGATORIAS = ("pid", "pCR", "dataset", "test", "pre", "post_early", "post_late")
FASES = ("pre", "post_early", "post_late")


@dataclass(frozen=True)
class Paciente:
    identificador: str
    pcr: int
    particao_codigo: int
    caminhos_dce: tuple[Path, Path, Path]
    caminho_mascara: Path


def carregar_metadados(caminho_csv: str | Path) -> pd.DataFrame:
    """Valida o CSV específico do Canvas; não mistura pacientes de outros datasets."""
    caminho = Path(caminho_csv)
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {caminho}")
    tabela = pd.read_csv(caminho)
    ausentes = sorted(set(COLUNAS_OBRIGATORIAS) - set(tabela.columns))
    if ausentes:
        raise ValueError(f"CSV sem colunas obrigatórias: {', '.join(ausentes)}")
    if len(tabela) != 172 or not tabela['dataset'].eq('spy1').all():
        raise ValueError("Esperados exatamente 172 pacientes do subconjunto spy1 (I-SPY1 MinCrop).")
    if tabela['pid'].isna().any() or tabela['pid'].duplicated().any():
        raise ValueError('IDs de pacientes vazios ou duplicados no CSV.')
    for coluna in ('pCR', 'test', *FASES):
        if tabela[coluna].isna().any():
            raise ValueError(f'Valores ausentes na coluna {coluna}.')
        valores = tabela[coluna].to_numpy(dtype=float)
        if not np.isfinite(valores).all() or (valores != np.floor(valores)).any():
            raise ValueError(f'Valores não inteiros ou inválidos na coluna {coluna}.')
    if not set(tabela['pCR'].astype(int)).issubset({0, 1}):
        raise ValueError('Classe pCR deve ser 0 ou 1.')
    if not set(tabela['test'].astype(int)).issubset({0, 1, 2}):
        raise ValueError('Código de partição fora do conjunto {0, 1, 2}.')
    return tabela


def criar_manifesto(tabela: pd.DataFrame, pasta_dataset: str | Path) -> list[Paciente]:
    """Calcula caminhos de cada fase a partir do CSV (índices não necessariamente 0/1/2)."""
    raiz = Path(pasta_dataset)
    manifesto: list[Paciente] = []
    for registro in tabela.itertuples(index=False):
        pid = str(registro.pid)
        prefixo = f'{pid}_spy1_vis1'
        caminhos = tuple(raiz / 'dce' / f'{prefixo}_dce_aqc_{int(getattr(registro, fase))}.nii.gz'
                         for fase in FASES)
        mascara = raiz / 'mask' / f'{prefixo}_mask.nii.gz'
        manifesto.append(Paciente(pid, int(registro.pCR), int(registro.test), caminhos, mascara))
    return manifesto


def verificar_arquivos(manifesto: list[Paciente]) -> list[str]:
    """Nunca inventa uma imagem zerada caso o arquivo não exista."""
    erros: list[str] = []
    for paciente in manifesto:
        for nome, caminho in zip((*FASES, 'mask'), (*paciente.caminhos_dce, paciente.caminho_mascara)):
            if not caminho.is_file():
                erros.append(f'{paciente.identificador}: arquivo {nome} ausente: {caminho}')
    return erros


def resumo(tabela: pd.DataFrame) -> str:
    linhas = ['I-SPY1 MinCrop: distribuição de pacientes (códigos test sem interpretação clínica):']
    linhas.append(f'Total: {len(tabela)} | pCR=0: {(tabela.pCR == 0).sum()} | pCR=1: {(tabela.pCR == 1).sum()}')
    for codigo, fatia in tabela.groupby('test', sort=True):
        linhas.append(f'test={int(codigo)}: {len(fatia)} pacientes; pCR=0: {(fatia.pCR == 0).sum()}; pCR=1: {(fatia.pCR == 1).sum()}')
    linhas.append('ATENÇÃO: confirmar no README oficial se test=0/1/2 significa treino/validação/teste.')
    return '\n'.join(linhas)


def main() -> int:
    parser = argparse.ArgumentParser(description='Validar CSV e arquivos do I-SPY1 MinCrop (Issue #2).')
    parser.add_argument('--csv', default='dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    parser.add_argument('--dataset', default='dataset/BreastDCEDL_ISPY1_min_crop')
    parser.add_argument('--verificar-arquivos', action='store_true', help='Conferir fisicamente todos os NIfTI')
    args = parser.parse_args()
    try:
        tabela = carregar_metadados(args.csv)
        print(resumo(tabela))
        manifesto = criar_manifesto(tabela, args.dataset)
        if args.verificar_arquivos:
            erros = verificar_arquivos(manifesto)
            print(f'Validação física: {len(erros)} arquivo(s) ausente(s).')
            for erro in erros[:20]:
                print('ERRO:', erro)
            return 1 if erros else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f'ERRO: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
