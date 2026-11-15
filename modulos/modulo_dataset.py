"""Issue #2: leitura, validação e preparação de DCE-MRI I-SPY1 MinCrop.

Este módulo fornece cortes de volumes NIfTI 3D às redes convolucionais 2D.
Usa apenas imagens DCE como entrada; o CSV fornece rótulos e índices de fase,
e a máscara serve para selecionar o corte de maior área tumoral.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

COLUNAS_OBRIGATORIAS = ("pid", "pCR", "dataset", "test", "pre", "post_early", "post_late")
FASES = ("pre", "post_early", "post_late")
PARTICOES = {"treino": 0, "teste": 1, "validacao": 2}
ESTRATEGIAS = ("central", "maior_tumor")
OPCOES_FASE = (*FASES, "tres")


@dataclass(frozen=True)
class Paciente:
    identificador: str
    pcr: int
    particao_codigo: int
    caminhos_dce: tuple[Path, Path, Path]
    caminho_mascara: Path


def carregar_metadados(caminho_csv: str | Path) -> pd.DataFrame:
    """Valida os 172 registros I-SPY1 do CSV oficial do Canvas."""
    caminho = Path(caminho_csv)
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {caminho}")
    tabela = pd.read_csv(caminho)
    ausentes = sorted(set(COLUNAS_OBRIGATORIAS) - set(tabela.columns))
    if ausentes:
        raise ValueError(f"CSV sem colunas obrigatórias: {', '.join(ausentes)}")
    if len(tabela) != 172 or not tabela["dataset"].eq("spy1").all():
        raise ValueError("Esperados exatamente 172 pacientes I-SPY1 (dataset=spy1).")
    if tabela["pid"].isna().any() or tabela["pid"].duplicated().any():
        raise ValueError("Identificadores de pacientes vazios ou duplicados.")
    if not tabela["pid"].astype(str).str.fullmatch(r"ISPY1_\d+").all():
        raise ValueError("O CSV contém identificadores de pacientes inesperados.")
    for coluna in ("pCR", "test", *FASES):
        numeros = pd.to_numeric(tabela[coluna], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeros).all() or (numeros != np.floor(numeros)).any():
            raise ValueError(f"Valores ausentes, não inteiros ou inválidos em {coluna}.")
        if coluna in FASES and (numeros < 0).any():
            raise ValueError(f"Índices temporais negativos em {coluna}.")
    if not set(tabela["pCR"].astype(int)).issubset({0, 1}):
        raise ValueError("pCR deve ser 0 ou 1.")
    if not set(tabela["test"].astype(int)).issubset(set(PARTICOES.values())):
        raise ValueError("Código de partição fora de {0,1,2}.")
    return tabela


def criar_manifesto(tabela: pd.DataFrame, pasta_dataset: str | Path) -> list[Paciente]:
    """Associa cada paciente às fases reais, sem supor índices 0/1/2 fixos."""
    raiz = Path(pasta_dataset)
    manifesto = []
    for registro in tabela.itertuples(index=False):
        pid = str(registro.pid)
        prefixo = f"{pid}_spy1_vis1"
        caminhos = tuple(
            raiz / "dce" / f"{prefixo}_dce_aqc_{int(getattr(registro, fase))}.nii.gz"
            for fase in FASES
        )
        mascara = raiz / "mask" / f"{prefixo}_mask.nii.gz"
        manifesto.append(Paciente(pid, int(registro.pCR), int(registro.test), caminhos, mascara))
    return manifesto


def selecionar_particao(manifesto: list[Paciente], nome: str) -> list[Paciente]:
    """Separa por paciente; códigos oficiais: 0 treino, 1 teste, 2 validação."""
    if nome not in PARTICOES:
        raise ValueError(f"Partição inválida: {nome}. Use {', '.join(PARTICOES)}.")
    return [p for p in manifesto if p.particao_codigo == PARTICOES[nome]]


def verificar_arquivos(manifesto: list[Paciente]) -> list[str]:
    """Lista arquivos faltantes; não substitui volumes ausentes por zeros."""
    erros = []
    for paciente in manifesto:
        caminhos = (*paciente.caminhos_dce, paciente.caminho_mascara)
        for nome, caminho in zip((*FASES, "mask"), caminhos):
            if not caminho.is_file():
                erros.append(f"{paciente.identificador}: arquivo {nome} ausente: {caminho}")
    return erros


def ler_volume(caminho: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê um NIfTI 3D [Z,256,256] em float32, preservando affine.

    A importação tardia permite validar o CSV mesmo sem o nibabel instalado.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Instale nibabel: pip install -r requirements.txt") from exc
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Volume NIfTI ausente: {caminho}")
    imagem = nib.load(str(caminho))
    dados = np.asarray(imagem.dataobj, dtype=np.float32)
    if dados.ndim == 4 and dados.shape[-1] == 1:
        dados = dados[..., 0]
    if dados.ndim != 3 or min(dados.shape) < 1:
        raise ValueError(f"Esperado NIfTI 3D, obtido {dados.shape}: {caminho}")
    # O BreastDCEDL MinCrop armazena cada volume como [Z, H, W].
    # A checagem evita extrair um plano [Z, W] confundindo o eixo de cortes.
    if dados.shape[1:] != (256, 256):
        raise ValueError(f"Esperado volume MinCrop [Z,256,256], obtido {dados.shape}: {caminho}")
    if not np.isfinite(dados).all():
        raise ValueError(f"Volume contém NaN/Inf: {caminho}")
    return dados, np.asarray(imagem.affine)


def selecionar_corte(mascara: np.ndarray, estrategia: str) -> int:
    """Retorna índice de corte no eixo 0 de um volume [Z,H,W]."""
    if mascara.ndim != 3:
        raise ValueError("Máscara deve ser tridimensional.")
    if estrategia == "central":
        return mascara.shape[0] // 2
    if estrategia == "maior_tumor":
        areas = np.count_nonzero(mascara > 0, axis=(1, 2))
        if not areas.any():
            raise ValueError("Máscara não possui pixels tumorais positivos.")
        return int(areas.argmax())
    raise ValueError(f"Estratégia inválida: {estrategia}. Use {ESTRATEGIAS}.")


def normalizar_canais(canais: np.ndarray) -> np.ndarray:
    """Percentis 1/99 calculados por amostra, conjuntamente entre canais.

    Preserva diferenças relativas de intensidade entre as fases dentro da amostra,
    sem utilizar estatísticas de outros pacientes ou do conjunto de teste.
    """
    if canais.ndim != 3 or not np.isfinite(canais).all():
        raise ValueError("Esperado tensor de canais [C,H,W] finito.")
    inferior, superior = np.percentile(canais, [1, 99])
    if superior <= inferior:
        return np.zeros_like(canais, dtype=np.float32)
    escala = (canais.astype(np.float32) - np.float32(inferior)) / np.float32(superior - inferior)
    return np.ascontiguousarray(np.clip(escala, 0.0, 1.0), dtype=np.float32)


def carregar_amostra(
    paciente: Paciente,
    fase: str = "tres",
    estrategia: str = "maior_tumor",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Carrega 1 ou 3 fases, seleciona Z no eixo 0 e devolve cortes 2D.

    Imagem: [C,H,W]; máscara: [1,H,W]. A máscara nunca entra como canal
    de classificação (a comparação com máscara é opcional no enunciado).
    """
    if fase not in OPCOES_FASE:
        raise ValueError(f"Fase inválida: {fase}. Use {OPCOES_FASE}.")
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"Estratégia inválida: {estrategia}. Use {ESTRATEGIAS}.")
    mascara, affine_mask = ler_volume(paciente.caminho_mascara)
    indice_fases = range(3) if fase == "tres" else (FASES.index(fase),)
    imagens = []
    for indice in indice_fases:
        dados, affine = ler_volume(paciente.caminhos_dce[indice])
        if dados.shape != mascara.shape:
            raise ValueError(f"Dimensões diferentes de DCE/máscara para {paciente.identificador}: {dados.shape} vs {mascara.shape}")
        if not np.allclose(affine, affine_mask, rtol=1e-4, atol=1e-3):
            raise ValueError(f"Volumes e máscara não estão alinhados espacialmente: {paciente.identificador}")
        imagens.append(dados)
    z = selecionar_corte(mascara, estrategia)
    canais = np.stack([volume[z, :, :] for volume in imagens], axis=0)
    dados_normalizados = normalizar_canais(canais)
    mascara_2d = np.ascontiguousarray((mascara[z, :, :] > 0)[None, :, :], dtype=np.float32)
    return torch.from_numpy(dados_normalizados), torch.from_numpy(mascara_2d), z


class DatasetDCE(Dataset):
    """Dataset PyTorch sob demanda: 1 paciente/corte por item, sem vazamento.

    Selecione primeiro a partição com selecionar_particao(), e apenas depois
    instancie este Dataset. Augmentação é responsabilidade da Issue #1.
    """
    def __init__(self, pacientes: list[Paciente], fase: str = "tres", estrategia: str = "maior_tumor"):
        if fase not in OPCOES_FASE or estrategia not in ESTRATEGIAS:
            raise ValueError("Configuração inválida de fase ou estratégia.")
        if len({p.identificador for p in pacientes}) != len(pacientes):
            raise ValueError("O Dataset recebeu pacientes repetidos.")
        self.pacientes = list(pacientes)
        self.fase = fase
        self.estrategia = estrategia

    def __len__(self) -> int:
        return len(self.pacientes)

    def __getitem__(self, indice: int) -> dict:
        paciente = self.pacientes[indice]
        imagem, mascara, z = carregar_amostra(paciente, self.fase, self.estrategia)
        return {
            "imagem": imagem,
            "mascara": mascara,
            "rotulo": torch.tensor(float(paciente.pcr), dtype=torch.float32),
            "paciente": paciente.identificador,
            "corte_z": z,
        }


def resumo(tabela: pd.DataFrame) -> str:
    linhas = ["BreastDCEDL I-SPY1 MinCrop — distribuição oficial por paciente:"]
    linhas.append(f"Total: {len(tabela)} | pCR=0: {(tabela.pCR == 0).sum()} | pCR=1: {(tabela.pCR == 1).sum()}")
    for nome, codigo in PARTICOES.items():
        fatia = tabela.loc[tabela["test"] == codigo]
        linhas.append(f"{nome} (test={codigo}): {len(fatia)}; pCR=0: {(fatia.pCR == 0).sum()}; pCR=1: {(fatia.pCR == 1).sum()}")
    return "\n".join(linhas)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validar os dados I-SPY1 MinCrop e inspecionar amostras.")
    parser.add_argument("--csv", default="dataset/BreastDCEDL_SPY1_min_crop_metadata.csv")
    parser.add_argument("--dataset", default="dataset/BreastDCEDL_ISPY1_min_crop")
    parser.add_argument("--verificar-arquivos", action="store_true", help="Conferir os caminhos dos 172 pacientes")
    parser.add_argument("--inspecionar-paciente", metavar="PID", help="Carregar e analisar NIfTI real de um paciente")
    parser.add_argument("--fase", choices=OPCOES_FASE, default="tres")
    parser.add_argument("--estrategia", choices=ESTRATEGIAS, default="maior_tumor")
    args = parser.parse_args()
    try:
        tabela = carregar_metadados(args.csv)
        print(resumo(tabela))
        manifesto = criar_manifesto(tabela, args.dataset)
        if args.verificar_arquivos:
            erros = verificar_arquivos(manifesto)
            print(f"Arquivos ausentes: {len(erros)}")
            for erro in erros[:20]:
                print("ERRO:", erro)
            if erros:
                return 1
        if args.inspecionar_paciente:
            paciente = next((p for p in manifesto if p.identificador == args.inspecionar_paciente), None)
            if paciente is None:
                raise ValueError(f"Paciente não encontrado no CSV: {args.inspecionar_paciente}")
            imagem, mascara, z = carregar_amostra(paciente, args.fase, args.estrategia)
            print(f"Paciente: {paciente.identificador}; pCR={paciente.pcr}; z={z}")
            print(f"Imagem [canais,altura,largura]: {tuple(imagem.shape)}; máscara: {tuple(mascara.shape)}")
            print(f"Faixa de intensidade: {imagem.min().item():.4f}–{imagem.max().item():.4f}")
    except (OSError, ImportError, ValueError, TypeError) as exc:
        print(f"ERRO: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
