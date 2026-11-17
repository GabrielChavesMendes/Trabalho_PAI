"""Recarga segura e verificavel dos checkpoints externos da Issue #4.

Nao altera o treinamento existente, nem usa fase espacial 2,5D ou ensemble.
Os modelos sao inicializados sem download: os pesos ImageNet ja fazem parte do
state_dict treinado e recebido em arquivo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion

from modulos.modulo_redes import criar_modelo

FASES_TRES = ["pre", "post_early", "post_late"]
ARQUITETURAS = {"cnn", "resnet50", "mobilenetv2"}


def carregar_dados_checkpoint(caminho: str | Path) -> dict[str, Any]:
    """Nao permite desserializacao Python arbitraria (weights_only=True)."""
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            Path(caminho), map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, dict) or checkpoint.get("formato_checkpoint") != 1:
        raise ValueError("Formato de checkpoint invalido")
    for campo in ("model_state_dict", "configuracao", "preprocessamento", "selecao_validacao"):
        if not isinstance(checkpoint.get(campo), dict):
            raise ValueError(f"Checkpoint sem metadados obrigatorios: {campo}")
    validar_configuracao(checkpoint)
    return checkpoint


def validar_configuracao(checkpoint: dict[str, Any]) -> None:
    conf = checkpoint["configuracao"]
    prep = checkpoint["preprocessamento"]
    arq = conf.get("arquitetura")
    fase = conf.get("configuracao_temporal")
    corte = conf.get("estrategia_corte")
    canais = conf.get("canais_entrada")
    if arq not in ARQUITETURAS or fase not in ("post_early", "tres"):
        raise ValueError("Arquitetura ou fase temporal fora da grade de 12")
    if corte not in ("central", "maior_tumor") or canais != (3 if fase == "tres" else 1):
        raise ValueError("Corte/canais incompativeis com a grade de 12")
    esperado = FASES_TRES if fase == "tres" else ["post_early"]
    if prep.get("fases") != esperado or prep.get("corte") != corte:
        raise ValueError("Metadados de fases/corte diferentes do modelo")
    if prep.get("mascara_como_filtro") or prep.get("recorte_roi"):
        raise ValueError("Checkpoint exige transformacao adicional nao suportada")
    if prep.get("representacao_entrada", "fases") != "fases":
        raise ValueError("Apenas as fases temporais originais sao admitidas")
    norma = prep.get("normalizacao")
    tipo = norma.get("tipo") if isinstance(norma, dict) else None
    if tipo not in ("percentil_1_99_conjunto", "percentil_1_99_seguido_normalizacao_imagenet"):
        raise ValueError("Normalizacao nao suportada")
    if tipo.endswith("imagenet"):
        medias = [0.485, 0.456, 0.406] if canais == 3 else [0.449]
        desvios = [0.229, 0.224, 0.225] if canais == 3 else [0.226]
        if any(abs(float(v) - e) > 1e-5 for v, e in zip(norma.get("medias", []), medias)) or len(norma.get("medias", [])) != canais:
            raise ValueError("Medias ImageNet incompativeis")
        if any(abs(float(v) - e) > 1e-5 for v, e in zip(norma.get("desvios", []), desvios)) or len(norma.get("desvios", [])) != canais:
            raise ValueError("Desvios ImageNet incompativeis")
    selecao = checkpoint["selecao_validacao"]
    if not 0 < float(selecao.get("limiar", 0)) < 1:
        raise ValueError("Limiar de validacao ausente/invalido")
    if not isinstance(selecao.get("metricas"), dict):
        raise ValueError("Metricas de validacao ausentes")


def normalizacao_checkpoint(checkpoint: dict[str, Any]) -> str:
    tipo = checkpoint["preprocessamento"]["normalizacao"]["tipo"]
    return "imagenet" if tipo.endswith("imagenet") else "percentil"


def carregar_modelo_checkpoint(
    caminho: str | Path, dispositivo: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Recupera o modelo SEM treinar nem baixar pesos; strict=True e eval()."""
    dados = carregar_dados_checkpoint(caminho)
    conf = dados["configuracao"]
    modelo = criar_modelo(
        conf["arquitetura"], int(conf["canais_entrada"]), pre_treinado=False
    )
    modelo.load_state_dict(dados["model_state_dict"], strict=True)
    modelo.to(dispositivo)
    modelo.eval()
    return modelo, dados
