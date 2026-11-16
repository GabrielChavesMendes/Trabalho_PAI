"""Smoke tests locais da Issue #4; nao produzem resultados experimentais."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.modulo_redes import criar_modelo  # noqa: E402
from modulos.modulo_treinamento import (  # noqa: E402
    ConfiguracaoTreinamento,
    avaliar_modelo,
    carregar_checkpoint,
    escolher_dispositivo,
    salvar_grafico_convergencia,
    salvar_historico_csv,
    salvar_matriz_confusao,
    salvar_resultado_json,
    treinar_experimento,
)


class _DatasetSintetico(Dataset):
    def __init__(self, canais: int, prefixo: str = "SMOKE"):
        gerador = torch.Generator().manual_seed(123)
        self.imagens = torch.rand(8, canais, 64, 64, generator=gerador)
        self.rotulos = torch.tensor([0, 1] * 4).float()
        self.prefixo = prefixo

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, indice):
        return self.imagens[indice], self.rotulos[indice], f"{self.prefixo}-{indice:02d}"


def criar_carregadores_sinteticos(
    configuracao_temporal: str,
    estrategia_corte: str,
    tamanho_lote: int,
    seed: int,
):
    """Fabrica exclusiva para validar o CLI; nunca usar como experimento real."""

    canais = 3 if configuracao_temporal == "tres_fases" else 1
    return {
        "treino": DataLoader(
            _DatasetSintetico(canais, "TREINO"),
            batch_size=tamanho_lote,
            shuffle=False,
        ),
        "validacao": DataLoader(
            _DatasetSintetico(canais, "VALIDACAO"),
            batch_size=tamanho_lote,
            shuffle=False,
        ),
        "teste": DataLoader(
            _DatasetSintetico(canais, "TESTE"),
            batch_size=tamanho_lote,
            shuffle=False,
        ),
        "metadados": {
            "fases": [configuracao_temporal + "_sintetico"],
            "normalizacao": "nenhuma_sintetica",
            "corte": estrategia_corte + "_sintetico",
            "augmentation": "nenhum_sintetico",
            "particoes": "ids_sinteticos_disjuntos",
            "seed_factory": seed,
            "nao_e_resultado_experimental": True,
        },
    }


def validar_redes(dispositivo: torch.device, baixar_pesos: bool) -> None:
    for canais in (1, 3):
        imagens = torch.rand(2, canais, 64, 64, device=dispositivo)
        rotulos = torch.tensor([[0.0], [1.0]], device=dispositivo)
        for nome in ("cnn", "resnet50", "mobilenetv2"):
            pre_treinado = baixar_pesos and nome != "cnn"
            modelo = criar_modelo(nome, canais, pre_treinado=pre_treinado).to(dispositivo)
            modelo.train()
            logits = modelo(imagens)
            assert logits.shape == (2, 1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, rotulos)
            loss.backward()
            treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
            assert treinaveis > 0
            print(
                f"OK forward/backward: {nome}, canais={canais}, "
                f"device={dispositivo}, treinaveis={treinaveis}"
            )
            del modelo, logits, loss
            if dispositivo.type == "cuda":
                torch.cuda.empty_cache()


def validar_treino_checkpoint(dispositivo: torch.device) -> None:
    carregador = DataLoader(_DatasetSintetico(1), batch_size=4, shuffle=False)
    modelo = criar_modelo("cnn", 1, pre_treinado=False)
    configuracao = ConfiguracaoTreinamento(
        arquitetura="cnn",
        canais_entrada=1,
        configuracao_temporal="post_early",
        estrategia_corte="central",
        epocas=2,
        tamanho_lote=4,
        paciencia=2,
    )
    with tempfile.TemporaryDirectory(prefix="pai-issue4-") as pasta:
        checkpoint = Path(pasta) / "smoke.pt"
        historico = treinar_experimento(
            modelo,
            carregador,
            carregador,
            configuracao,
            checkpoint,
            {
                "fases": ["post_early_sintetico"],
                "normalizacao": "nenhuma_sintetica",
                "corte": "nao_aplicavel_sintetico",
                "augmentation": "nenhum_sintetico",
                "particoes": "nao_aplicavel_sintetico",
                "tipo": "dados_sinteticos_exclusivos_para_smoke_test",
                "nao_e_resultado_experimental": True,
            },
            dispositivo,
        )
        recarregado = criar_modelo("cnn", 1, pre_treinado=False)
        dados = carregar_checkpoint(checkpoint, recarregado)
        assert historico
        assert dados["preprocessamento"]["nao_e_resultado_experimental"] is True
        recarregado.to(dispositivo)
        criterio = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([3.0], device=dispositivo)
        )
        _, metricas, predicoes = avaliar_modelo(
            recarregado, carregador, criterio, dispositivo
        )
        salvar_historico_csv(Path(pasta) / "history.csv", historico)
        salvar_grafico_convergencia(Path(pasta) / "loss.png", historico)
        salvar_matriz_confusao(Path(pasta) / "confusion.png", metricas)
        salvar_resultado_json(Path(pasta) / "metrics.json", metricas, predicoes)
        for arquivo in ("history.csv", "loss.png", "confusion.png", "metrics.json"):
            assert (Path(pasta) / arquivo).is_file()
        print(f"OK treino/checkpoint sintetico: melhor epoca={dados['epoca']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--baixar-pesos",
        action="store_true",
        help="Valida tambem os pesos ImageNet oficiais",
    )
    args = parser.parse_args()
    dispositivo = escolher_dispositivo(args.device)
    print(f"PyTorch={torch.__version__}; CUDA build={torch.version.cuda}")
    if dispositivo.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(dispositivo)}")
    validar_redes(dispositivo, args.baixar_pesos)
    validar_treino_checkpoint(dispositivo)
    print("Todos os smoke tests da Issue #4 passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
