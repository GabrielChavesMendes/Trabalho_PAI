"""Treina ensemble estratificado de 5 folds sem usar o teste para selecao."""

from __future__ import annotations

import json
import random
import sys
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulos.fabrica_carregadores import (  # noqa: E402
    CSV_PADRAO,
    DatasetCacheMemoria,
    DatasetPadronizado,
    DatasetVizinhanca2p5D,
)
from modulos.modulo_dataset import (  # noqa: E402
    DatasetDCE,
    carregar_metadados,
    criar_manifesto,
    selecionar_particao,
)
from modulos.modulo_augmentacao import listar_variacoes, transformar_par  # noqa: E402
from modulos.modulo_redes import configurar_fine_tuning, criar_modelo  # noqa: E402
from modulos.modulo_treinamento import (  # noqa: E402
    ConfiguracaoTreinamento,
    RegistroPredicao,
    avaliar_modelo,
    carregar_checkpoint,
    configurar_reprodutibilidade,
    escolher_dispositivo,
    registrar_validacao_checkpoint,
    salvar_historico_csv,
    salvar_matriz_confusao,
    salvar_resultado_json,
    selecionar_limiar_validacao,
    treinar_experimento,
)


def _folds_estratificados(pacientes, quantidade: int, seed: int):
    rng = random.Random(seed)
    folds = [[] for _ in range(quantidade)]
    for rotulo in (0, 1):
        classe = [p for p in pacientes if p.pcr == rotulo]
        rng.shuffle(classe)
        for indice, paciente in enumerate(classe):
            folds[indice % quantidade].append(paciente)
    return folds


def _base(pacientes, representacao: str):
    if representacao == "vizinhanca_2p5d":
        dataset = DatasetCacheMemoria(DatasetVizinhanca2p5D(pacientes))
        return DatasetPadronizado(
            dataset,
            torch.tensor([0.485, 0.456, 0.406]),
            torch.tensor([0.229, 0.224, 0.225]),
        )
    dataset = DatasetCacheMemoria(
        DatasetDCE(pacientes, fase="post_early", estrategia="maior_tumor")
    )
    return DatasetPadronizado(dataset, torch.tensor([0.449]), torch.tensor([0.226]))


class DatasetAumentadoCV(Dataset):
    """Mesmo augmentation oficial, sem depender dos codigos da particao original."""

    def __init__(self, dataset_base):
        self.dataset_base = dataset_base
        self.variacoes = listar_variacoes()

    def __len__(self):
        return len(self.dataset_base) * len(self.variacoes)

    def __getitem__(self, indice):
        indice_base, indice_variacao = divmod(indice, len(self.variacoes))
        amostra = dict(self.dataset_base[indice_base])
        variacao = self.variacoes[indice_variacao]
        imagem, mascara = transformar_par(
            amostra["imagem"].numpy(), amostra["mascara"].numpy(),
            variacao.angulo, variacao.espelhada,
        )
        amostra["imagem"] = torch.from_numpy(imagem)
        amostra["mascara"] = torch.from_numpy(mascara)
        return amostra


def main() -> int:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representacao", choices=("fase", "vizinhanca_2p5d"), default="fase"
    )
    args = parser.parse_args()

    pasta_dataset = Path(os.environ["PAI_DATASET_DIR"])
    sufixo = "ensemble_cv_2p5d_issue4" if args.representacao == "vizinhanca_2p5d" else "ensemble_cv_issue4"
    saida = RAIZ / "resultados" / sufixo
    checkpoints = RAIZ / "checkpoints" / sufixo
    saida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    tabela = carregar_metadados(CSV_PADRAO)
    manifesto = criar_manifesto(tabela, pasta_dataset)
    desenvolvimento = selecionar_particao(manifesto, "treino") + selecionar_particao(
        manifesto, "validacao"
    )
    teste = selecionar_particao(manifesto, "teste")
    folds = _folds_estratificados(desenvolvimento, 5, 84)
    dispositivo = escolher_dispositivo("cuda")
    oof: list[RegistroPredicao] = []
    teste_por_paciente: dict[str, list[float]] = defaultdict(list)
    rotulos_teste: dict[str, int] = {}

    for indice, validacao_pacientes in enumerate(folds, start=1):
        ids_validacao = {p.identificador for p in validacao_pacientes}
        treino_pacientes = [p for p in desenvolvimento if p.identificador not in ids_validacao]
        seed = 84 + indice - 1
        configurar_reprodutibilidade(seed)
        treino_base = _base(treino_pacientes, args.representacao)
        validacao_base = _base(validacao_pacientes, args.representacao)
        gerador = torch.Generator().manual_seed(seed)
        opcoes = {"batch_size": 32, "num_workers": 0, "pin_memory": True}
        treino_loader = DataLoader(
            DatasetAumentadoCV(treino_base), shuffle=True,
            generator=gerador, **opcoes,
        )
        # A validacao CV usa uma previsao por paciente, sem augmentation.
        validacao_loader = DataLoader(validacao_base, shuffle=False, **opcoes)
        configuracao = ConfiguracaoTreinamento(
            arquitetura="resnet50",
            canais_entrada=3 if args.representacao == "vizinhanca_2p5d" else 1,
            configuracao_temporal="post_early", estrategia_corte="maior_tumor",
            epocas=12, tamanho_lote=32, taxa_aprendizado=1e-4,
            taxa_aprendizado_backbone=1e-5, decaimento_pesos=1e-4,
            pos_weight=1, paciencia=4, criterio_limiar="acuracia_balanceada",
            criterio_checkpoint="auc_roc", fine_tuning="layer4",
            normalizacao_entrada="imagenet", tag=f"cv-fold-{indice}", seed=seed,
            representacao_entrada=args.representacao,
        )
        modelo = criar_modelo(
            "resnet50", 3 if args.representacao == "vizinhanca_2p5d" else 1,
            pre_treinado=True,
        )
        resumo = configurar_fine_tuning(modelo, "resnet50", "layer4")
        checkpoint = checkpoints / (
            f"resnet50__post-early__maior-tumor__seed-{seed}__"
            f"cv-fold-{indice}__{args.representacao}.pt"
        )
        metadados = {
            "fases": ["post_early"],
            "normalizacao": {"tipo": "percentil_1_99_seguido_normalizacao_imagenet"},
            "corte": "maior_tumor",
            "augmentation": {"treino": True, "validacao": False},
            "particoes": {"desenvolvimento": [0, 2], "fold": indice},
            "modelo": resumo,
            "protocolo": "5-fold estratificado; teste excluido",
            "representacao_entrada": args.representacao,
        }
        print(
            f"Fold {indice}/5: treino={len(treino_pacientes)}, "
            f"validacao={len(validacao_pacientes)}, seed={seed}", flush=True,
        )
        historico = treinar_experimento(
            modelo, treino_loader, validacao_loader, configuracao,
            checkpoint, metadados, dispositivo,
        )
        salvar_historico_csv(saida / f"fold-{indice}__history.csv", historico)
        carregar_checkpoint(checkpoint, modelo, dispositivo)
        criterio = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], device=dispositivo)
        )
        _, _, predicoes_fold = avaliar_modelo(
            modelo, validacao_loader, criterio, dispositivo, 0.5
        )
        oof.extend(predicoes_fold)

        teste_loader = DataLoader(_base(teste, args.representacao), shuffle=False, **opcoes)
        _, _, predicoes_teste = avaliar_modelo(
            modelo, teste_loader, criterio, dispositivo, 0.5
        )
        for item in predicoes_teste:
            teste_por_paciente[item.paciente].append(item.probabilidade)
            rotulos_teste[item.paciente] = item.rotulo

    limiar, metricas_oof = selecionar_limiar_validacao(
        oof, criterio="acuracia_balanceada"
    )
    salvar_resultado_json(saida / "ensemble__oof.json", metricas_oof, oof)
    salvar_matriz_confusao(saida / "ensemble__oof__confusion-matrix.png", metricas_oof)

    ensemble_teste = [
        RegistroPredicao(paciente, rotulos_teste[paciente], sum(valores) / len(valores))
        for paciente, valores in sorted(teste_por_paciente.items())
    ]
    from modulos.modulo_treinamento import calcular_metricas  # noqa: E402

    metricas_teste = calcular_metricas(ensemble_teste, limiar)
    salvar_resultado_json(saida / "ensemble__test.json", metricas_teste, ensemble_teste)
    salvar_matriz_confusao(
        saida / "ensemble__test__confusion-matrix.png", metricas_teste
    )
    resumo = {
        "protocolo": "ensemble de 5 folds estratificados em treino+validacao",
        "representacao": args.representacao,
        "seeds": [84, 85, 86, 87, 88],
        "limiar_oof": limiar,
        "metricas_oof": metricas_oof,
        "metricas_teste": metricas_teste,
        "configuracao": asdict(configuracao),
    }
    (saida / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Ensemble concluido: AUC OOF={metricas_oof['auc_roc']:.4f}; "
        f"AUC teste={metricas_teste['auc_roc']:.4f}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
