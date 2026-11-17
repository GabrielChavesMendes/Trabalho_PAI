"""Registra o protocolo e os hashes ANTES dos 12 treinamentos autorizados."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import pandas as pd
import torch
from modulos.modulo_dataset import carregar_metadados


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for bloco in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    a = p.parse_args()
    if a.base.exists():
        raise FileExistsError('Diretório já existe; preservar experimentos')
    csv = RAIZ / 'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv'
    tabela = carregar_metadados(csv)
    contagens = {int(k): int(v) for k, v in tabela.test.value_counts().items()}
    if contagens != {0: 104, 2: 33, 1: 35} or tabela.pid.duplicated().any():
        raise ValueError('Partições oficiais inválidas')
    # Apenas metadados de partição; nenhuma imagem/predição de teste é acessada.
    casos = []
    intervencoes = {
        'cnn_groupnorm': ('cnn', 1e-3, None, 'GroupNorm(4) entre cada Conv e ReLU'),
        'resnet50': ('resnet50', 1e-4, 1e-5, 'layer4+fc; todas BN affine/estatisticas congeladas'),
        'mobilenetv2': ('mobilenetv2', 1e-4, 1e-5, 'features[17:19]+classifier; todas BN congeladas'),
    }
    for arquitetura, (familia, lr, lrb, intervencao) in intervencoes.items():
        for fase in ('post_early', 'tres'):
            for corte in ('central', 'maior_tumor'):
                casos.append(dict(id=f'c_{familia}_{fase}_{corte}', arquitetura=arquitetura,
                                  fase=fase, corte=corte, lr=lr, lr_backbone=lrb,
                                  intervencao=intervencao, normalizacao_adicional='nenhuma', grade_final=True,
                                  comparador=f'{familia}__{fase.replace("_", "-")}__{corte.replace("_", "-")}__seed-42'))
    fontes = ['modulos/modulo_treinamento.py', 'modulos/modulo_redes.py', 'modulos/modulo_dataset.py',
              'modulos/modulo_dataset_issue1.py', 'modulos/rodada_final_issue4.py', 'modulos/grade_final_issue4.py',
              'scripts/executar_rodada_final_issue4.py', 'scripts/executar_ultima_grade_issue4.py',
              'scripts/refinar_limiares_validacao_issue4.py']
    plano = dict(registrado_em_utc=datetime.now(timezone.utc).isoformat(),
                 finalidade='Última rodada autorizada: exatamente 12 treinamentos; sem busca posterior',
                 seed=42, batch=4, epocas=30, paciencia=5, pos_weight=3., weight_decay=1e-4,
                 otimizador='Adam; sem scheduler; FP32', teste_autorizado=False,
                 avaliacao_final_autorizada_pelo_usuario=True,
                 checkpoint='maior AUC validacao; empate menor loss',
                 limiar='BA validacao; pontos medios de escores; empate especificidade, sensibilidade, proximidade .5',
                 selecao='AUC decrescente, BA decrescente, pior BA limiar ±.01 decrescente, sensibilidade decrescente, '
                         'F1 decrescente, loss crescente, ID lexicografico; empates numericos exatos',
                 interpretacao='Deltas pequenos nao provam superioridade. Nenhum descarte ou troca por teste.',
                 casos=casos, csv_sha256=sha(csv), particoes=contagens,
                 fontes_sha256={s: sha(RAIZ/s) for s in fontes},
                 ambiente=dict(torch=torch.__version__, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    antigos = {}
    for raiz in ['resultados/execucao_issue4_20261117_01', 'resultados/rodada_final_issue4_20261117_01',
                 'checkpoints/execucao_issue4_20261117_01', 'checkpoints/rodada_final_issue4_20261117_01']:
        for arquivo in sorted((RAIZ/raiz).rglob('*')):
            if arquivo.is_file():
                antigos[arquivo.relative_to(RAIZ).as_posix()] = sha(arquivo)
    a.base.mkdir(parents=True)
    for nome, dados in [('plano_antes_treinamento.json', plano), ('preservacao_antes.json', antigos)]:
        with (a.base/nome).open('x', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    print('Plano registrado:', len(casos), 'casos;', len(antigos), 'arquivos anteriores protegidos por hash')


if __name__ == '__main__':
    main()
