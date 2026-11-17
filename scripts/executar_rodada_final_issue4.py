"""Executa apenas os três casos previamente registrados; nunca avalia teste."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.rodada_final_issue4 import (
    criar_carregador, criar_modelo_registrado, carregar_modelo_registrado, metadados_rodada,
)
from modulos.modulo_treinamento import (
    ConfiguracaoTreinamento, configurar_reprodutibilidade, escolher_dispositivo,
    treinar_experimento, avaliar_modelo, registrar_validacao_checkpoint,
    salvar_historico_csv, salvar_grafico_convergencia, salvar_resultado_json,
    salvar_matriz_confusao, selecionar_limiar_validacao,
)
from scripts.refinar_limiares_validacao_issue4 import candidatos_por_escores


def nome_caso(caso, seed=42):
    return (f"{caso['id']}__{caso['arquitetura'].replace('_','-')}__"
            f"{caso['fase'].replace('_','-')}__{caso['corte'].replace('_','-')}__seed-{seed}")


def executar_caso(caso, plano, base, checkpoints, device, plano_hash):
    pasta = base / caso['id']
    nome = nome_caso(caso, plano['seed'])
    cp = checkpoints / (nome + '.pt')
    if pasta.exists() or cp.exists():
        raise FileExistsError(f'Execução anterior/parcial preservada: {pasta} | {cp}')
    pasta.mkdir(parents=True)
    conf = ConfiguracaoTreinamento(
        arquitetura=caso['arquitetura'], canais_entrada=3 if caso['fase']=='tres' else 1,
        configuracao_temporal=caso['fase'], estrategia_corte=caso['corte'],
        epocas=plano['epocas'], tamanho_lote=plano['batch'], taxa_aprendizado=caso['lr'],
        decaimento_pesos=plano['weight_decay'], pos_weight=plano['pos_weight'],
        paciencia=plano['paciencia'], seed=plano['seed'], criterio_checkpoint='auc_roc',
    )
    meta = metadados_rodada(caso)
    meta['plano_antes_treinamento_sha256'] = plano_hash
    (pasta / 'configuracao_antes_treino.json').write_text(
        json.dumps({'configuracao': asdict(conf), 'preprocessamento': meta}, indent=2), encoding='utf-8')
    treino = criar_carregador('treino', asdict(conf), caso['normalizacao_adicional'])
    validacao = criar_carregador('validacao', asdict(conf), caso['normalizacao_adicional'])
    configurar_reprodutibilidade(conf.seed)
    modelo = criar_modelo_registrado(conf.arquitetura, conf.canais_entrada,
                                    pre_treinado=conf.arquitetura == 'resnet50')
    historico = treinar_experimento(modelo, treino, validacao, conf, cp, meta, device)
    salvar_historico_csv(pasta / (nome + '__history.csv'), historico)
    salvar_grafico_convergencia(pasta / (nome + '__loss.png'), historico)
    del modelo
    modelo, bruto, norm = carregar_modelo_registrado(cp, device)
    criterio = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([conf.pos_weight], device=device))
    loss, _, ps = avaliar_modelo(modelo, validacao, criterio, device)
    candidatos = candidatos_por_escores(p.probabilidade for p in ps)
    limiar, metricas = selecionar_limiar_validacao(ps, candidatos, 'acuracia_balanceada')
    metricas = {**metricas, 'loss': loss}
    if metricas['numero_pacientes'] != 33:
        raise ValueError('Esperados 33 pacientes de validação')
    salvar_resultado_json(pasta / (nome + '__validation.json'), metricas, ps)
    salvar_matriz_confusao(pasta / (nome + '__validation__confusion-matrix.png'), metricas)
    registrar_validacao_checkpoint(cp, limiar, {**metricas, 'criterio_selecao_limiar':'acuracia_balanceada',
                                              'metodo_busca':'pontos_medios_escores_validacao'})
    loss_treino, metricas_treino, ps_treino = avaliar_modelo(modelo, treino, criterio, device, limiar)
    salvar_resultado_json(pasta / (nome + '__train.json'), {**metricas_treino,'loss':loss_treino}, ps_treino)
    resumo = {'id': caso['id'], 'nome': nome, 'checkpoint': str(cp),
              'checkpoint_sha256': hashlib.sha256(cp.read_bytes()).hexdigest(),
              'validacao': str(pasta / (nome + '__validation.json')),
              'historico': str(pasta / (nome + '__history.csv')),
              'configuracao': asdict(conf), 'preprocessamento': meta,
              'epoca_escolhida': bruto['epoca'], 'epocas_executadas':len(historico),
              'tempo_treinamento_segundos':sum(h.tempo_segundos for h in historico),
              'metricas':metricas, 'metricas_treino':metricas_treino,
              'plano_sha256':plano_hash, 'normalizacao_recarga':norm, 'teste_avaliado':False}
    (pasta / 'concluido.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"CONCLUÍDO {caso['id']}: AUC={metricas['auc_roc']:.4f}; BA={metricas['acuracia_balanceada']:.4f}; F1={metricas['f1']:.4f}",flush=True)
    del modelo, bruto, treino, validacao
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args = parser.parse_args()
    plano_arquivo = args.base / 'plano_antes_treinamento.json'
    plano_hash = hashlib.sha256(plano_arquivo.read_bytes()).hexdigest()
    plano = json.loads(plano_arquivo.read_text(encoding='utf-8'))
    if len(plano['casos']) != 3 or len({c['id'] for c in plano['casos']}) != 3:
        raise ValueError('Esta rodada autoriza exatamente três casos únicos')
    if plano.get('teste_autorizado'):
        raise ValueError('Este script é exclusivo de treino/validação')
    device = escolher_dispositivo(args.device)
    for numero, caso in enumerate(plano['casos'],1):
        concluido = args.base / caso['id'] / 'concluido.json'
        if concluido.is_file():
            dados = json.loads(concluido.read_text(encoding='utf-8'))
            if dados['plano_sha256'] != plano_hash or not Path(dados['checkpoint']).is_file():
                raise ValueError('Artefato concluído diverge do plano; não sobrescrever')
            print('Já concluído e preservado:',caso['id'],flush=True)
            continue
        print(f'[{numero}/3] Iniciando {caso["id"]}',flush=True)
        executar_caso(caso,plano,args.base,args.checkpoints,device,plano_hash)


if __name__ == '__main__':
    main()
