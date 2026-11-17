"""Reavalia apenas a validação do checkpoint escolhido, sem treinamento."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modulos.fabrica_carregadores import criar_carregadores
from modulos.modulo_redes import criar_modelo
from modulos.modulo_treinamento import (
    avaliar_modelo, carregar_checkpoint, configurar_reprodutibilidade,
    escolher_dispositivo, salvar_resultado_json,
)

parser = argparse.ArgumentParser()
parser.add_argument('--resultados', type=Path, required=True)
parser.add_argument('--checkpoints', type=Path, required=True)
parser.add_argument('--saida', type=Path, required=True)
args = parser.parse_args()
if args.saida.exists():
    raise FileExistsError(args.saida)
selecao = json.loads((args.resultados / 'modelo_selecionado.json').read_text(encoding='utf-8'))
nome = selecao['experimento']
caminho = args.checkpoints / (nome + '.pt')
bruto = torch.load(caminho, map_location='cpu', weights_only=False)
conf = bruto['configuracao']
configurar_reprodutibilidade(conf['seed'])
device = escolher_dispositivo('cuda')
modelo = criar_modelo(conf['arquitetura'], conf['canais_entrada'], pre_treinado=False)
carregar_checkpoint(caminho, modelo, 'cpu')
modelo.to(device)
carregadores = criar_carregadores(conf['configuracao_temporal'], conf['estrategia_corte'],
                                  conf['tamanho_lote'], conf['seed'])
criterio = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([conf['pos_weight']], device=device))
limiar = bruto['selecao_validacao']['limiar']
loss, metricas, predicoes = avaliar_modelo(modelo, carregadores['validacao'], criterio, device, limiar)
originais = json.loads((args.resultados / (nome + '__validation.json')).read_text(encoding='utf-8'))
por_id = {p['paciente']: p for p in originais['predicoes_por_paciente']}
if {p.paciente for p in predicoes} != set(por_id):
    raise ValueError('IDs divergentes após recarga')
erro_max = max(abs(p.probabilidade - por_id[p.paciente]['probabilidade']) for p in predicoes)
divergencias = sum(int(p.probabilidade >= limiar) != por_id[p.paciente]['previsto'] for p in predicoes)
args.saida.mkdir(parents=True)
salvar_resultado_json(args.saida / 'predicoes_validacao_recarregada.json', {**metricas, 'loss': loss}, predicoes)
relatorio = {
    'experimento': nome, 'checkpoint': str(caminho),
    'checkpoint_sha256': hashlib.sha256(caminho.read_bytes()).hexdigest(),
    'device': str(device), 'limiar': limiar, 'pacientes': len(predicoes),
    'maior_diferenca_absoluta_probabilidade': erro_max,
    'classificacoes_divergentes': divergencias,
    'menor_distancia_original_ao_limiar': min(abs(p['probabilidade'] - limiar) for p in por_id.values()),
    'treinamento_executado': False, 'teste_avaliado': False,
    'metricas': {**metricas, 'loss': loss},
    'limite': 'Verifica repetibilidade na mesma GPU/versoes/batch; não garante estabilidade entre ambientes ou generalizacao.',
}
(args.saida / 'relatorio.json').write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(relatorio, ensure_ascii=False, indent=2))
if divergencias or erro_max > 1e-6:
    raise SystemExit('Recarga com divergência: revisar antes de qualquer teste final.')
