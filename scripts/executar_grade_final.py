"""Executa exatamente os 12 casos pré-registrados, apenas treino/validação."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts.executar_rodada_final_issue4 import executar_caso
from modulos.modulo_treinamento import escolher_dispositivo


def validar_plano(plano):
    casos = plano['casos']
    esperadas = {(a, f, c) for a in ('cnn_groupnorm', 'resnet50', 'mobilenetv2')
                 for f in ('post_early', 'tres') for c in ('central', 'maior_tumor')}
    if len(casos) != 12 or len({c['id'] for c in casos}) != 12:
        raise ValueError('Exatamente 12 casos únicos são obrigatórios')
    if {(c['arquitetura'], c['fase'], c['corte']) for c in casos} != esperadas:
        raise ValueError('Grade incompleta ou duplicada')
    for arquitetura in ('cnn_groupnorm', 'resnet50', 'mobilenetv2'):
        parametros = [{k: v for k, v in c.items() if k not in ('id', 'fase', 'corte', 'comparador')}
                      for c in casos if c['arquitetura'] == arquitetura]
        if any(p != parametros[0] for p in parametros):
            raise ValueError('A intervenção deve ser a mesma nas quatro configurações')
    if plano['seed'] != 42 or plano['batch'] != 4 or plano['epocas'] != 30 or plano['paciencia'] != 5:
        raise ValueError('Condições do protocolo divergentes')
    if plano['teste_autorizado'] is not False:
        raise ValueError('Script de treino não autoriza leitura de teste')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    arquivo = a.base / 'plano_antes_treinamento.json'
    plano = json.loads(arquivo.read_text(encoding='utf-8'))
    validar_plano(plano)
    digest = hashlib.sha256(arquivo.read_bytes()).hexdigest()
    for caminho, esperado in plano['fontes_sha256'].items():
        if hashlib.sha256((RAIZ / caminho).read_bytes()).hexdigest() != esperado:
            raise ValueError(f'Fonte alterada desde o planejamento: {caminho}')
    device = escolher_dispositivo(a.device)
    for i, caso in enumerate(plano['casos'], 1):
        concluido = a.base / caso['id'] / 'concluido.json'
        if concluido.exists():
            d = json.loads(concluido.read_text(encoding='utf-8'))
            cp = Path(d['checkpoint'])
            if d['plano_sha256'] != digest or hashlib.sha256(cp.read_bytes()).hexdigest() != d['checkpoint_sha256']:
                raise ValueError('Resultado concluído diverge do plano/checkpoint')
            print('Preservado, sem retreinar:', caso['id'], flush=True)
            continue
        print(f'[{i}/12] Iniciando {caso["id"]}', flush=True)
        executar_caso(caso, plano, a.base, a.checkpoints, device, digest)


if __name__ == '__main__':
    main()
