"""Recarga e inferência de validação dos três novos pesos, sem treinamento/teste."""
import argparse
import json
import sys
from pathlib import Path

import torch

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from modulos.rodada_final_issue4 import carregar_modelo_registrado, criar_carregador
from modulos.modulo_treinamento import configurar_reprodutibilidade, avaliar_modelo, salvar_resultado_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    pasta=args.base/'recarga_verificada'
    if pasta.exists(): raise FileExistsError(pasta)
    plano=json.loads((args.base/'plano_antes_treinamento.json').read_text(encoding='utf-8'))
    pasta.mkdir()
    resultados=[]
    for caso in plano['casos']:
        dados=json.loads((args.base/caso['id']/'concluido.json').read_text(encoding='utf-8'))
        configurar_reprodutibilidade(dados['configuracao']['seed'])
        modelo,bruto,norm=carregar_modelo_registrado(dados['checkpoint'],'cuda')
        loader=criar_carregador('validacao',bruto['configuracao'],norm)
        criterio=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.],device='cuda'))
        limiar=bruto['selecao_validacao']['limiar']
        loss,metricas,ps=avaliar_modelo(modelo,loader,criterio,torch.device('cuda'),limiar)
        anteriores=json.loads(Path(dados['validacao']).read_text(encoding='utf-8'))
        por_id={p['paciente']:p for p in anteriores['predicoes_por_paciente']}
        if set(por_id)!={p.paciente for p in ps}:
            raise ValueError('Pacientes divergentes')
        maximo=max(abs(p.probabilidade-por_id[p.paciente]['probabilidade']) for p in ps)
        classes=sum(int(p.probabilidade>=limiar)!=por_id[p.paciente]['previsto'] for p in ps)
        salvar_resultado_json(pasta/(caso['id']+'__validation_reloaded.json'),{**metricas,'loss':loss},ps)
        resultados.append({'id':caso['id'],'normalizacao':norm,'diferenca_max_prob':maximo,
                            'classes_divergentes':classes,'pacientes':len(ps),'teste_avaliado':False})
        print(resultados[-1],flush=True)
        del modelo,bruto,loader
        torch.cuda.empty_cache()
    (pasta/'relatorio.json').write_text(json.dumps(resultados,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(d['classes_divergentes'] or d['diferenca_max_prob']>1e-6 for d in resultados):
        raise ValueError('Recarga divergente; revisar antes de qualquer avaliação final')


if __name__=='__main__': main()
