import json

import pandas as pd
import pytest

from scripts.relatorio_final_issue4 import ranking, auditar_predicoes


def test_ranking_teste_prioriza_ba_depois_f1_sem_mudar_a_selecao():
    df = pd.DataFrame([
        dict(id='vencedor_validacao',acuracia_balanceada=.6,f1=.5,auc_roc=.9,representante=True),
        dict(id='observacao_teste',acuracia_balanceada=.7,f1=.4,auc_roc=.5,representante=False),
        dict(id='desempate_f1',acuracia_balanceada=.7,f1=.5,auc_roc=.4,representante=True)])
    antes = df.copy(deep=True)
    r = ranking(df,'acuracia_balanceada')
    assert r.id.tolist()==['desempate_f1','observacao_teste','vencedor_validacao']
    assert ranking(df,'auc_roc').iloc[0].id=='vencedor_validacao'
    pd.testing.assert_frame_equal(df,antes)


def test_auditoria_independente_rejeita_classe_ou_metrica_incorreta(tmp_path):
    csv = pd.DataFrame(dict(pid=['n1','n2','p1','p2'],pCR=[0,0,1,1],test=[1]*4))
    pred = [dict(paciente=id,rotulo=y,probabilidade=p,previsto=int(p>=.5))
            for id,y,p in zip(csv.pid,csv.pCR,[.1,.6,.4,.8])]
    metricas = dict(sensibilidade=.5,especificidade=.5,precisao=.5,acuracia=.5,f1=.5,
                    acuracia_balanceada=.5,auc_roc=.75,tn=1,fp=1,fn=1,tp=1,numero_pacientes=4,limiar=.5)
    arquivo = tmp_path/'resultados.json'
    def salvar(): arquivo.write_text(json.dumps(dict(metricas=metricas,predicoes_por_paciente=pred)),encoding='utf-8')
    salvar()
    auditar_predicoes(arquivo,csv,1,.5)
    metricas['auc_roc']=.8
    salvar()
    with pytest.raises(ValueError,match='sklearn'):
        auditar_predicoes(arquivo,csv,1,.5)
    metricas['auc_roc']=.75
    pred[0]['previsto']=1
    salvar()
    with pytest.raises(ValueError,match='Classe'):
        auditar_predicoes(arquivo,csv,1,.5)
