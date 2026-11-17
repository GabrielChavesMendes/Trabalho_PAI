import pytest

from modulos.modulo_treinamento import RegistroPredicao
from modulos.analise_validacao_issue4 import (
    distribuicao_probabilidades, estabilidade_limiar, diferenca_auc_bootstrap,
)
from scripts import avaliar_manifesto_issue4


def test_bootstrap_pareado_identico_tem_delta_e_intervalo_zero():
    ps = [RegistroPredicao('a',0,.1),RegistroPredicao('b',1,.8)]
    d = diferenca_auc_bootstrap(ps,ps,n=20)
    assert d['delta_auc']==d['ic95_percentil_inferior']==d['ic95_percentil_superior']==0


def test_bootstrap_distingue_ordenacao_perfeita_de_invertida():
    a = [RegistroPredicao('n',0,.9),RegistroPredicao('p',1,.1)]
    b = [RegistroPredicao('n',0,.1),RegistroPredicao('p',1,.9)]
    d = diferenca_auc_bootstrap(a,b,n=20)
    assert d['delta_auc']==d['ic95_percentil_inferior']==d['ic95_percentil_superior']==1
    with pytest.raises(ValueError,match='mesmos pacientes'):
        diferenca_auc_bootstrap(a,[RegistroPredicao('outro',1,.9)])


def test_escore_constante_e_fragilidade_do_limiar_sao_visiveis():
    ps = [RegistroPredicao('n',0,.5),RegistroPredicao('p',1,.5)]
    assert distribuicao_probabilidades(ps)['todos']['std']==0
    d = estabilidade_limiar(ps,.5)
    assert d['pior_acuracia_balanceada_mais_menos_001']==.5
    assert d['avaliacoes'][-1]['classes_alteradas']==2


def test_avaliador_nao_abre_manifesto_ou_teste_sem_confirmacao(monkeypatch,tmp_path):
    monkeypatch.setattr('sys.argv',['avaliar_manifesto_issue4.py','--manifesto',str(tmp_path/'inexistente.json'),
                                    '--saida',str(tmp_path/'saida')])
    with pytest.raises(SystemExit) as exc:
        avaliar_manifesto_issue4.main()
    assert exc.value.code==2
    assert not (tmp_path/'saida').exists()


def test_manifesto_incompleto_e_rejeitado_antes_do_teste():
    with pytest.raises(ValueError,match='12 configurações'):
        avaliar_manifesto_issue4.validar_manifesto({'teste_usado_para_selecao':False,'modelos':[]})


def test_pesos_modificados_sao_rejeitados_antes_de_carregar_modelo(tmp_path,monkeypatch):
    cp=tmp_path/'alterado.pt'
    cp.write_bytes(b'pesos-alterados-depois-do-congelamento')
    entradas=[{'id':f'{a}-{f}-{c}','grupo':'originais','checkpoint':str(cp),
               'checkpoint_sha256':'0'*64,
               'configuracao':{'arquitetura':a,'configuracao_temporal':f,'estrategia_corte':c}}
              for a in ('cnn','resnet50','mobilenetv2')
              for f in ('post_early','tres') for c in ('central','maior_tumor')]
    def proibido(*args,**kwargs):
        raise AssertionError('Modelo não pode carregar antes de validar seu hash')
    monkeypatch.setattr(avaliar_manifesto_issue4,'carregar_modelo_registrado',proibido)
    with pytest.raises(ValueError,match='Checkpoint alterado'):
        avaliar_manifesto_issue4.validar_manifesto({'teste_usado_para_selecao':False,'modelos':entradas})


def test_saida_existente_impede_repetir_avaliacao_final(tmp_path,monkeypatch):
    saida=tmp_path/'resultado-anterior'
    saida.mkdir()
    registro=saida/'protegido.json'
    registro.write_text('preservar',encoding='utf-8')
    monkeypatch.setattr('sys.argv',['avaliar_manifesto_issue4.py','--manifesto',str(tmp_path/'inexistente.json'),
                                    '--saida',str(saida),'--confirmar-validacao-fechada'])
    with pytest.raises(FileExistsError):
        avaliar_manifesto_issue4.main()
    assert registro.read_text(encoding='utf-8')=='preservar'
