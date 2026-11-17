"""Regressões dos comandos sem treino e do preparo compartilhado."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modulos import checkpoints_integrados_issue4 as integrados
from modulos.execucao_issue4 import escolher_device
from modulos.modulo_redes import criar_modelo
from scripts.inferir_paciente_issue4 import modelos_selecionados
from scripts.verificar_issue4 import executar_etapa
from tests.test_checkpoints_integrados_issue4 import _checkpoint_minimo


def test_recarga_segura_dos_pesos_proprios_no_mesmo_adaptador(tmp_path):
    ck = _checkpoint_minimo()
    ck['formato_checkpoint'] = 1
    ck['configuracao']['arquitetura'] = 'cnn'
    ck['preprocessamento']['normalizacao'] = 'percentis 1/99 conjuntos por amostra; clip [0,1]'
    ck['model_state_dict'] = criar_modelo('cnn',1,pre_treinado=False).state_dict()
    arquivo = tmp_path/'proprio.pt'
    torch.save(ck,arquivo)
    m,d = integrados.carregar_modelo_checkpoint(arquivo)
    assert not m.training and integrados.normalizacao_checkpoint(d)=='percentil'
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in ck['model_state_dict'].items())
    ck['preprocessamento']['normalizacao_adicional'] = 'imagenet_rgb'
    with pytest.raises(ValueError,match='Normalizacao'):
        integrados.validar_configuracao(ck)


def dados_com_aumento():
    ck = _checkpoint_minimo()
    ck['preprocessamento']['augmentation'] = dict(angulos_graus=[-20,-10,0,10,20],
        espelhamento=[False,True],variacoes_por_paciente=10)
    return ck


class BaseSintetica:
    def __init__(self,pacientes,**kwargs): self.pacientes=pacientes
    def __len__(self): return len(self.pacientes)
    def __getitem__(self,i):
        return dict(imagem=torch.full((1,32,32),.5),mascara=torch.ones(1,32,32),
                    paciente=self.pacientes[i].identificador,rotulo=torch.tensor(1.),corte_z=3)


def test_padronizacao_ocorre_antes_da_rotacao_da_validacao(monkeypatch):
    from modulos import modulo_dataset_issue1 as aumento
    monkeypatch.setattr(integrados,'DatasetDCE',BaseSintetica)
    transformar_original=aumento.transformar_par
    chamadas=[]
    def verificar_entrada(imagem,mascara,angulo,espelhada):
        assert float(imagem[0,0,0])==pytest.approx((.5-.449)/.226,abs=1e-6)
        chamadas.append((angulo,espelhada))
        return transformar_original(imagem,mascara,angulo,espelhada)
    monkeypatch.setattr(aumento,'transformar_par',verificar_entrada)
    pacientes=[SimpleNamespace(identificador='P1',particao_codigo=2)]
    ds=integrados.dataset_checkpoint(pacientes,dados_com_aumento(),aumentar_validacao=True)
    assert len(ds)==10
    amostra=ds[0]
    assert chamadas==[(-20,False)]
    assert float(amostra['imagem'][0,16,16])==pytest.approx((.5-.449)/.226,abs=1e-6)


def test_inferencia_usa_limiar_salvo_sem_gradiente_ou_aumento(monkeypatch):
    monkeypatch.setattr(integrados,'DatasetDCE',BaseSintetica)
    class Modelo(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit=nn.Parameter(torch.logit(torch.tensor(.4)))
        def forward(self,x):
            assert not self.training and not torch.is_grad_enabled()
            assert x.shape==(1,1,32,32)
            assert float(x[0,0,0,0])==pytest.approx((.5-.449)/.226,abs=1e-6)
            return self.logit.reshape(1,1)
    modelo=Modelo()
    antes=modelo.logit.detach().clone()
    r=integrados.prever_paciente(modelo,dados_com_aumento(),SimpleNamespace(identificador='P1'),'cpu')
    assert r['probabilidade_pcr']==pytest.approx(.4) and r['classificacao_pcr']==1
    assert r['limiar']==.25 and r['aumento'] is False and r['corte_z']==3
    assert torch.equal(antes,modelo.logit.detach()) and modelo.logit.grad is None


def test_catalogo_preserva_as_doze_combinacoes():
    modelos=modelos_selecionados()
    assert len(modelos)==12
    assert {a:sum(m['arquitetura']==a for m in modelos) for a in ('cnn','resnet50','mobilenetv2')}=={
        'cnn':4,'resnet50':4,'mobilenetv2':4}
    assert sum(m['origem']=='Gustavo' for m in modelos)==4


def test_device_auto_cpu_cuda_e_erro_explicito(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert escolher_device()=='cpu'
    with pytest.raises(RuntimeError,match='indisponivel'): escolher_device('cuda')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    assert escolher_device()=='cuda' and escolher_device('cpu')=='cpu'


def test_comando_unico_propaga_falha_e_guarda_log(tmp_path):
    import os
    log=tmp_path/'falha.txt'
    with pytest.raises(RuntimeError,match='codigo 3'):
        executar_etapa('Etapa sintetica',['-c','import sys; print("causa concreta"); sys.exit(3)'],log,os.environ)
    assert 'causa concreta' in log.read_text(encoding='utf-8')


def test_verificacao_nao_sobrescreve_saida_anterior(tmp_path,monkeypatch):
    from scripts import verificar_checkpoints_integrados_issue4 as script
    arquivo=tmp_path/'protegido.txt'
    arquivo.write_text('preservar',encoding='utf-8')
    monkeypatch.setattr('sys.argv',['verificar','--saida',str(tmp_path)])
    with pytest.raises(FileExistsError): script.main()
    assert arquivo.read_text(encoding='utf-8')=='preservar'
