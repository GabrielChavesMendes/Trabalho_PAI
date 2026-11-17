import pytest
import torch
from torch import nn

from modulos.modulo_redes import CNNSimples
from modulos.rodada_final_issue4 import (
    CNNLeaky, criar_carregador, criar_modelo_registrado, carregar_modelo_registrado,
    normalizar_adicional, DatasetNormalizado,
)


def test_leaky_preserva_parametros_mas_evitaria_carga_silenciosa_na_relu(tmp_path):
    torch.manual_seed(42)
    original = CNNSimples(1)
    torch.manual_seed(42)
    novo = CNNLeaky(1)
    assert all(torch.equal(p,q) for p,q in zip(original.parameters(),novo.parameters()))
    assert sum(isinstance(m,nn.LeakyReLU) for m in novo.modules()) == 5
    assert not any(isinstance(m,nn.ReLU) for m in novo.modules())
    with pytest.raises(RuntimeError):
        original.load_state_dict(novo.state_dict(),strict=True)
    x = torch.randn(2,1,32,32)
    novo.eval()
    esperado = novo(x).detach()
    cp = tmp_path / 'leaky.pt'
    torch.save({'formato_checkpoint':1,'configuracao':{'arquitetura':'cnn_leaky','canais_entrada':1},
                'preprocessamento':{'normalizacao_adicional':'nenhuma'},'model_state_dict':novo.state_dict()},cp)
    recarregado, _, norm = carregar_modelo_registrado(cp)
    assert norm == 'nenhuma'
    assert torch.equal(recarregado(x),esperado)


def test_leaky_deixa_gradiente_no_ramo_negativo():
    ativacao = CNNLeaky(1).caracteristicas[1]
    x = torch.tensor([-2.,2.],requires_grad=True)
    ativacao(x).sum().backward()
    assert x.grad.tolist() == pytest.approx([.01,1.])


def test_padronizacao_imagenet_preserva_imagem_original_e_mascara():
    x = torch.zeros(3,4,4)
    y = normalizar_adicional(x,'imagenet_rgb')
    assert torch.allclose(y[:,0,0],torch.tensor([-.485/.229,-.456/.224,-.406/.225]))
    assert torch.equal(x,torch.zeros_like(x))
    class Base:
        def __len__(self): return 1
        def __getitem__(self,i): return {'imagem':x,'mascara':torch.ones(1,4,4),'rotulo':torch.tensor(1.)}
    ds = DatasetNormalizado(Base(),'imagenet_rgb')
    assert torch.equal(ds[0]['mascara'],torch.ones(1,4,4))
    assert len(ds)==1
    assert normalizar_adicional(x,'nenhuma') is x


@pytest.mark.parametrize('norm', ['desconhecida','imagenet_rgb'])
def test_recusa_normalizacao_invalida(norm):
    with pytest.raises(ValueError):
        normalizar_adicional(torch.zeros(1,8,8),norm)


def test_teste_e_bloqueado_antes_de_ler_csv_ou_imagens():
    with pytest.raises(ValueError,match='bloqueado'):
        criar_carregador('teste',{})


def test_pipeline_antigo_continua_criando_relu():
    modelo = criar_modelo_registrado('cnn',1)
    assert type(modelo) is CNNSimples
    assert sum(isinstance(m,nn.ReLU) for m in modelo.modules())==5
