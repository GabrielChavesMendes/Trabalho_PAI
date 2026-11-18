import pytest
import torch
from torch import nn

from modulos.grade_experimental import CNNGroupNorm, preparar_fine_tuning, selecionar_representantes
from modulos.modulo_redes import CNNSimples
from modulos.modulo_treinamento import (
    ConfiguracaoTreinamento, grupos_parametros_otimizador, _batchnorm_congelado_em_eval,
)
from modulos.variantes_experimentais import criar_modelo_registrado, carregar_modelo_registrado


def test_groupnorm_inicializacao_conv_linear_e_recarga(tmp_path):
    torch.manual_seed(42)
    antiga = CNNSimples(1)
    torch.manual_seed(42)
    nova = CNNGroupNorm(1)
    camadas = lambda m: [x for x in m.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
    assert len(camadas(nova)) == 6
    for a, b in zip(camadas(antiga), camadas(nova), strict=True):
        assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
    assert sum(isinstance(x, nn.GroupNorm) for x in nova.modules()) == 4
    x = torch.randn(2, 1, 32, 32)
    nova.eval()
    y = nova(x).detach()
    # GN não depende das outras pacientes do lote.
    assert torch.allclose(nova(x[:1]), y[:1], atol=1e-7)
    cp = tmp_path / 'gn.pt'
    torch.save({'formato_checkpoint': 1, 'configuracao': {'arquitetura': 'cnn_groupnorm', 'canais_entrada': 1},
                'preprocessamento': {}, 'model_state_dict': nova.state_dict()}, cp)
    m, _, _ = carregar_modelo_registrado(cp)
    assert torch.equal(m(x), y)
    with pytest.raises(RuntimeError):
        antiga.load_state_dict(nova.state_dict(), strict=True)


@pytest.mark.parametrize('arquitetura', ['mobilenetv2', 'resnet50'])
def test_finetuning_gradientes_grupos_lr_e_bn_imoveis(arquitetura):
    m = criar_modelo_registrado(arquitetura, 1, pre_treinado=False)
    preparar_fine_tuning(m, arquitetura)
    conf = ConfiguracaoTreinamento(arquitetura=arquitetura, canais_entrada=1,
                                  configuracao_temporal='post_early', estrategia_corte='central', taxa_aprendizado=1e-4,
                                  taxa_aprendizado_backbone=1e-5)
    grupos = grupos_parametros_otimizador(m, conf)
    assert [g['lr'] for g in grupos] == [1e-5, 1e-4]
    assert {id(p) for g in grupos for p in g['params']} == {id(p) for p in m.parameters() if p.requires_grad}
    assert len([p for g in grupos for p in g['params']]) == len({id(p) for g in grupos for p in g['params']})
    bn = [(b, b.running_mean.clone(), b.running_var.clone(), b.num_batches_tracked.clone())
          for b in m.modules() if isinstance(b, nn.BatchNorm2d)]
    m.train()
    _batchnorm_congelado_em_eval(m)
    m(torch.randn(2, 1, 32, 32)).sum().backward()
    for b, mean, var, contagem in bn:
        assert not b.training and torch.equal(b.running_mean, mean) and torch.equal(b.running_var, var)
        assert torch.equal(b.num_batches_tracked, contagem)
        assert all(p.grad is None for p in b.parameters())
    for nome, p in m.named_parameters():
        permitido = (nome.startswith(('layer4.', 'fc.')) if arquitetura == 'resnet50'
                     else nome.startswith(('features.17.', 'features.18.', 'classifier.')))
        if p.requires_grad:
            assert permitido and p.grad is not None and torch.isfinite(p.grad).all()
        else:
            assert p.grad is None


def test_representantes_sao_unicos_e_apenas_pela_validacao():
    linhas = []
    for a in ('cnn', 'resnet50', 'mobilenetv2'):
        for f in ('post_early', 'tres'):
            for c in ('central', 'maior_tumor'):
                for grupo, auc in [('A', .6), ('C', .7)]:
                    linhas.append(dict(id=f'{a}_{f}_{c}_{grupo}', auc_roc=auc, acuracia_balanceada=.65,
                                       sensibilidade=.5, f1=.5, loss=1.,
                                       estabilidade={'pior_acuracia_balanceada_mais_menos_001': .6},
                                       configuracao=dict(arquitetura=a, configuracao_temporal=f, estrategia_corte=c)))
    reps, melhores, geral = selecionar_representantes(linhas)
    assert len(set(reps)) == 12 and all(r.endswith('_C') for r in reps)
    assert set(melhores) == {'cnn', 'resnet50', 'mobilenetv2'} and geral in reps
    # Empate de AUC: BA prefere evitar ignorar uma classe, sem olhar a acurácia.
    linhas[0].update(auc_roc=.7, acuracia_balanceada=.75, acuracia=.1)
    assert linhas[0]['id'] in selecionar_representantes(linhas)[0]
    with pytest.raises(ValueError):
        selecionar_representantes(linhas[2:])


def coorte_sintetica():
    entradas = []
    for grupo in ('A', 'C'):
        for a in ('cnn', 'resnet50', 'mobilenetv2'):
            for f in ('post_early', 'tres'):
                for c in ('central', 'maior_tumor'):
                    entradas.append(dict(id=f'{grupo}-{a}-{f}-{c}', grupo=grupo,
                                         checkpoint=f'{grupo}-{a}-{f}-{c}.pt',
                                         configuracao=dict(arquitetura=a,configuracao_temporal=f,estrategia_corte=c)))
    entradas.extend(dict(entradas[i],id=id,grupo='B',checkpoint=f'B-{i}.pt') for i,id in
                    enumerate(('e1_cnn_lr_menor','e2_cnn_leaky','e3_resnet_imagenet')))
    return dict(modelos=entradas, representantes=[m['id'] for m in entradas if m['grupo']=='A'])


def test_coorte_27_rejeita_quantidade_e_representantes_duplicados_antes_de_abrir_pesos():
    from scripts.avaliar_manifesto import validar_coorte_27
    m = coorte_sintetica()
    m['modelos'].pop()
    with pytest.raises(ValueError,match='A=12'):
        validar_coorte_27(m)
    m = coorte_sintetica()
    m['representantes'][1] = m['representantes'][0]
    with pytest.raises(ValueError,match='12 representantes'):
        validar_coorte_27(m)
    m = coorte_sintetica()
    m['representantes'][0] = m['modelos'][13]['id']  # duplica combinação, sem duplicar ID
    with pytest.raises(ValueError,match='12 combinações'):
        validar_coorte_27(m)


def test_plano_recusa_variar_intervencao_apos_primeiro_resultado():
    from scripts.executar_grade_final import validar_plano
    casos = [dict(id=f'{a}-{f}-{c}',arquitetura=a,fase=f,corte=c,lr=1e-4)
             for a in ('cnn_groupnorm','resnet50','mobilenetv2')
             for f in ('post_early','tres') for c in ('central','maior_tumor')]
    plano = dict(casos=casos,seed=42,batch=4,epocas=30,paciencia=5,teste_autorizado=False)
    validar_plano(plano)
    casos[1]['lr'] = 1e-3
    with pytest.raises(ValueError,match='mesma'):
        validar_plano(plano)
