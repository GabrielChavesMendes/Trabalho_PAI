"""Intervenções fixas da última grade; inferência mantém o contrato [N,C,H,W]."""
from torch import nn

from modulos.modulo_redes import CNNSimples


class CNNGroupNorm(CNNSimples):
    """Somente adiciona GroupNorm(4 grupos) entre cada Conv e sua ReLU.

    Convoluções, inicialização, pooling, cabeça e dropout são os da CNN original.
    As quatro convoluções e duas lineares continuam presentes.
    """

    def __init__(self, canais_entrada=1):
        super().__init__(canais_entrada)
        camadas = []
        for camada in self.caracteristicas:
            camadas.append(camada)
            if isinstance(camada, nn.Conv2d):
                camadas.append(nn.GroupNorm(4, camada.out_channels))
        self.caracteristicas = nn.Sequential(*camadas)


def familia(arquitetura):
    return 'cnn' if arquitetura in ('cnn', 'cnn_leaky', 'cnn_groupnorm') else arquitetura


def preparar_fine_tuning(modelo, arquitetura):
    """Uma política fixa por família, sem consultar resultados.

    ResNet: layer4 + fc, com todas as BN congeladas (affine e estatísticas).
    Mobile: features[17:19] (último residual + projeção) + classifier, BN congelada.
    O treinador existente mantém BN com affine congelado em eval a cada época.
    """
    if arquitetura == 'cnn_groupnorm':
        return
    for p in modelo.parameters():
        p.requires_grad = False
    if arquitetura == 'resnet50':
        blocos = (modelo.layer4, modelo.fc)
    elif arquitetura == 'mobilenetv2':
        blocos = (modelo.features[17:], modelo.classifier)
    else:
        raise ValueError('Intervenção não registrada para esta arquitetura')
    for bloco in blocos:
        for p in bloco.parameters():
            p.requires_grad = True
    for modulo in modelo.modules():
        if isinstance(modulo, nn.modules.batchnorm._BatchNorm):
            modulo.eval()
            for p in modulo.parameters():
                p.requires_grad = False


def chave_selecao(linha):
    """Ordem prévia: AUC, BA, robustez ±.01, sensibilidade, F1, loss, ID.

    Diferenças pequenas de AUC não demonstram superioridade estatística.
    O ranking é determinístico; não substitui a interpretação da incerteza.
    """
    return (-linha['auc_roc'], -linha['acuracia_balanceada'],
            -linha['estabilidade']['pior_acuracia_balanceada_mais_menos_001'],
            -linha['sensibilidade'], -linha['f1'], linha['loss'], linha['id'])


def selecionar_representantes(linhas):
    esperadas = {(a, f, c) for a in ('cnn', 'resnet50', 'mobilenetv2')
                 for f in ('post_early', 'tres') for c in ('central', 'maior_tumor')}
    grupos = {}
    for linha in linhas:
        conf = linha['configuracao']
        chave = (familia(conf['arquitetura']), conf['configuracao_temporal'], conf['estrategia_corte'])
        grupos.setdefault(chave, []).append(linha)
    if set(grupos) != esperadas:
        raise ValueError('Cobertura deve conter exatamente as 12 combinações')
    representantes = [sorted(grupos[k], key=chave_selecao)[0] for k in sorted(grupos)]
    melhores = {a: sorted([r for r in representantes if familia(r['configuracao']['arquitetura']) == a],
                          key=chave_selecao)[0]['id'] for a in ('cnn', 'resnet50', 'mobilenetv2')}
    return [r['id'] for r in representantes], melhores, sorted(linhas, key=chave_selecao)[0]['id']
