"""Extensões optativas da rodada final; não altera os pipelines anteriores."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modulos.fabrica_carregadores import DatasetCacheMemoria
from modulos.modulo_dataset import (
    DatasetDCE, FASES, PARTICOES, carregar_metadados, criar_manifesto,
    selecionar_particao, verificar_arquivos,
)
from modulos.dataset_aumentado import DatasetDCEAumentado
from modulos.modulo_redes import CNNSimples, criar_modelo

RAIZ = Path(__file__).resolve().parents[1]
MEDIA_IMAGENET = (.485, .456, .406)
DESVIO_IMAGENET = (.229, .224, .225)


class CNNLeaky(CNNSimples):
    """Mesmos seis níveis com parâmetros, somente ativações diferentes."""
    def __init__(self, canais_entrada=1):
        super().__init__(canais_entrada)
        for sequencia in (self.caracteristicas, self.classificador):
            for indice, modulo in enumerate(sequencia):
                if isinstance(modulo, nn.ReLU):
                    sequencia[indice] = nn.LeakyReLU(negative_slope=.01)
        # Faz uma carga incorreta na CNN antiga falhar em vez de trocar a função.
        self.register_buffer('assinatura_leaky_001', torch.tensor(.01))


def criar_modelo_registrado(arquitetura, canais_entrada, pre_treinado=False):
    if arquitetura == 'cnn_leaky':
        return CNNLeaky(canais_entrada)
    if arquitetura == 'cnn_groupnorm':
        from modulos.grade_experimental import CNNGroupNorm
        return CNNGroupNorm(canais_entrada)
    return criar_modelo(arquitetura, canais_entrada, pre_treinado=pre_treinado)


def normalizar_adicional(imagem, metodo):
    if metodo == 'nenhuma':
        return imagem
    if metodo != 'imagenet_rgb':
        raise ValueError(f'Normalização adicional desconhecida: {metodo}')
    if imagem.ndim != 3 or imagem.shape[0] != 3:
        raise ValueError('imagenet_rgb requer três canais [C,H,W]')
    media = imagem.new_tensor(MEDIA_IMAGENET).reshape(3, 1, 1)
    desvio = imagem.new_tensor(DESVIO_IMAGENET).reshape(3, 1, 1)
    return (imagem - media) / desvio


class DatasetNormalizado(Dataset):
    def __init__(self, base, metodo):
        self.base = base
        self.metodo = metodo

    def __len__(self):
        return len(self.base)

    def __getitem__(self, indice):
        amostra = dict(self.base[indice])
        amostra['imagem'] = normalizar_adicional(amostra['imagem'], self.metodo)
        return amostra


def criar_carregador(particao, configuracao, normalizacao='nenhuma', *, permitir_teste=False):
    """Constrói SOMENTE a partição pedida; teste exige opt-in do avaliador final."""
    if particao not in PARTICOES:
        raise ValueError('Partição desconhecida')
    if particao == 'teste' and not permitir_teste:
        raise ValueError('Teste bloqueado durante a investigação e seleção')
    fase = configuracao['configuracao_temporal']
    corte = configuracao['estrategia_corte']
    if normalizacao not in ('nenhuma', 'imagenet_rgb'):
        raise ValueError('Normalização desconhecida')
    if normalizacao == 'imagenet_rgb' and fase != 'tres':
        raise ValueError('Padronização ImageNet desta rodada requer três fases')
    csv = Path(os.environ.get('PAI_CSV', RAIZ / 'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv'))
    pasta = Path(os.environ.get('PAI_DATASET_DIR', RAIZ / 'dataset/BreastDCEDL_ISPY1_min_crop'))
    tabela = carregar_metadados(csv)
    # Filtrar ANTES do manifesto: nem construir/abrir caminhos de outra partição.
    tabela = tabela.loc[tabela.test.eq(PARTICOES[particao])]
    manifesto = criar_manifesto(tabela, pasta)
    esperado = {'treino': 104, 'validacao': 33, 'teste': 35}[particao]
    if len(manifesto) != esperado:
        raise ValueError(f'Esperados {esperado} pacientes em {particao}')
    erros = verificar_arquivos(manifesto)
    if erros:
        raise FileNotFoundError('\n'.join(erros[:5]))
    pacientes = selecionar_particao(manifesto, particao)
    base = DatasetCacheMemoria(DatasetDCE(pacientes, fase=fase, estrategia=corte))
    dataset = DatasetDCEAumentado(base, particao) if particao != 'teste' else base
    dataset = DatasetNormalizado(dataset, normalizacao)
    lote = int(configuracao['tamanho_lote'])
    if lote < 1:
        raise ValueError('Batch deve ser positivo')
    # Mesmo shuffle por seed, sem workers/cópias paralelas no Windows.
    return DataLoader(dataset, batch_size=lote, shuffle=particao == 'treino',
                      generator=torch.Generator().manual_seed(int(configuracao['seed'])),
                      num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)


def metadados_rodada(caso):
    return {
        'fases': list(FASES) if caso['fase'] == 'tres' else [caso['fase']],
        'normalizacao': 'percentis 1/99 conjuntos por amostra; clip [0,1]',
        'normalizacao_adicional': caso['normalizacao_adicional'],
        'parametros_normalizacao_adicional': (
            {'media': list(MEDIA_IMAGENET), 'desvio': list(DESVIO_IMAGENET),
             'momento': 'depois do aumento; sem resize/crop adicional'}
            if caso['normalizacao_adicional'] == 'imagenet_rgb' else {}),
        'corte': caso['corte'],
        'augmentation': {'particoes': ['treino', 'validacao'], 'angulos_graus': [-20, -10, 0, 10, 20],
                         'espelhamento': [False, True], 'variacoes_por_paciente': 10, 'teste_aumentado': False},
        'particoes': {'treino': 0, 'validacao': 2, 'teste': 1},
        'pacientes': {'treino': 104, 'validacao': 33, 'teste': 35},
        'rodada_final_issue4': {'id': caso['id'], 'arquitetura': caso['arquitetura'],
                              'pre_treinado': caso['arquitetura'] == 'resnet50',
                              'fine_tuning': False, 'dropout_cnn': .3,
                              'ativacao_cnn': 'LeakyReLU(0.01)' if caso['arquitetura'] == 'cnn_leaky' else 'ReLU'},
    }


def carregar_modelo_registrado(caminho, dispositivo='cpu'):
    bruto = torch.load(caminho, map_location='cpu', weights_only=False)
    if bruto.get('formato_checkpoint') != 1:
        raise ValueError('Formato de checkpoint desconhecido')
    conf = bruto['configuracao']
    modelo = criar_modelo_registrado(conf['arquitetura'], conf['canais_entrada'])
    modelo.load_state_dict(bruto['model_state_dict'], strict=True)
    modelo.to(dispositivo).eval()
    normalizacao = bruto['preprocessamento'].get('normalizacao_adicional', 'nenhuma')
    if normalizacao not in ('nenhuma', 'imagenet_rgb'):
        raise ValueError('Checkpoint com normalização não suportada')
    if normalizacao == 'imagenet_rgb' and conf['canais_entrada'] != 3:
        raise ValueError('Configuração/normalização divergentes')
    return modelo, bruto, normalizacao
