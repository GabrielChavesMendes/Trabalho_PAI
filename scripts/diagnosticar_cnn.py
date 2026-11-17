"""Diagnóstico sem otimização: imagens de treino, ativações e gradientes da CNN."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modulos.modulo_dataset import carregar_metadados, criar_manifesto, DatasetDCE
from modulos.modulo_redes import criar_modelo
from modulos.modulo_treinamento import configurar_reprodutibilidade


def medir(modelo, imagens, rotulos):
    modelo.eval()
    ativacoes = {}
    hooks = []
    def hook(nome):
        def registrar(modulo, entrada, saida):
            x = saida.detach().float()
            dims = (0, 2, 3) if x.ndim == 4 else (0,)
            ativacoes[nome] = {
                'shape': list(x.shape), 'media': x.mean().item(), 'std': x.std().item(),
                'fracao_zeros': (x == 0).float().mean().item(),
                'fracao_canais_totalmente_zero': (x.abs().sum(dim=dims) == 0).float().mean().item(),
                'variacao_entre_pacientes': x.flatten(1).std(dim=0).mean().item(),
            }
        return registrar
    for nome, modulo in modelo.named_modules():
        if isinstance(modulo, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(modulo.register_forward_hook(hook(nome)))
    logits = modelo(imagens)
    loss = nn.BCEWithLogitsLoss(pos_weight=imagens.new_tensor([3.]))(logits, rotulos)
    loss.backward()
    for h in hooks:
        h.remove()
    return {
        'loss_treino_probe': loss.item(),
        'logits': logits.detach().cpu().flatten().tolist(),
        'amplitude_probabilidades': torch.sigmoid(logits).detach().max().item() - torch.sigmoid(logits).detach().min().item(),
        'ativacoes': ativacoes,
        'gradientes_norma': {n: p.grad.norm().item() if p.grad is not None else None for n, p in modelo.named_parameters()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--saida', type=Path, required=True)
    args = parser.parse_args()
    if args.saida.exists():
        raise FileExistsError(args.saida)
    tabela = carregar_metadados(RAIZ / 'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    treino = tabela.loc[tabela.test.eq(0)]
    amostra = torch.tensor([0])  # nenhuma seleção baseada na validação
    ids = set(treino.loc[treino.pCR.eq(0), 'pid'].head(4)) | set(treino.loc[treino.pCR.eq(1), 'pid'].head(4))
    pacientes = [p for p in criar_manifesto(treino, RAIZ / 'dataset/BreastDCEDL_ISPY1_min_crop') if p.identificador in ids]
    ds = DatasetDCE(pacientes, fase='post_early', estrategia='maior_tumor')
    dados = [ds[i] for i in range(len(ds))]
    imagens = torch.stack([d['imagem'] for d in dados]).cuda()
    rotulos = torch.stack([d['rotulo'] for d in dados]).reshape(-1, 1).cuda()
    bruto = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    configurar_reprodutibilidade(42)
    modelo = criar_modelo('cnn', 1, pre_treinado=False).cuda()
    inicial = medir(modelo, imagens, rotulos)
    modelo.zero_grad(set_to_none=True)
    modelo.load_state_dict(bruto['model_state_dict'])
    salvo = medir(modelo, imagens, rotulos)
    resultado = {
        'particao': 'somente treino test=0', 'pacientes': [p.identificador for p in pacientes],
        'otimizacoes_executadas': 0, 'teste_acessado': False,
        'imagens': {'min': imagens.min().item(), 'max': imagens.max().item(),
                    'std': imagens.std().item(), 'std_espacial_media': imagens.flatten(1).std(dim=1).mean().item()},
        'inicializacao_seed42': inicial, 'checkpoint_epoca': bruto['epoca'], 'checkpoint': salvo,
        'referencia_loss_constante_balanceada_treino': '1.5*ln(2) = 1.03972077, para proporcao 78/26 e pos_weight=3',
    }
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
