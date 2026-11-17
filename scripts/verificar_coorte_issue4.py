"""Processo independente: recarga dos 27 e inferência só em treino/validação."""
import argparse
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import pandas as pd
import torch
from modulos.rodada_final_issue4 import carregar_modelo_registrado, criar_carregador
from modulos.modulo_treinamento import avaliar_modelo, configurar_reprodutibilidade, salvar_resultado_json
from scripts.diagnosticar_cnn_issue4 import medir
from scripts.consolidar_coorte_issue4 import ler, gravar
from scripts.preparar_ultima_grade_issue4 import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    a = p.parse_args()
    destino = a.base/'recarga_independente'
    if destino.exists():
        raise FileExistsError('Recargas anteriores preservadas')
    catalogo = a.base/'catalogo_27_antes_recarga.json'
    dados = ler(catalogo)
    if len(dados['modelos'])!=27:
        raise ValueError('Necessários os 27 checkpoints')
    destino.mkdir()
    loaders, relatorio, diagnosticos = {}, [], []
    tabela = pd.read_csv(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv')
    treino = tabela.loc[tabela.test.eq(0)]
    ids_probe = set(treino.loc[treino.pCR.eq(0),'pid'].head(4)) | set(treino.loc[treino.pCR.eq(1),'pid'].head(4))
    for i, entrada in enumerate(dados['modelos'], 1):
        cp = RAIZ/entrada['checkpoint']
        if sha(cp)!=entrada['checkpoint_sha256']:
            raise ValueError('Checkpoint alterado antes de recarregar')
        configurar_reprodutibilidade(42)
        # Reconstrução e strict load CPU; só depois mover para GPU.
        modelo, bruto, norm = carregar_modelo_registrado(cp)
        modelo.cuda().eval()
        conf = bruto['configuracao']
        key = (conf['configuracao_temporal'],conf['estrategia_corte'],norm)
        criterio = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([conf['pos_weight']],device='cuda'))
        limiar = entrada['limiar']
        for particao in ('validacao', 'treino'):
            chave = (particao, *key)
            if chave not in loaders:
                loaders[chave] = criar_carregador(particao,conf,norm)
            loss, metricas, ps = avaliar_modelo(modelo,loaders[chave],criterio,torch.device('cuda'),limiar)
            salvar_resultado_json(destino/(entrada['id']+'__'+particao+'.json'),{**metricas,'loss':loss},ps)
            if particao == 'validacao':
                anterior = ler(RAIZ/entrada['validacao'])
                por_id = {p['paciente']:p for p in anterior['predicoes_por_paciente']}
                if set(por_id)!={p.paciente for p in ps}:
                    raise ValueError('IDs de validação divergentes')
                maximo = max(abs(p.probabilidade-por_id[p.paciente]['probabilidade']) for p in ps)
                classes = sum(int(p.probabilidade>=limiar)!=por_id[p.paciente]['previsto'] for p in ps)
                if maximo>1e-6 or classes or abs(loss-anterior['metricas']['loss'])>1e-6:
                    raise ValueError(f'Recarga não reproduz validação: {entrada["id"]}')
                relatorio.append(dict(id=entrada['id'],checkpoint_sha256=sha(cp),
                                      diferenca_max_prob=maximo,classes_divergentes=classes,pacientes=len(ps),
                                      normalizacao=norm,recarga_estrita_cpu=True,teste_avaliado=False))
        if conf['arquitetura'] in ('cnn','cnn_groupnorm') and entrada['grupo'] in ('A','C'):
            # Retira os wrappers de augmentation/cache para a mesma sonda sem aumento.
            from modulos.modulo_dataset import criar_manifesto, DatasetDCE
            pacientes = [p for p in criar_manifesto(treino,RAIZ/'dataset/BreastDCEDL_ISPY1_min_crop')
                         if p.identificador in ids_probe]
            ds = DatasetDCE(pacientes,fase=conf['configuracao_temporal'],estrategia=conf['estrategia_corte'])
            amostras = [ds[j] for j in range(len(ds))]
            x = torch.stack([d['imagem'] for d in amostras]).cuda()
            y = torch.stack([d['rotulo'] for d in amostras]).reshape(-1,1).cuda()
            modelo.zero_grad(set_to_none=True)
            diagnosticos.append(dict(id=entrada['id'],grupo=entrada['grupo'],pacientes=[p.identificador for p in pacientes],
                                      otimizacoes=0,**medir(modelo,x,y)))
            del x,y
        print(f'[{i}/27] Recarga e validação reproduzidas: {entrada["id"]}',flush=True)
        del modelo,bruto
        torch.cuda.empty_cache()
    gravar(destino/'diagnostico_ativacoes_cnn.json',diagnosticos)
    gravar(destino/'relatorio.json',dict(catalogo_sha256=sha(catalogo),modelos=relatorio,
                                      teste_avaliado=False,novos_treinamentos=0))


if __name__ == '__main__':
    main()
