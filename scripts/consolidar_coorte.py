"""Consolida A(12)+B(3)+C(12) exclusivamente pela validação e congela decisões."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np
import pandas as pd
from modulos.grade_experimental import familia, chave_selecao, selecionar_representantes
from modulos.analise_validacao import diferenca_auc_bootstrap
from scripts.analisar_experimentos import analisar, registros
from scripts.preparar_grade_final import sha


def ler(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def relativo(p):
    """Porta caminhos históricos D: e alias do sandbox para a raiz do clone."""
    texto = str(p).replace('\\', '/')
    for marcador in ('resultados/', 'checkpoints/'):
        if marcador in texto:
            return marcador + texto.split(marcador, 1)[1]
    raise ValueError(f'Caminho de artefato fora do projeto: {p}')


def gravar(p, dados):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def tabela(linhas):
    campos = ('auc_roc', 'sensibilidade', 'especificidade', 'precisao', 'acuracia', 'f1',
              'acuracia_balanceada', 'loss', 'tempo_treinamento_segundos', 'epoca',
              'epocas_executadas', 'limiar', 'tn', 'fp', 'fn', 'tp')
    saida = []
    for d in linhas:
        c = d['configuracao']
        saida.append(dict(codigo=d.get('codigo',''), id=d['id'], arquitetura=familia(c['arquitetura']), variante=c['arquitetura'],
                          fases=c['configuracao_temporal'], corte=c['estrategia_corte'], grupo=d['grupo'],
                          intervencao=d['intervencao'], **{k: d[k] for k in campos},
                          **{'prob_' + k: v for k, v in d['distribuicao']['todos'].items()},
                          pior_ba_limiar_001=d['estabilidade']['pior_acuracia_balanceada_mais_menos_001']))
    return pd.DataFrame(saida)


def consolidar(base):
    if (base / 'comparacao_validacao_27.json').exists():
        raise FileExistsError('Consolidação existente será preservada')
    anterior = ler(RAIZ / 'resultados/rodada_final_issue4_20261117_01/comparacao_validacao.json')
    plano = ler(base / 'plano_antes_treinamento.json')
    fontes = []
    for d in anterior['modelos']:
        grupo = 'A' if d['tipo'] == 'originais' else 'B'
        fontes.append(dict(id=d['id'], grupo=grupo, checkpoint=relativo(d['checkpoint']),
                           validacao=relativo(d['validacao']), historico=relativo(d['historico']),
                           intervencao='grade original' if grupo == 'A' else d['id'],
                           metrica_treino=d.get('metricas_treino')))
    for caso in plano['casos']:
        d = ler(base / caso['id'] / 'concluido.json')
        if d['plano_sha256'] != sha(base / 'plano_antes_treinamento.json') or d['teste_avaliado']:
            raise ValueError('Execução não corresponde ao planejamento sem teste')
        if sha(RAIZ / relativo(d['checkpoint'])) != d['checkpoint_sha256']:
            raise ValueError('Checkpoint modificado')
        fontes.append(dict(id=d['id'], grupo='C', checkpoint=relativo(d['checkpoint']),
                           validacao=relativo(d['validacao']), historico=relativo(d['historico']),
                           intervencao=caso['intervencao'], comparador=caso['comparador'],
                           metrica_treino=d['metricas_treino']))
    if len(fontes) != 27 or {g: sum(d['grupo'] == g for d in fontes) for g in 'ABC'} != {'A': 12, 'B': 3, 'C': 12}:
        raise ValueError('A coorte deve ser 12+3+12, sem duplicações')
    if len({d['checkpoint'] for d in fontes}) != 27:
        raise ValueError('Checkpoint duplicado')
    linhas, predicoes = [], {}
    numeros = {g: 0 for g in 'ABC'}
    for f in fontes:
        linha, ps = analisar(f['id'], f['grupo'], RAIZ/f['checkpoint'], RAIZ/f['validacao'], RAIZ/f['historico'])
        linha.update(grupo=f['grupo'], intervencao=f['intervencao'],
                     metricas_treino_anteriores=f['metrica_treino'])
        numeros[f['grupo']] += 1
        linha['codigo'] = f"{f['grupo']}{numeros[f['grupo']]:02d}"
        for k in ('checkpoint', 'validacao', 'historico'):
            linha[k] = relativo(linha[k])
        for k in ('checkpoint', 'validacao'):
            linha['entrada_manifesto'][k] = relativo(linha['entrada_manifesto'][k])
        hist = pd.read_csv(RAIZ/linha['historico'])
        melhor = hist.sort_values(['auc_roc', 'loss_validacao'], ascending=[False, True]).iloc[0]
        if int(melhor.epoca) != linha['epoca']:
            raise ValueError('Época selecionada não coincide com AUC/loss do histórico')
        linha['loss_treino_epoca_escolhida'] = float(hist.loc[hist.epoca.eq(linha['epoca']), 'loss_treino'].iloc[0])
        predicoes[f['id']] = ps
        if f['grupo'] == 'C':
            linha['comparador'] = f['comparador']
        linhas.append(linha)
        print('Validação auditada:', f['id'], flush=True)
    comparacoes = []
    for linha in linhas:
        if linha['grupo'] == 'C':
            comparador = next(d for d in linhas if d['id'] == linha['comparador'])
            bootstrap = diferenca_auc_bootstrap(predicoes[comparador['id']], predicoes[linha['id']])
            linha['bootstrap_vs_A'] = bootstrap
            linha['bootstrap_vs_melhor_anterior'] = diferenca_auc_bootstrap(
                predicoes[anterior['referencia_principal']],predicoes[linha['id']])
            comparacoes.append(dict(id=linha['id'], comparador=linha['comparador'], **bootstrap,
                                    delta_ba=linha['acuracia_balanceada']-comparador['acuracia_balanceada'],
                                    delta_sensibilidade=linha['sensibilidade']-comparador['sensibilidade'],
                                    delta_especificidade=linha['especificidade']-comparador['especificidade']))
    reps, melhores, geral = selecionar_representantes(linhas)
    selecao = dict(registrado_em_utc=datetime.now(timezone.utc).isoformat(), teste_usado=False,
                   representantes=reps, melhores_por_arquitetura=melhores, melhor_geral=geral,
                   criterio=plano['selecao'], ranking_validacao=[d['id'] for d in sorted(linhas, key=chave_selecao)])
    gravar(base/'selecao_validacao_congelada.json', selecao)
    gravar(base/'comparacao_validacao_27.json', dict(modelos=linhas, selecao=selecao, teste_avaliado=False))
    tabela(linhas).to_csv(base/'comparacao_validacao_27.csv', index=False)
    tabela([d for d in linhas if d['id'] in reps]).to_csv(base/'representantes_validacao_12.csv', index=False)
    pd.DataFrame(comparacoes).to_csv(base/'comparacao_pareada_C_vs_A.csv', index=False)
    gravar(base/'catalogo_27_antes_recarga.json', dict(modelos=[d['entrada_manifesto'] for d in linhas],
                                                    selecao=selecao, teste_usado_para_selecao=False))
    print(json.dumps(selecao, ensure_ascii=False, indent=2))


def congelar(base):
    destino = base/'manifesto_final_27.json'
    if destino.exists():
        raise FileExistsError('Manifesto final é imutável')
    catalogo = ler(base/'catalogo_27_antes_recarga.json')
    recarga = ler(base/'recarga_independente/relatorio.json')
    if recarga['catalogo_sha256'] != sha(base/'catalogo_27_antes_recarga.json'):
        raise ValueError('Catálogo mudou após recarga independente')
    if len(recarga['modelos']) != 27 or {d['id'] for d in recarga['modelos']} != {d['id'] for d in catalogo['modelos']}:
        raise ValueError('Recargas incompletas')
    for d in recarga['modelos']:
        if d['classes_divergentes'] != 0 or d['diferenca_max_prob'] > 1e-6 or d['pacientes'] != 33:
            raise ValueError('Recarga não reproduz validação')
        entrada = next(m for m in catalogo['modelos'] if m['id']==d['id'])
        original = ler(RAIZ/entrada['validacao'])['metricas']
        reproduzido = ler(base/'recarga_independente'/(d['id']+'__validacao.json'))['metricas']
        for k,v in original.items():
            if isinstance(v,(int,float)) and k in reproduzido:
                tolerancia = 1e-6 if k=='loss' else 1e-12
                if not np.isclose(v,reproduzido[k],rtol=0,atol=tolerancia):
                    raise ValueError(f'Métrica diverge após recarga: {d["id"]}/{k}')
    preservacao = ler(base/'preservacao_antes.json')
    divergentes = [p for p, h in preservacao.items() if sha(RAIZ/p) != h]
    if divergentes:
        raise ValueError(f'Artefatos anteriores alterados: {divergentes}')
    auditoria_preservacao = dict(arquivos=len(preservacao), alterados=divergentes,
                                verificado_em_utc=datetime.now(timezone.utc).isoformat())
    selecao = ler(base/'selecao_validacao_congelada.json')
    manifesto = dict(versao_protocolo=2, estado='congelado_antes_teste',
                     registrado_em_utc=datetime.now(timezone.utc).isoformat(),
                     autorizacao='Prompt definitivo do usuário: item 8, execução automática após verificações',
                     teste_usado_para_selecao=False, modelos=catalogo['modelos'],
                     representantes=selecao['representantes'], melhores_por_arquitetura=selecao['melhores_por_arquitetura'],
                     modelo_escolhido_pela_validacao=selecao['melhor_geral'],
                     csv_sha256=sha(RAIZ/'dataset/BreastDCEDL_SPY1_min_crop_metadata.csv'),
                     selecao_arquivo=relativo(base/'selecao_validacao_congelada.json'),
                     selecao_sha256=sha(base/'selecao_validacao_congelada.json'),
                     plano_arquivo=relativo(base/'plano_antes_treinamento.json'),
                     plano_sha256=sha(base/'plano_antes_treinamento.json'),
                     recarga_arquivo=relativo(base/'recarga_independente/relatorio.json'),
                     recarga_sha256=sha(base/'recarga_independente/relatorio.json'),
                     revisao_validacao_arquivo=relativo(base/'REVISAO_VALIDACAO_27.md'),
                     revisao_validacao_sha256=sha(base/'REVISAO_VALIDACAO_27.md'),
                     fontes_avaliacao_sha256={s: sha(RAIZ/s) for s in
                         ['scripts/avaliar_manifesto.py', 'modulos/modulo_treinamento.py',
                          'modulos/variantes_experimentais.py', 'modulos/grade_experimental.py',
                          'modulos/modulo_redes.py', 'modulos/modulo_dataset.py',
                          'scripts/consolidar_coorte.py', 'scripts/verificar_coorte_issue4.py']},
                     protocolo_teste=dict(pacientes=35, codigo_particao=1, aumento=False,
                                          limiar='fixo de cada checkpoint; sem arredondamento',
                                          selecionar_vencedor_no_teste=False))
    # Preflight completo antes de escrever o manifesto final.
    from scripts.avaliar_manifesto import validar_manifesto
    validar_manifesto(manifesto)
    gravar(base/'preservacao_antes_teste.json', auditoria_preservacao)
    gravar(destino, manifesto)
    with (base/'manifesto_final_27.sha256').open('x', encoding='ascii') as f:
        f.write(sha(destino)+'  manifesto_final_27.json\n')
    print('Manifesto aprovado e congelado:', sha(destino), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--congelar', action='store_true')
    a = p.parse_args()
    (congelar if a.congelar else consolidar)(a.base)


if __name__ == '__main__':
    main()
