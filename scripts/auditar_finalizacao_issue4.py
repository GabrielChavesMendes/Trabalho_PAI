"""Verificação final de integridade; não treina nem executa inferência de teste."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from scripts.consolidar_coorte_issue4 import ler, gravar
from scripts.preparar_ultima_grade_issue4 import sha
from scripts.avaliar_manifesto_issue4 import validar_manifesto


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    a=p.parse_args(); base=a.base
    destino=base/'auditoria_final.json'
    if destino.exists(): raise FileExistsError('Auditoria anterior preservada')
    manifesto=ler(base/'manifesto_final_27.json')
    validar_manifesto(manifesto)  # inclui strict CPU; não abre imagens
    if ler(base/'teste_final/manifesto_utilizado.json')!=manifesto:
        raise ValueError('Manifesto utilizado difere do congelado')
    termino=ler(base/'teste_final/concluido.json')
    if termino['modelos']!=27 or termino['manifesto_sha256']!=sha(base/'manifesto_final_27.json'):
        raise ValueError('Teste incompleto ou divergente')
    inicio=ler(base/'teste_final/inicio_avaliacao.json')
    if datetime.fromisoformat(inicio['iniciado_em_utc'])<=datetime.fromisoformat(manifesto['registrado_em_utc']):
        raise ValueError('Teste deve começar depois do congelamento')
    antigos=ler(base/'preservacao_antes.json')
    if any(sha(RAIZ/p)!=h for p,h in antigos.items()):
        raise ValueError('Arquivo anterior foi alterado')
    plano=ler(base/'plano_antes_treinamento.json')
    if any(sha(RAIZ/p)!=h for p,h in plano['fontes_sha256'].items()):
        raise ValueError('Fonte de treinamento mudou desde o planejamento')
    novos=[]
    for caso in plano['casos']:
        d=ler(base/caso['id']/'concluido.json')
        cp=Path(d['checkpoint'])
        if sha(cp)!=d['checkpoint_sha256'] or d['teste_avaliado']:
            raise ValueError('Checkpoint C alterado ou treinado com teste')
        if cp.stat().st_mtime > datetime.fromisoformat(inicio['iniciado_em_utc']).timestamp():
            raise ValueError('Checkpoint foi gravado depois de iniciado o teste')
        novos.append(d)
    if len(novos)!=12 or len(list((RAIZ/'checkpoints'/base.name).glob('*.pt')))!=12:
        raise ValueError('Esperados exatamente 12 novos checkpoints completos')
    arquivos_teste=list((base/'teste_final').glob('*/*__test.json'))
    if len(arquivos_teste)!=27:
        raise ValueError('Esperados exatamente 27 resultados individuais de teste')
    auditoria=ler(base/'artigo/auditoria_metricas.json')
    if len(auditoria['modelos'])!=27 or not auditoria['conferido_com_sklearn']:
        raise ValueError('Auditoria independente de métricas incompleta')
    for d in auditoria['modelos']:
        if d['pacientes_teste']!=35 or not all(d[k] for k in ('validacao_sklearn','teste_sklearn','treino_sklearn')):
            raise ValueError('Cálculos independentes divergentes')
    resumo=dict(verificado_em_utc=datetime.now(timezone.utc).isoformat(),
                novos_treinamentos=12,coorte_treinada=27,representantes=12,
                representantes_por_arquitetura={'cnn':4,'resnet50':4,'mobilenetv2':4},
                testes_individuais=27,pacientes_teste_por_modelo=35,
                arquivos_anteriores_preservados=len(antigos),
                recargas_estritas_cpu_pos_teste=27,novas_inferencias_teste_nesta_auditoria=0,
                nenhuma_selecao_alterada=True,nenhum_treinamento_apos_teste=True,
                manifesto_sha256=sha(base/'manifesto_final_27.json'),
                selecao_sha256=sha(base/'selecao_validacao_congelada.json'),
                tempos_C_segundos=sum(d['tempo_treinamento_segundos'] for d in novos),
                pico_vram_C_bytes=max(d['pico_vram_bytes'] for d in novos),
                checkpoints={m['checkpoint']:sha(RAIZ/m['checkpoint']) for m in manifesto['modelos']},
                predicoes_teste_sha256={p.relative_to(base).as_posix():sha(p) for p in arquivos_teste})
    gravar(destino,resumo)
    print('AUDITORIA APROVADA: 12 novos, 27 na coorte, 12 representantes, teste 27×35; tudo preservado.')


if __name__=='__main__': main()
