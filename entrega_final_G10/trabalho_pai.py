# Integrantes: Alessandra Faria Rodrigues; Felipe Barros Ratton de Almeida; Gabriel Chaves Mendes; Gustavo Henrique Rodrigues de Castro
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

RAIZ = Path(__file__).resolve().parents[1]
CSV_PADRAO = Path(os.environ.get("PAI_CSV", RAIZ / "dataset/BreastDCEDL_SPY1_min_crop_metadata.csv"))
DATASET_PADRAO = Path(os.environ.get("PAI_DATASET_DIR", RAIZ / "dataset/BreastDCEDL_ISPY1_min_crop"))
FASES = ("pre", "post_early", "post_late")
FASES_TRES = ["pre", "post_early", "post_late"]
ARQUITETURAS = {"cnn", "resnet50", "mobilenetv2"}


@dataclass(frozen=True)
class Paciente:
    identificador: str
    pcr: int
    caminhos_dce: tuple[Path, Path, Path]
    caminho_mascara: Path
    particao_codigo: int = 0


def carregar_metadados(caminho: Path) -> list[Paciente]:
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV de metadados nao encontrado: {caminho}")
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        obrigatorias = {"pid", "pCR", "dataset", "test", "pre", "post_early", "post_late"}
        ausentes = obrigatorias - set(leitor.fieldnames or [])
        if ausentes:
            raise ValueError("CSV sem colunas: " + ", ".join(sorted(ausentes)))
        registros = list(leitor)
    if len(registros) != 172:
        raise ValueError(f"Esperados 172 pacientes I-SPY1; encontrados {len(registros)}.")
    pacientes = []
    vistos = set()
    for linha in registros:
        pid = str(linha["pid"]).strip()
        if linha["dataset"] != "spy1" or not pid.startswith("ISPY1_") or pid in vistos:
            raise ValueError(f"Registro/identificador invalido no CSV: {pid}")
        vistos.add(pid)
        try:
            def converter_inteiro(valor, campo):
                numero = float(valor)

                if not numero.is_integer():
                    raise ValueError(
                        f"Valor nao inteiro no campo {campo}: {valor}"
                    )

                return int(numero)

            rotulo = converter_inteiro(linha["pCR"], "pCR")
            particao = converter_inteiro(linha["test"], "test")
            indices = [
                converter_inteiro(linha[f], f)
                for f in FASES
            ]

        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Metadados invalidos para {pid}: {exc}"
            ) from exc
        if rotulo not in (0, 1) or particao not in (0, 1, 2) or min(indices) < 0:
            raise ValueError(f"Rotulo ou indices temporais invalidos para {pid}")
        prefixo = f"{pid}_spy1_vis1"
        dce = tuple(DATASET_PADRAO / "dce" / f"{prefixo}_dce_aqc_{idx}.nii.gz" for idx in indices)
        mascara = DATASET_PADRAO / "mask" / f"{prefixo}_mask.nii.gz"
        pacientes.append(Paciente(pid, rotulo, dce, mascara, particao))
    return pacientes


def ler_volume(caminho: Path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Dependencia nibabel ausente. Instale com pip install nibabel.") from exc
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo NIfTI ausente: {caminho}")
    nii = nib.load(str(caminho))
    dados = np.asarray(nii.dataobj, dtype=np.float32)
    if dados.ndim == 4 and dados.shape[-1] == 1:
        dados = dados[..., 0]
    if dados.ndim != 3 or dados.shape[1:] != (256, 256):
        raise ValueError(f"Volume deve ter formato [Z,256,256], recebido {dados.shape}: {caminho}")
    if not np.isfinite(dados).all():
        raise ValueError(f"Volume contem NaN/Inf: {caminho}")
    return dados, np.asarray(nii.affine)


def selecionar_corte(mascara: np.ndarray, estrategia: str) -> int:
    if estrategia == "central":
        return mascara.shape[0] // 2
    if estrategia == "maior_tumor":
        areas = np.count_nonzero(mascara > 0, axis=(1, 2))
        if not areas.any():
            raise ValueError("Mascara sem pixels tumorais positivos.")
        return int(areas.argmax())
    raise ValueError(f"Estrategia de corte nao suportada: {estrategia}")


def normalizar_percentis(canais: np.ndarray) -> np.ndarray:
    if canais.ndim != 3 or not np.isfinite(canais).all():
        raise ValueError("Entrada deve ser [C,H,W] e conter valores finitos.")
    inferior, superior = np.percentile(canais, [1, 99])
    if superior <= inferior:
        return np.zeros_like(canais, dtype=np.float32)
    return np.ascontiguousarray(np.clip((canais - inferior) / (superior - inferior), 0, 1), dtype=np.float32)


def preparar_paciente(paciente: Paciente, configuracao: dict, preprocessamento: dict):
    fase = configuracao["configuracao_temporal"]
    corte = configuracao["estrategia_corte"]
    mascara, affine_mascara = ler_volume(paciente.caminho_mascara)
    indices = range(3) if fase == "tres" else (FASES.index(fase),)
    canais = []
    for indice in indices:
        volume, affine = ler_volume(paciente.caminhos_dce[indice])
        if volume.shape != mascara.shape or not np.allclose(affine, affine_mascara, rtol=1e-4, atol=1e-3):
            raise ValueError(f"DCE e mascara desalinhados para {paciente.identificador}.")
        canais.append(volume)
    z = selecionar_corte(mascara, corte)
    imagem = normalizar_percentis(np.stack([v[z] for v in canais], axis=0))
    norma = preprocessamento.get("normalizacao", "percentis 1/99 conjuntos por amostra; clip [0,1]")
    tipo = norma.get("tipo", "") if isinstance(norma, dict) else ""
    adicional = preprocessamento.get("normalizacao_adicional", "nenhuma")
    usar_imagenet = adicional in ("imagenet_rgb", "imagenet") or tipo.endswith("imagenet")
    if usar_imagenet:
        if imagem.shape[0] == 3:
            media = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            desvio = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        elif imagem.shape[0] == 1:
            media = np.array([0.449], dtype=np.float32)[:, None, None]
            desvio = np.array([0.226], dtype=np.float32)[:, None, None]
        else:
            raise ValueError("Normalizacao ImageNet requer 1 ou 3 canais.")
        imagem = (imagem - media) / desvio
    mascara_2d = (mascara[z] > 0).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(imagem)), mascara_2d, z


class CNNSimples(nn.Module):
    def __init__(self, canais_entrada=3):
        super().__init__()
        if canais_entrada not in (1, 3):
            raise ValueError("A CNN aceita 1 ou 3 canais.")
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(canais_entrada, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.classificador = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, imagens):
        return self.classificador(self.caracteristicas(imagens))


def adaptar_primeira_convolucao(original: nn.Conv2d, canais: int):
    if canais == 3:
        return original
    nova = nn.Conv2d(1, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None)
    with torch.no_grad():
        nova.weight.copy_(original.weight.sum(dim=1, keepdim=True))
        if original.bias is not None:
            nova.bias.copy_(original.bias)
    return nova


def criar_modelo(arquitetura: str, canais: int, pretreinado: bool = False):
    if arquitetura == "cnn":
        return CNNSimples(canais)
    try:
        from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
    except ImportError as exc:
        raise ImportError("torchvision compativel com PyTorch nao esta instalado.") from exc
    if arquitetura == "resnet50":
        modelo = resnet50(weights=ResNet50_Weights.DEFAULT if pretreinado else None)
        modelo.conv1 = adaptar_primeira_convolucao(modelo.conv1, canais)
        modelo.fc = nn.Linear(modelo.fc.in_features, 1)
        return modelo
    if arquitetura == "mobilenetv2":
        modelo = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretreinado else None)
        modelo.features[0][0] = adaptar_primeira_convolucao(modelo.features[0][0], canais)
        modelo.classifier[1] = nn.Linear(modelo.last_channel, 1)
        return modelo
    raise ValueError(f"Arquitetura nao suportada: {arquitetura}")


def validar_checkpoint(caminho: Path) -> dict:
    if caminho.suffix.lower() != ".pt":
        raise ValueError("Selecione um checkpoint .pt.")
    if not caminho.is_file() or caminho.stat().st_size == 0:
        raise FileNotFoundError("Checkpoint ausente ou vazio.")
    try:
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            dados = torch.load(caminho, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Nao foi possivel ler o checkpoint com seguranca: {exc}") from exc
    if not isinstance(dados, dict) or dados.get("formato_checkpoint") != 1:
        raise ValueError("Formato de checkpoint invalido ou nao suportado.")
    for campo in ("model_state_dict", "configuracao", "preprocessamento", "selecao_validacao"):
        if not isinstance(dados.get(campo), dict):
            raise ValueError(f"Checkpoint sem metadados obrigatorios: {campo}.")
    conf, prep = dados["configuracao"], dados["preprocessamento"]
    arq, fase, corte, canais = conf.get("arquitetura"), conf.get("configuracao_temporal"), conf.get("estrategia_corte"), conf.get("canais_entrada")
    if arq not in ARQUITETURAS:
        raise ValueError("Checkpoint incompativel: arquitetura nao suportada.")
    if fase not in ("post_early", "tres") or corte not in ("central", "maior_tumor"):
        raise ValueError("Configuracao temporal ou estrategia de corte nao suportada.")
    if canais != (3 if fase == "tres" else 1):
        raise ValueError("Quantidade de canais incompativel com as fases registradas.")
    fases_esperadas = FASES_TRES if fase == "tres" else ["post_early"]
    if prep.get("fases") != fases_esperadas or prep.get("corte") != corte:
        raise ValueError("Metadados de fases/corte divergem da arquitetura.")
    if prep.get("mascara_como_filtro") or prep.get("recorte_roi") or prep.get("representacao_entrada", "fases") != "fases":
        raise ValueError("Checkpoint exige transformacao de entrada nao suportada.")
    norma = prep.get("normalizacao")
    tipo = norma.get("tipo") if isinstance(norma, dict) else None
    legado = norma == "percentis 1/99 conjuntos por amostra; clip [0,1]" and prep.get("normalizacao_adicional", "nenhuma") == "nenhuma"
    if not legado and tipo not in ("percentil_1_99_conjunto", "percentil_1_99_seguido_normalizacao_imagenet"):
        raise ValueError("Normalizacao do checkpoint ausente ou nao suportada.")
    if not 0 < float(dados["selecao_validacao"].get("limiar", 0)) < 1:
        raise ValueError("Limiar de classificacao ausente ou invalido.")
    if not isinstance(dados["selecao_validacao"].get("metricas"), dict):
        raise ValueError("Metricas de validacao ausentes no checkpoint.")
    if not isinstance(dados["model_state_dict"], dict):
        raise ValueError("Pesos state_dict ausentes ou invalidos.")
    return dados


def carregar_modelo_checkpoint(caminho: Path, dispositivo):
    dados = validar_checkpoint(caminho)
    conf = dados["configuracao"]
    modelo = criar_modelo(conf["arquitetura"], int(conf["canais_entrada"]))
    try:
        modelo.load_state_dict(dados["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("Pesos incompativeis com a estrutura da rede (strict=True).") from exc
    modelo.to(dispositivo).eval()
    return modelo, dados


def resolver_camada_gradcam(modelo: nn.Module, arquitetura: str):
    if arquitetura == "resnet50":
        return modelo.layer4[-1]
    if arquitetura == "mobilenetv2":
        return modelo.features[-1]
    if arquitetura == "cnn":
        camadas = list(modelo.caracteristicas.children())
        for indice in range(len(camadas) - 1, -1, -1):
            if isinstance(camadas[indice], nn.Conv2d):
                return camadas[indice + 1] if indice + 1 < len(camadas) and isinstance(camadas[indice + 1], nn.ReLU) else camadas[indice]
    raise ValueError("Nao foi possivel resolver a camada Grad-CAM.")


def gerar_gradcam(
    modelo: nn.Module,
    imagem: torch.Tensor,
    arquitetura: str,
    classe_alvo: str = "prevista",
    limiar: float = 0.5,
):
    if classe_alvo not in ("prevista", "pcr1"):
        raise ValueError("Classe-alvo invalida: use prevista ou pcr1.")
    if imagem.ndim != 4 or imagem.shape[0] != 1 or not bool(torch.isfinite(imagem).all()):
        raise ValueError("Grad-CAM requer tensor finito [1,C,H,W].")
    camada = resolver_camada_gradcam(modelo, arquitetura)
    ativacoes, gradientes = [], []

    def capturar(_modulo, _entrada, saida):
        if not isinstance(saida, torch.Tensor) or saida.ndim != 4:
            raise RuntimeError("Camada Grad-CAM deve produzir [N,C,H,W].")
        ativacoes.append(saida)
        if saida.requires_grad:
            saida.register_hook(lambda grad: gradientes.append(grad))

    estava_treinando = modelo.training
    handle = camada.register_forward_hook(capturar)
    try:
        modelo.eval()
        modelo.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = modelo(imagem)
            if logits.shape != (1, 1) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Modelo deve retornar um logit finito [1,1].")
            logit = logits.reshape(-1)[0]
            prob = float(torch.sigmoid(logit.detach()).cpu())
            prevista = int(prob >= limiar)
            classe_explicada = prevista if classe_alvo == "prevista" else 1
            escore_explicado = logit if classe_explicada == 1 else -logit
            escore_explicado.backward()
        if not ativacoes or not gradientes:
            raise RuntimeError("Ativacoes/gradientes Grad-CAM nao foram capturados.")
        pesos = gradientes[-1].mean(dim=(2, 3), keepdim=True)
        bruto = F.relu((pesos * ativacoes[-1]).sum(dim=1, keepdim=True))
        minimo, maximo = float(bruto.min()), float(bruto.max())
        amplitude = maximo - minimo
        if amplitude <= 1e-12:
            mapa = torch.zeros(
                (1, 1, *imagem.shape[-2:]), device=bruto.device, dtype=bruto.dtype
            )
        else:
            mapa = (bruto - minimo) / amplitude
            mapa = F.interpolate(
                mapa, size=imagem.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        diagnostico = {
            "classe_explicada": classe_explicada,
            "classe_prevista": prevista,
            "minimo_bruto": minimo,
            "maximo_bruto": maximo,
            "amplitude_bruta": amplitude,
            "resolucao_nativa": tuple(bruto.shape[-2:]),
            "mapa_nulo": amplitude <= 1e-12,
        }
        return mapa[0, 0].detach().cpu().numpy(), prob, diagnostico
    finally:
        handle.remove()
        modelo.zero_grad(set_to_none=True)
        modelo.train(estava_treinando)


class AplicacaoPAI:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("PAI | Análise de resposta patológica")
        self.raiz.geometry("1380x940")
        self.raiz.minsize(1120, 760)
        self.estilo = ttk.Style()
        if "clam" in self.estilo.theme_names():
            self.estilo.theme_use("clam")
        self.estilo.configure("TFrame", background="#F3F5F8")
        self.estilo.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        self.estilo.configure("TLabel", background="#F3F5F8", foreground="#243247", font=("Segoe UI", 10))
        self.estilo.configure("Title.TLabel", background="#172B4D", foreground="#FFFFFF", font=("Segoe UI", 19, "bold"), padding=(18, 14))
        self.estilo.configure("Subtitle.TLabel", background="#172B4D", foreground="#DCE6F2", font=("Segoe UI", 9), padding=(18, 0, 18, 14))
        self.estilo.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(12, 8))
        self.estilo.map("Accent.TButton", background=[("active", "#2457A6"), ("!disabled", "#326BC5")], foreground=[("!disabled", "#FFFFFF")])
        self.estilo.configure("Accent.TButton", background="#326BC5", foreground="#FFFFFF")
        self.estilo.configure("TCombobox", padding=6)
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = None
        self.checkpoint = None
        self.caminho_checkpoint = None
        self.pacientes = []
        self.figura = None
        self.canvas = None
        self.volumes_visualizacao = {}
        self.mascara_volume = None
        self.paciente_visualizacao = None
        self.gradcam_atual = None
        self._construir_interface()
        self.status.set(f"Dispositivo: {self.dispositivo}. Carregue um checkpoint .pt compativel.")

    def _construir_interface(self):
        cabecalho = ttk.Frame(self.raiz, style="Card.TFrame")
        cabecalho.pack(fill="x")
        ttk.Label(cabecalho, text="PAI | Análise de resposta patológica", style="Title.TLabel").pack(fill="x")
        ttk.Label(cabecalho, text="Classificação pCR • DCE-MRI • Visualização Grad-CAM | Grupo G10", style="Subtitle.TLabel").pack(fill="x")
        topo = ttk.Frame(self.raiz, style="Card.TFrame", padding=16)
        topo.pack(fill="x", padx=14, pady=(14, 8))
        self.caminho_var = tk.StringVar()
        self.csv_var = tk.StringVar(value=str(CSV_PADRAO))
        self.dataset_var = tk.StringVar(value=str(DATASET_PADRAO))
        self.checkpoint_nome_var = tk.StringVar(value="Nenhum checkpoint selecionado")
        self.csv_nome_var = tk.StringVar(value=Path(CSV_PADRAO).name)
        self.dce_nome_var = tk.StringVar(value="Dataset padrão do projeto")
        ttk.Label(topo, text="Checkpoint treinado (.pt)").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Label(topo, textvariable=self.checkpoint_nome_var, width=42).grid(row=0, column=1, padx=8, sticky="w")
        ttk.Button(topo, text="Selecionar arquivo .pt", command=self.selecionar_checkpoint).grid(row=0, column=2, padx=5)
        ttk.Button(topo, text="Validar e carregar", style="Accent.TButton", command=self.carregar_checkpoint).grid(row=0, column=3, padx=5)
        ttk.Label(topo, text="Metadados dos pacientes (.csv)").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Label(topo, textvariable=self.csv_nome_var, width=42).grid(row=1, column=1, padx=8, sticky="w")
        ttk.Button(topo, text="Selecionar arquivo .csv", command=self.selecionar_csv).grid(row=1, column=2, padx=5)
        ttk.Label(topo, text="Arquivo DCE de referência").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Label(topo, textvariable=self.dce_nome_var, width=42).grid(row=2, column=1, padx=8, sticky="w")
        ttk.Button(topo, text="Localizar arquivo .nii.gz", command=self.selecionar_dataset).grid(row=2, column=2, padx=5)
        topo.columnconfigure(1, weight=1)

        selecao = ttk.Frame(self.raiz, style="Card.TFrame", padding=14)
        selecao.pack(fill="x", padx=14, pady=6)
        ttk.Label(selecao, text="Paciente:").pack(side="left")
        self.paciente_var = tk.StringVar()
        self.combo = ttk.Combobox(selecao, textvariable=self.paciente_var, state="readonly", width=22)
        self.combo.pack(side="left", padx=8)
        self.combo.bind("<<ComboboxSelected>>", self._paciente_selecionado)
        ttk.Button(selecao, text="Carregar pacientes", command=self.carregar_pacientes).pack(side="left", padx=6)
        ttk.Button(selecao, text="Classificar e gerar Grad-CAM", style="Accent.TButton", command=self.classificar).pack(side="left", padx=6)
        ttk.Button(selecao, text="Salvar figura para o artigo", command=self.salvar_figura).pack(side="left", padx=6)
        controles = ttk.Frame(self.raiz, style="Card.TFrame", padding=14)
        controles.pack(fill="x", padx=14, pady=6)
        ttk.Label(controles, text="Fase exibida:").pack(side="left")
        self.fase_visualizacao = tk.StringVar(value="post_early")
        fases_combo = ttk.Combobox(
            controles, textvariable=self.fase_visualizacao,
            values=FASES, state="readonly", width=16,
        )
        fases_combo.pack(side="left", padx=6)
        fases_combo.bind("<<ComboboxSelected>>", self.atualizar_visualizacao)
        ttk.Label(controles, text="Corte Z:").pack(side="left", padx=(12, 0))
        self.corte_var = tk.DoubleVar(value=0)
        self.controle_corte = ttk.Scale(
            controles, from_=0, to=0, orient="horizontal",
            variable=self.corte_var, command=self.atualizar_visualizacao,
        )
        self.controle_corte.pack(side="left", fill="x", expand=True, padx=8)
        self.corte_info = tk.StringVar(value="Sem volume")
        ttk.Label(controles, textvariable=self.corte_info, width=12).pack(side="left")
        ttk.Label(controles, text="Explicar:").pack(side="left", padx=(10, 0))
        self.classe_gradcam = tk.StringVar(value="prevista")
        ttk.Combobox(
            controles, textvariable=self.classe_gradcam,
            values=("prevista", "pcr1"), state="readonly", width=11,
        ).pack(side="left", padx=5)

        self.resultado = tk.StringVar(value="Aguardando checkpoint.")
        painel_resultado = ttk.Frame(self.raiz, style="Card.TFrame", padding=14)
        painel_resultado.pack(fill="x", padx=14, pady=6)
        self.resultado = tk.StringVar(value="Aguardando checkpoint.")
        ttk.Label(painel_resultado, textvariable=self.resultado, font=("Segoe UI", 11, "bold"), wraplength=1280).pack(fill="x")
        self.status = tk.StringVar(value="")
        ttk.Label(painel_resultado, textvariable=self.status).pack(fill="x", pady=(5, 0))
        self.diagnostico_var = tk.StringVar(value="O Grad-CAM nao foi calculado.")
        ttk.Label(painel_resultado, textvariable=self.diagnostico_var, wraplength=1280).pack(fill="x", pady=(5, 0))
        self.area = ttk.Frame(self.raiz, style="Card.TFrame", padding=12)
        self.area.pack(fill="both", expand=True, padx=14, pady=(6, 14))

    def _limpar_visualizacao(self):
        self.volumes_visualizacao = {}
        self.mascara_volume = None
        self.paciente_visualizacao = None
        self.gradcam_atual = None
        self.corte_var.set(0)
        self.controle_corte.configure(to=0)
        self.corte_info.set("Sem volume")
        self.diagnostico_var.set("O Grad-CAM nao foi calculado.")
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.figura is not None:
            from matplotlib import pyplot as plt
            plt.close(self.figura)
            self.figura = None

    def selecionar_checkpoint(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar checkpoint treinado",
            filetypes=[("Checkpoint PyTorch", "*.pt")],
        )
        if caminho:
            self.caminho_var.set(caminho)
            self.checkpoint_nome_var.set(Path(caminho).name)

    def selecionar_csv(self):
        caminho = filedialog.askopenfilename(title="Selecionar CSV oficial", filetypes=[("CSV", "*.csv")])
        if caminho:
            self.csv_var.set(caminho)
            self.csv_nome_var.set(Path(caminho).name)
            self.pacientes = []
            self.combo.set("")
            self._limpar_visualizacao()

    def selecionar_dataset(self):
        caminho = filedialog.askopenfilename(
            title="Selecione um arquivo DCE NIfTI da pasta dce",
            filetypes=[("Volumes NIfTI", "*.nii.gz *.nii")],
        )
        if caminho:
            arquivo = Path(caminho).resolve()
            if arquivo.parent.name.lower() != "dce":
                messagebox.showerror("Arquivo invalido", "Selecione um volume dentro da pasta dce do dataset.")
                return
            raiz_dataset = arquivo.parent.parent
            if not (raiz_dataset / "mask").is_dir():
                messagebox.showerror("Dataset invalido", "A pasta mask nao foi encontrada ao lado de dce.")
                return
            self.dataset_var.set(str(raiz_dataset))
            self.dce_nome_var.set(arquivo.name)
            self.pacientes = []
            self.combo.set("")
            self._limpar_visualizacao()

    def carregar_checkpoint(self):
        try:
            caminho = Path(self.caminho_var.get()).expanduser().resolve()
            modelo, dados = carregar_modelo_checkpoint(caminho, self.dispositivo)
            self.modelo, self.checkpoint, self.caminho_checkpoint = modelo, dados, caminho
            self.gradcam_atual = None
            conf = dados["configuracao"]
            self.resultado.set(
                f"Modelo: {conf['arquitetura']} | Fases: {conf['configuracao_temporal']} "
                f"| Corte: {conf['estrategia_corte']} "
                f"| Limiar: {float(dados['selecao_validacao']['limiar']):.8f}"
            )
            self.status.set(
                f"Checkpoint validado com strict=True. Dispositivo: {self.dispositivo}. Nenhum treinamento executado."
            )
            messagebox.showinfo(
                "Checkpoint validado",
                "Pesos e metadados compativeis. O modelo foi carregado em modo de inferencia.",
            )
            self.atualizar_visualizacao()
        except Exception as exc:
            self.modelo = self.checkpoint = None
            self.gradcam_atual = None
            self.resultado.set("Checkpoint nao carregado.")
            messagebox.showerror("Falha ao validar checkpoint", str(exc))

    def carregar_pacientes(self):
        try:
            global DATASET_PADRAO
            DATASET_PADRAO = Path(self.dataset_var.get()).expanduser().resolve()
            self._limpar_visualizacao()
            self.pacientes = carregar_metadados(Path(self.csv_var.get()).expanduser().resolve())
            self.combo["values"] = [p.identificador for p in self.pacientes]
            if self.pacientes:
                self.combo.current(0)
                self._paciente_selecionado()
            self.status.set(f"{len(self.pacientes)} pacientes carregados do CSV.")
        except Exception as exc:
            messagebox.showerror("Falha ao carregar pacientes", str(exc))

    def _paciente_selecionado(self, _evento=None):
        pid = self.paciente_var.get()
        paciente = next((p for p in self.pacientes if p.identificador == pid), None)
        if paciente is None:
            return
        try:
            self._limpar_visualizacao()
            mascara, affine_mascara = ler_volume(paciente.caminho_mascara)
            volumes = {}
            for fase, caminho in zip(FASES, paciente.caminhos_dce):
                volume, affine = ler_volume(caminho)
                if volume.shape != mascara.shape or not np.allclose(
                    affine, affine_mascara, rtol=1e-4, atol=1e-3
                ):
                    raise ValueError(f"DCE e mascara desalinhados: {pid} / {fase}")
                volumes[fase] = volume
            self.volumes_visualizacao = volumes
            self.mascara_volume = mascara
            self.paciente_visualizacao = pid
            self.controle_corte.configure(to=mascara.shape[0] - 1)
            self.corte_var.set(float(mascara.shape[0] // 2))
            self.atualizar_visualizacao()
        except Exception as exc:
            self._limpar_visualizacao()
            messagebox.showerror("Falha ao abrir volumes", str(exc))

    def atualizar_visualizacao(self, _evento=None):
        if self.mascara_volume is None or self.paciente_visualizacao != self.paciente_var.get():
            return
        z = int(round(self.corte_var.get()))
        z = max(0, min(z, self.mascara_volume.shape[0] - 1))
        self.corte_info.set(f"{z}/{self.mascara_volume.shape[0]-1}")
        fase = self.fase_visualizacao.get()
        if fase not in self.volumes_visualizacao:
            fase = "post_early"
        original = normalizar_percentis(self.volumes_visualizacao[fase][z][None])[0]
        mascara = (self.mascara_volume[z] > 0).astype(np.uint8)
        dados = self.gradcam_atual
        if dados is not None and (
            dados["z"] != z or dados["paciente"] != self.paciente_var.get()
        ):
            dados = None
        self._mostrar_figuras(
            original, mascara,
            dados["mapa"] if dados else None,
            fase, dados["diagnostico"] if dados else None,
        )
        if dados is None:
            self.diagnostico_var.set(
                "Visualizacao do corte Z. O Grad-CAM so e exibido no corte efetivamente classificado."
            )
        else:
            self.diagnostico_var.set(dados["resumo"])

    def classificar(self):
        if self.modelo is None or self.checkpoint is None:
            messagebox.showwarning(
                "Checkpoint necessario", "Selecione e valide um checkpoint antes da inferencia."
            )
            return
        pid = self.paciente_var.get()
        paciente = next((p for p in self.pacientes if p.identificador == pid), None)
        if paciente is None:
            messagebox.showwarning("Paciente necessario", "Carregue o CSV e selecione um paciente.")
            return
        try:
            conf, prep = self.checkpoint["configuracao"], self.checkpoint["preprocessamento"]
            imagem, mascara, z = preparar_paciente(paciente, conf, prep)
            entrada = imagem.unsqueeze(0).to(self.dispositivo)
            limiar = float(self.checkpoint["selecao_validacao"]["limiar"])
            mapa, prob, diagnostico = gerar_gradcam(
                self.modelo, entrada, conf["arquitetura"],
                classe_alvo=self.classe_gradcam.get(), limiar=limiar,
            )
            previsto = int(prob >= limiar)
            resumo = resumir_relevancia_gradcam(mapa, mascara, diagnostico)
            self.gradcam_atual = {
                "paciente": pid, "z": z, "mapa": mapa,
                "diagnostico": diagnostico, "resumo": resumo,
            }
            self.resultado.set(
                f"{pid} | pCR real={paciente.pcr} | probabilidade pCR=1: {prob:.6f} "
                f"| limiar: {limiar:.8f} | previsto: pCR={previsto} | corte Z={z}"
            )
            self.corte_var.set(float(z))
            self.atualizar_visualizacao()
            self.status.set(
                "Inferencia por checkpoint, sem retreino. Grad-CAM no corte selecionado pelo protocolo."
            )
        except Exception as exc:
            messagebox.showerror("Falha na inferencia", str(exc))
            self.status.set("Nao foi possivel concluir a inferencia.")

    def _mostrar_figuras(self, original, mascara, mapa, fase, diagnostico=None):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as exc:
            raise ImportError("Instale matplotlib para visualizar imagem, mascara e Grad-CAM.") from exc
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.figura is not None:
            from matplotlib import pyplot as plt
            plt.close(self.figura)
        figura = Figure(figsize=(13, 4), dpi=100)
        eixos = figura.subplots(1, 4)
        eixos[0].imshow(original, cmap="gray", vmin=0, vmax=1)
        eixos[0].set_title(f"DCE-MRI ({fase})")
        eixos[1].imshow(original, cmap="gray", vmin=0, vmax=1)
        if mascara.any():
            eixos[1].contour(mascara, levels=[0.5], colors=["lime"], linewidths=1.2)
        eixos[1].set_title("Mascara sobre a imagem")
        if mapa is None:
            for eixo in eixos[2:]:
                eixo.text(
                    0.5, 0.5, "Classifique para\nver o Grad-CAM",
                    ha="center", va="center", transform=eixo.transAxes,
                )
            eixos[2].set_title("Grad-CAM")
            eixos[3].set_title("Grad-CAM + tumor")
        else:
            for eixo in eixos[2:]:
                eixo.imshow(original, cmap="gray", vmin=0, vmax=1)
                eixo.imshow(
                    mapa, cmap="turbo", vmin=0, vmax=1,
                    alpha=np.clip(mapa * 0.65, 0, 0.65),
                )
            if mascara.any():
                eixos[3].contour(mascara, levels=[0.5], colors=["lime"], linewidths=1.2)
            classe = diagnostico["classe_explicada"] if diagnostico else 1
            eixos[2].set_title(f"Grad-CAM pCR={classe}")
            eixos[3].set_title("Grad-CAM + tumor")
        for eixo in eixos:
            eixo.axis("off")
        figura.tight_layout()
        figura.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.90, bottom=0.02)
        self.figura = figura
        self.canvas = FigureCanvasTkAgg(figura, master=self.area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def salvar_figura(self):
        if self.figura is None or self.gradcam_atual is None:
            messagebox.showwarning("Figura indisponivel", "Classifique um paciente antes de salvar a figura.")
            return
        paciente = self.paciente_visualizacao or "paciente"
        caminho = filedialog.asksaveasfilename(
            title="Salvar figura para o artigo",
            initialfile=f"gradcam_{paciente}.png",
            defaultextension=".png",
            filetypes=[("Imagem PNG", "*.png"), ("Imagem PDF", "*.pdf"), ("Imagem SVG", "*.svg")],
        )
        if not caminho:
            return
        try:
            self.figura.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
            self.status.set(f"Figura exportada em alta resolucao: {Path(caminho).name}")
            messagebox.showinfo("Figura salva", "Figura exportada. Confira a legenda e a identificação do caso antes de inserir no artigo.")
        except Exception as exc:
            messagebox.showerror("Falha ao salvar figura", str(exc))


def resumir_relevancia_gradcam(mapa: np.ndarray, mascara: np.ndarray, diagnostico: dict) -> str:
    if mapa.shape != mascara.shape or mapa.ndim != 2:
        raise ValueError("Mapa e mascara devem ter o mesmo formato 2D.")
    if not np.isfinite(mapa).all():
        raise ValueError("Mapa Grad-CAM contem valores invalidos.")
    proporcao_area = float(np.mean(mascara > 0))
    soma = float(np.sum(mapa))
    fracao_tumor = float(np.sum(mapa[mascara > 0]) / soma) if soma > 1e-12 else 0.0
    aviso = " [MAPA NULO]" if diagnostico["mapa_nulo"] else ""
    return (
        f"Classe explicada pCR={diagnostico['classe_explicada']} | "
        f"relevancia no tumor: {fracao_tumor:.1%} | area tumoral: {proporcao_area:.1%} "
        f"| resolucao CAM: {diagnostico['resolucao_nativa']} "
        f"| maximo bruto: {diagnostico['maximo_bruto']:.3g}{aviso}"
    )


class DatasetTreinoDCE(torch.utils.data.Dataset):
    ANGULOS = (-20, -10, 0, 10, 20)

    def __init__(self, pacientes, configuracao, prep, augmentar):
        from scipy.ndimage import rotate
        self._rotate = rotate
        self.amostras = []
        self.augmentar = augmentar
        for paciente in pacientes:
            imagem, _, _ = preparar_paciente(paciente, configuracao, prep)
            self.amostras.append((imagem, float(paciente.pcr), paciente.identificador))
        if not self.amostras:
            raise ValueError("Conjunto vazio para treinamento ou validacao.")

    def __len__(self):
        return len(self.amostras) * (10 if self.augmentar else 1)

    def __getitem__(self, indice):
        factor = 10 if self.augmentar else 1
        imagem, rotulo, pid = self.amostras[indice // factor]
        if not self.augmentar:
            return imagem, torch.tensor(rotulo, dtype=torch.float32), pid
        variante = indice % 10
        angulo = self.ANGULOS[variante % 5]
        dados = np.stack([
            self._rotate(
                canal, angulo, reshape=False, order=1,
                mode="constant", cval=0.0,
            ) for canal in imagem.numpy()
        ], axis=0)
        if variante >= 5:
            dados = np.flip(dados, axis=2)
        return (
            torch.from_numpy(np.ascontiguousarray(dados, dtype=np.float32)),
            torch.tensor(rotulo, dtype=torch.float32),
            pid,
        )


def auc_roc_pacientes(rotulos: np.ndarray, probabilidades: np.ndarray) -> float:
    positivos = probabilidades[rotulos == 1]
    negativos = probabilidades[rotulos == 0]
    if len(positivos) == 0 or len(negativos) == 0:
        raise ValueError("AUC requer ambas as classes.")
    comparacoes = positivos[:, None] - negativos[None, :]
    return float(((comparacoes > 0) + 0.5 * (comparacoes == 0)).mean())


def resumir_predicoes_validacao(registros: list[tuple[str, int, float]]):
    agrupados = {}
    for pid, rotulo, prob in registros:
        if pid in agrupados and agrupados[pid][0] != rotulo:
            raise ValueError("Rotulos conflitantes para o mesmo paciente.")
        if pid not in agrupados:
            agrupados[pid] = (rotulo, [])
        agrupados[pid][1].append(prob)
    rotulos = np.asarray([v[0] for v in agrupados.values()], dtype=int)
    probabilidades = np.asarray(
        [np.mean(v[1]) for v in agrupados.values()], dtype=float
    )
    auc = auc_roc_pacientes(rotulos, probabilidades)
    melhor = None
    for limiar in np.arange(0.05, 1., 0.05):
        previsto = (probabilidades >= limiar).astype(int)
        tp = int(np.sum((rotulos == 1) & (previsto == 1)))
        tn = int(np.sum((rotulos == 0) & (previsto == 0)))
        fp = int(np.sum((rotulos == 0) & (previsto == 1)))
        fn = int(np.sum((rotulos == 1) & (previsto == 0)))
        sens = tp / (tp + fn) if tp + fn else 0.
        esp = tn / (tn + fp) if tn + fp else 0.
        bacc = (sens + esp) / 2.
        precisao = tp / (tp + fp) if tp + fp else 0.
        f1 = 2 * precisao * sens / (precisao + sens) if precisao + sens else 0.
        metrica = {
            "auc_roc": auc, "acuracia_balanceada": bacc,
            "sensibilidade": sens, "especificidade": esp,
            "precisao": precisao, "acuracia": (tp + tn) / len(rotulos),
            "f1": f1, "tn": tn, "tp": tp, "fp": fp, "fn": fn,
            "limiar": float(round(limiar, 5)), "numero_pacientes": len(rotulos),
        }
        if melhor is None or (bacc, f1) > (
            melhor["acuracia_balanceada"], melhor["f1"]
        ):
            melhor = metrica
    return melhor


def configurar_treino_modelo(modelo: nn.Module, arquitetura: str):
    if arquitetura == "cnn":
        return
    for parametro in modelo.parameters():
        parametro.requires_grad = False
    if arquitetura == "resnet50":
        for parametro in modelo.layer4.parameters():
            parametro.requires_grad = True
        for parametro in modelo.fc.parameters():
            parametro.requires_grad = True
    else:
        for parametro in modelo.classifier.parameters():
            parametro.requires_grad = True
    for modulo in modelo.modules():
        if isinstance(modulo, nn.modules.batchnorm._BatchNorm):
            modulo.eval()
            for parametro in modulo.parameters():
                parametro.requires_grad = False


def executar_epoca_treino(modelo, carregador, criterio, otimizador, dispositivo, treino):
    modelo.train(treino)
    if treino:
        for modulo in modelo.modules():
            if isinstance(modulo, nn.modules.batchnorm._BatchNorm):
                modulo.eval()
    soma_loss, contagem = 0., 0
    registros = []
    contexto = torch.enable_grad() if treino else torch.no_grad()
    with contexto:
        for imagens, rotulos, ids in carregador:
            imagens = imagens.to(dispositivo)
            rotulos = rotulos.to(dispositivo)
            if treino:
                otimizador.zero_grad(set_to_none=True)
            logits = modelo(imagens).reshape(-1)
            loss = criterio(logits, rotulos)
            if treino:
                loss.backward()
                otimizador.step()
            soma_loss += float(loss.detach()) * len(rotulos)
            contagem += len(rotulos)
            if not treino:
                probabilidades = torch.sigmoid(logits).detach().cpu().tolist()
                registros.extend(
                    (str(pid), int(y), float(prob))
                    for pid, y, prob in zip(ids, rotulos.cpu().tolist(), probabilidades)
                )
    return soma_loss / max(contagem, 1), registros


def treinar_configuracao(pacientes, arquitetura, fase, corte, args, dispositivo):
    treino = [p for p in pacientes if p.particao_codigo == 0]
    validacao = [p for p in pacientes if p.particao_codigo == 2]
    teste = [p for p in pacientes if p.particao_codigo == 1]
    if (len(treino), len(validacao), len(teste)) != (104, 33, 35):
        raise ValueError("Particoes oficiais esperadas: 104 treino, 33 validacao, 35 teste.")
    if set(p.identificador for p in treino) & set(p.identificador for p in validacao):
        raise ValueError("Paciente em mais de uma particao.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    canais = 3 if fase == "tres" else 1
    conf = {
        "arquitetura": arquitetura, "canais_entrada": canais,
        "configuracao_temporal": fase, "estrategia_corte": corte,
        "seed": args.seed, "epocas": args.epocas, "tamanho_lote": args.batch,
    }
    prep = {
        "fases": FASES_TRES if fase == "tres" else [fase],
        "normalizacao": {"tipo": "percentil_1_99_conjunto"},
        "corte": corte, "representacao_entrada": "fases",
        "mascara_como_filtro": False, "recorte_roi": False,
        "augmentation": {
            "angulos_graus": list(DatasetTreinoDCE.ANGULOS),
            "espelhamento": [False, True], "variacoes_por_paciente": 10,
            "teste_aumentado": False,
        },
        "particoes": {"treino": 0, "validacao": 2, "teste": 1},
    }
    conjunto_treino = DatasetTreinoDCE(treino, conf, prep, augmentar=True)
    conjunto_validacao = DatasetTreinoDCE(validacao, conf, prep, augmentar=True)
    gerador = torch.Generator().manual_seed(args.seed)
    loader_treino = torch.utils.data.DataLoader(
        conjunto_treino, batch_size=args.batch, shuffle=True, generator=gerador,
    )
    loader_validacao = torch.utils.data.DataLoader(
        conjunto_validacao, batch_size=args.batch, shuffle=False,
    )
    modelo = criar_modelo(arquitetura, canais, pretreinado=arquitetura != "cnn")
    configurar_treino_modelo(modelo, arquitetura)
    modelo.to(dispositivo)
    params = [p for p in modelo.parameters() if p.requires_grad]
    taxa = args.lr if args.lr is not None else (
        1e-3 if arquitetura == "cnn" else 1e-4
    )
    otimizador = torch.optim.Adam(params, lr=taxa, weight_decay=1e-4)
    negativos = sum(p.pcr == 0 for p in treino)
    positivos = sum(p.pcr == 1 for p in treino)
    if not positivos or not negativos:
        raise ValueError("Treino exige as duas classes.")
    peso_positivo = (
        1.0 if arquitetura == "resnet50" else (negativos / positivos) ** 0.5
    )
    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([peso_positivo], device=dispositivo)
    )
    melhor_auc = -1.
    epoca_melhor, sem_melhora = 0, 0
    melhor_estado, melhor_metricas = None, None
    historico = []
    for epoca in range(1, args.epocas + 1):
        inicio = time.perf_counter()
        loss_treino, _ = executar_epoca_treino(
            modelo, loader_treino, criterio, otimizador, dispositivo, treino=True
        )
        loss_val, previsoes = executar_epoca_treino(
            modelo, loader_validacao, criterio, None, dispositivo, treino=False
        )
        metricas = resumir_predicoes_validacao(previsoes)
        atual = {
            "epoca": epoca, "loss_treino": loss_treino,
            "loss_validacao": loss_val, "auc_roc": metricas["auc_roc"],
            "tempo_segundos": time.perf_counter() - inicio,
        }
        historico.append(atual)
        print(
            f"{arquitetura}/{fase}/{corte} epoca {epoca}: "
            f"loss={loss_treino:.4f}/{loss_val:.4f} "
            f"AUC val={metricas['auc_roc']:.4f}", flush=True,
        )
        if metricas["auc_roc"] > melhor_auc + 1e-9:
            melhor_auc, epoca_melhor, sem_melhora = metricas["auc_roc"], epoca, 0
            melhor_metricas = metricas
            melhor_estado = {
                k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()
            }
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                break
    if melhor_estado is None:
        raise RuntimeError("Nao foi possivel selecionar um checkpoint pela validacao.")
    nome = f"{arquitetura}__{fase}__{corte}__seed-{args.seed}__monolito"
    args.saida_checkpoints.mkdir(parents=True, exist_ok=True)
    destino = args.saida_checkpoints / f"{nome}.pt"
    conf.update({"taxa_aprendizado": taxa, "pos_weight": peso_positivo})
    torch.save({
        "formato_checkpoint": 1,
        "model_state_dict": melhor_estado,
        "configuracao": conf,
        "preprocessamento": prep,
        "selecao_validacao": {
            "limiar": melhor_metricas["limiar"], "metricas": melhor_metricas,
        },
        "epoca": epoca_melhor,
    }, destino)
    with (destino.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(historico[0]))
        writer.writeheader()
        writer.writerows(historico)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    figura = Figure(figsize=(7, 4), dpi=125)
    eixo = figura.subplots(1, 1)
    eixo.plot(
        [e["epoca"] for e in historico],
        [e["loss_treino"] for e in historico], label="Treino",
    )
    eixo.plot(
        [e["epoca"] for e in historico],
        [e["loss_validacao"] for e in historico], label="Validacao",
    )
    eixo.set(xlabel="Epoca", ylabel="BCE loss", title="Convergencia")
    eixo.legend()
    figura.tight_layout()
    figura.savefig(destino.with_name(destino.stem + "__loss.png"))
    with (destino.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump({
            "arquitetura": arquitetura, "fase": fase, "corte": corte,
            "epoca_melhor": epoca_melhor, "metricas_validacao": melhor_metricas,
            "pacientes": {"treino": 104, "validacao": 33, "teste_excluido": 35},
            "observacao": "Checkpoint selecionado sem consultar o teste.",
        }, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint salvo: {destino} | AUC validacao={melhor_auc:.4f}")
    return destino


def executar_treinamento_cli():
    global DATASET_PADRAO
    parser = argparse.ArgumentParser(description="Treinar modelos do trabalho PAI")
    parser.add_argument("--treinar", action="store_true", required=True)
    parser.add_argument("--grade", action="store_true", help="Executar as 12 configuracoes")
    parser.add_argument("--arquitetura", choices=sorted(ARQUITETURAS), default="cnn")
    parser.add_argument("--fases", choices=("post_early", "tres"), default="post_early")
    parser.add_argument("--corte", choices=("central", "maior_tumor"), default="central")
    parser.add_argument("--csv", type=Path, default=CSV_PADRAO)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_PADRAO)
    parser.add_argument("--saida-checkpoints", type=Path, default=RAIZ / "checkpoints/monolito")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--paciencia", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epocas < 1 or args.batch < 1 or args.paciencia < 1:
        parser.error("Epocas, batch e paciencia devem ser positivos.")
    if args.lr is not None and args.lr <= 0:
        parser.error("A taxa de aprendizado deve ser positiva.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponivel neste ambiente.")
    dispositivo = torch.device(
        "cuda" if (args.device == "cuda" or args.device == "auto"
                   and torch.cuda.is_available()) else "cpu"
    )
    DATASET_PADRAO = args.dataset_dir.expanduser().resolve()
    pacientes = carregar_metadados(args.csv.expanduser().resolve())
    configuracoes = (
        [(arq, fase, corte)
         for arq in ("cnn", "resnet50", "mobilenetv2")
         for fase in ("post_early", "tres")
         for corte in ("central", "maior_tumor")]
        if args.grade else [(args.arquitetura, args.fases, args.corte)]
    )
    for arq, fase, corte in configuracoes:
        treinar_configuracao(pacientes, arq, fase, corte, args, dispositivo)


def main():
    import sys
    if "--treinar" in sys.argv[1:]:
        executar_treinamento_cli()
        return
    if len(sys.argv) > 1:
        raise SystemExit("Use --treinar para o modo de treinamento; GUI nao recebe argumentos.")
    raiz = tk.Tk()
    AplicacaoPAI(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
