"""Entrega integrada PAI - Grupo G10.

Arquivo monolitico: interface Tkinter, leitura DCE-MRI, validacao segura de
checkpoint, inferencia binaria e Grad-CAM. Nao realiza treinamento.
Requer: torch, torchvision, numpy, nibabel, matplotlib e Tkinter.
"""
from __future__ import annotations

import csv
import os
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


def carregar_metadados(caminho: Path) -> list[Paciente]:
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV de metadados nao encontrado: {caminho}")
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        obrigatorias = {"pid", "pCR", "dataset", "pre", "post_early", "post_late"}
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
            indices = [
                converter_inteiro(linha[f], f)
                for f in FASES
            ]

        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Metadados invalidos para {pid}: {exc}"
            ) from exc
        if rotulo not in (0, 1) or min(indices) < 0:
            raise ValueError(f"Rotulo ou indices temporais invalidos para {pid}")
        prefixo = f"{pid}_spy1_vis1"
        dce = tuple(DATASET_PADRAO / "dce" / f"{prefixo}_dce_aqc_{idx}.nii.gz" for idx in indices)
        mascara = DATASET_PADRAO / "mask" / f"{prefixo}_mask.nii.gz"
        pacientes.append(Paciente(pid, rotulo, dce, mascara))
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


def criar_modelo(arquitetura: str, canais: int):
    if arquitetura == "cnn":
        return CNNSimples(canais)
    try:
        from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
    except ImportError as exc:
        raise ImportError("torchvision compativel com PyTorch nao esta instalado.") from exc
    if arquitetura == "resnet50":
        modelo = resnet50(weights=None)
        modelo.conv1 = adaptar_primeira_convolucao(modelo.conv1, canais)
        modelo.fc = nn.Linear(modelo.fc.in_features, 1)
        return modelo
    if arquitetura == "mobilenetv2":
        modelo = mobilenet_v2(weights=None)
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


def gerar_gradcam(modelo: nn.Module, imagem: torch.Tensor, arquitetura: str):
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
            logit.backward()
        if not ativacoes or not gradientes:
            raise RuntimeError("Ativacoes/gradientes Grad-CAM nao foram capturados.")
        pesos = gradientes[-1].mean(dim=(2, 3), keepdim=True)
        mapa = F.relu((pesos * ativacoes[-1]).sum(dim=1, keepdim=True))
        mapa = F.interpolate(mapa, size=imagem.shape[-2:], mode="bilinear", align_corners=False)
        mapa = mapa - mapa.amin(dim=(2, 3), keepdim=True)
        mapa = mapa / mapa.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        return mapa[0, 0].detach().cpu().numpy(), float(torch.sigmoid(logit.detach()).cpu())
    finally:
        handle.remove()
        modelo.zero_grad(set_to_none=True)
        modelo.train(estava_treinando)


class AplicacaoPAI:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("PAI — Classificacao pCR e Grad-CAM | Grupo G10")
        self.raiz.geometry("1200x850")
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = None
        self.checkpoint = None
        self.caminho_checkpoint = None
        self.pacientes = []
        self.figura = None
        self.canvas = None
        self._construir_interface()
        self.status.set(f"Dispositivo: {self.dispositivo}. Carregue um checkpoint .pt compativel.")

    def _construir_interface(self):
        topo = ttk.Frame(self.raiz, padding=12)
        topo.pack(fill="x")
        ttk.Label(topo, text="Checkpoint treinado (.pt):").grid(row=0, column=0, sticky="w")
        self.caminho_var = tk.StringVar()
        ttk.Entry(topo, textvariable=self.caminho_var, width=90).grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Button(topo, text="Selecionar...", command=self.selecionar_checkpoint).grid(row=0, column=2)
        ttk.Button(topo, text="Validar e carregar", command=self.carregar_checkpoint).grid(row=0, column=3, padx=6)
        ttk.Label(topo, text="CSV:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.csv_var = tk.StringVar(value=str(CSV_PADRAO))
        ttk.Entry(topo, textvariable=self.csv_var, width=90).grid(row=1, column=1, padx=8, sticky="ew", pady=(10, 0))
        ttk.Button(topo, text="Escolher CSV", command=self.selecionar_csv).grid(row=1, column=2, pady=(10, 0))
        ttk.Label(topo, text="Pasta do dataset:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.dataset_var = tk.StringVar(value=str(DATASET_PADRAO))
        ttk.Entry(topo, textvariable=self.dataset_var, width=90).grid(row=2, column=1, padx=8, sticky="ew", pady=(10, 0))
        ttk.Button(topo, text="Escolher pasta", command=self.selecionar_dataset).grid(row=2, column=2, pady=(10, 0))
        topo.columnconfigure(1, weight=1)

        selecao = ttk.Frame(self.raiz, padding=(12, 0, 12, 8))
        selecao.pack(fill="x")
        ttk.Label(selecao, text="Paciente:").pack(side="left")
        self.paciente_var = tk.StringVar()
        self.combo = ttk.Combobox(selecao, textvariable=self.paciente_var, state="readonly", width=22)
        self.combo.pack(side="left", padx=8)
        ttk.Button(selecao, text="Carregar pacientes", command=self.carregar_pacientes).pack(side="left")
        ttk.Button(selecao, text="Classificar e gerar Grad-CAM", command=self.classificar).pack(side="left", padx=8)

        self.resultado = tk.StringVar(value="Aguardando checkpoint.")
        ttk.Label(self.raiz, textvariable=self.resultado, padding=12, font=("Segoe UI", 11, "bold")).pack(fill="x")
        self.status = tk.StringVar(value="")
        ttk.Label(self.raiz, textvariable=self.status, padding=(12, 0)).pack(fill="x")
        self.area = ttk.Frame(self.raiz, padding=8)
        self.area.pack(fill="both", expand=True)

    def selecionar_checkpoint(self):
        caminho = filedialog.askopenfilename(title="Selecionar checkpoint treinado", filetypes=[("Checkpoint PyTorch", "*.pt")])
        if caminho:
            self.caminho_var.set(caminho)

    def selecionar_csv(self):
        caminho = filedialog.askopenfilename(title="Selecionar CSV oficial", filetypes=[("CSV", "*.csv")])
        if caminho:
            self.csv_var.set(caminho)
            self.pacientes = []

    def selecionar_dataset(self):
        caminho = filedialog.askdirectory(title="Selecionar pasta contendo dce/ e mask/")
        if caminho:
            self.dataset_var.set(caminho)
            self.pacientes = []

    def carregar_checkpoint(self):
        try:
            caminho = Path(self.caminho_var.get()).expanduser().resolve()
            modelo, dados = carregar_modelo_checkpoint(caminho, self.dispositivo)
            self.modelo, self.checkpoint, self.caminho_checkpoint = modelo, dados, caminho
            conf = dados["configuracao"]
            self.resultado.set(f"Modelo: {conf['arquitetura']} | Fases: {conf['configuracao_temporal']} | Corte: {conf['estrategia_corte']} | Limiar: {float(dados['selecao_validacao']['limiar']):.8f}")
            self.status.set(f"Checkpoint validado com strict=True. Dispositivo: {self.dispositivo}. Nenhum treinamento executado.")
            messagebox.showinfo("Checkpoint validado", "Pesos e metadados compativeis. O modelo foi carregado em modo de inferencia.")
        except Exception as exc:
            self.modelo = self.checkpoint = None
            self.resultado.set("Checkpoint nao carregado.")
            messagebox.showerror("Falha ao validar checkpoint", str(exc))

    def carregar_pacientes(self):
        try:
            global DATASET_PADRAO
            DATASET_PADRAO = Path(self.dataset_var.get()).expanduser().resolve()
            self.pacientes = carregar_metadados(Path(self.csv_var.get()).expanduser().resolve())
            self.combo["values"] = [p.identificador for p in self.pacientes]
            if self.pacientes:
                self.combo.current(0)
            self.status.set(f"{len(self.pacientes)} pacientes carregados do CSV.")
        except Exception as exc:
            messagebox.showerror("Falha ao carregar pacientes", str(exc))

    def classificar(self):
        if self.modelo is None or self.checkpoint is None:
            messagebox.showwarning("Checkpoint necessario", "Selecione e valide um checkpoint antes da inferencia.")
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
            mapa, prob = gerar_gradcam(self.modelo, entrada, conf["arquitetura"])
            limiar = float(self.checkpoint["selecao_validacao"]["limiar"])
            previsto = int(prob >= limiar)
            self.resultado.set(f"{pid} | pCR real={paciente.pcr} | probabilidade pCR=1: {prob:.6f} | limiar: {limiar:.8f} | previsto: pCR={previsto} | corte Z={z}")
            self._mostrar_figuras(imagem.numpy(), mascara, mapa, conf["configuracao_temporal"])
            self.status.set("Inferencia concluida sem retreinamento. Grad-CAM e mascara correspondem ao mesmo paciente/corte.")
        except Exception as exc:
            messagebox.showerror("Falha na inferencia", str(exc))
            self.status.set("Nao foi possivel concluir a inferencia.")

    def _mostrar_figuras(self, imagem, mascara, mapa, fase):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib import cm
        except ImportError as exc:
            raise ImportError("Instale matplotlib para visualizar imagem, mascara e Grad-CAM.") from exc
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        figura = Figure(figsize=(13, 4.5), dpi=100)
        eixos = figura.subplots(1, 3)
        if imagem.shape[0] == 1:
            original = imagem[0]
            eixos[0].imshow(original, cmap="gray")
            base = np.clip(original, 0, 1)
            rgb = np.repeat(base[..., None], 3, axis=2)
        else:
            rgb = np.moveaxis(imagem, 0, -1)
            rgb = np.clip((rgb - rgb.min()) / max(float(rgb.max() - rgb.min()), 1e-8), 0, 1)
            eixos[0].imshow(rgb)
        eixos[0].set_title(f"Imagem original ({fase})")
        calor = cm.jet(np.clip(mapa, 0, 1))[..., :3]
        sobreposicao = np.clip(0.60 * rgb + 0.40 * calor, 0, 1)
        eixos[1].imshow(sobreposicao)
        eixos[1].set_title("Grad-CAM — relevancia da classe pCR=1")
        eixos[2].imshow(mascara, cmap="gray")
        eixos[2].set_title("Mascara tumoral — mesmo corte")
        for eixo in eixos:
            eixo.axis("off")
        figura.tight_layout()
        self.figura = figura
        self.canvas = FigureCanvasTkAgg(figura, master=self.area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


def main():
    raiz = tk.Tk()
    AplicacaoPAI(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
