import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# Nomes e Matrículas do Grupo
# COMPONENTE 1: Gabriel Chaves Mendes - Matrícula: XXXXXX - Curso/Campus: Ciência da Computação (Coração Eucarístico)
# COMPONENTE 2: [NOME] - Matrícula: [MATRICULA]
# COMPONENTE 3: [NOME] - Matrícula: [MATRICULA]
# COMPONENTE 4: [NOME] - Matrícula: [MATRICULA]

# =====================================================================
# BLOCO 1: PRÉ-PROCESSAMENTO, AUGMENTATION E LETRA C
# =====================================================================

def analisar_balanceamento(caminho_csv):
    """
    Função (Letra B): Lê o CSV oficial do professor para checar a proporção de pacientes.
    - Conta quantos têm pCR Ausente (Classe 0) e Presente (Classe 1).
    - Se a diferença passar de 20%, gera um alerta matemático no terminal recomendando
      o uso do hiperparâmetro 'pos_weight' na rede neural para penalizar a classe majoritária.
    """
    try:
        df = pd.read_csv(caminho_csv)
        pcr_col = [c for c in df.columns if 'pCR' in c or 'pcr' in c.lower()][0]
        df_validos = df.dropna(subset=[pcr_col])
        contagem = df_validos[pcr_col].value_counts()
        total = len(df_validos)
        c0, c1 = contagem.get(0, 0), contagem.get(1, 0)

        print("\n" + "=" * 50)
        print("  📊 RELATÓRIO DO CSV: ANÁLISE DE DESBALANCEAMENTO")
        print("=" * 50)
        print(pd.DataFrame({
            "Classe (pCR)": ["Ausente (0)", "Presente (1)"],
            "Qtd. Pacientes": [c0, c1],
            "Proporção": [f"{(c0 / total) * 100:.2f}%", f"{(c1 / total) * 100:.2f}%"]
        }).to_string(index=False))

        if abs(c0 - c1) / total > 0.2:
            print(f"\n[ALERTA] Desbalanceamento detectado! Usando pos_weight no treino.")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")


def extrair_2d_estrategia(volume_3d, estrategia="central"):
    """
    Função (Letra C): As redes neurais não leem volumes 3D brutos para não estourar memória.
    Esta função pega o cubo tridimensional da ressonância e o "amassa" em uma folha plana 2D:
    - 'central': Pega a fatia do exato meio da profundidade (eixo Z).
    - 'mip': Maximum Intensity Projection, pega o pixel mais brilhante da profundidade.
    """
    if estrategia == "central":
        return volume_3d[:, :, volume_3d.shape[2] // 2]
    return np.max(volume_3d, axis=2)


# =====================================================================
# BLOCO 2: DATASET PYTORCH
# =====================================================================

class DatasetCancerMama(Dataset):
    """
    Classe PyTorch (Dataloader): O "motorzinho" que alimenta a Rede Neural.
    Ao invés de carregar todas as 2.000 imagens pesadas na memória RAM ao mesmo tempo,
    esta classe lê o CSV e vai carregando o paciente do HD um por um, aplicando a extração
    2D (Letra C) em tempo real, enviando pequenos blocos (Lotes) para a placa de vídeo.
    """

    def __init__(self, df, diretorio_dce, estrategia_2d="central"):
        pcr_col = [c for c in df.columns if 'pCR' in c or 'pcr' in c.lower()][0]
        self.df = df.dropna(subset=[pcr_col]).reset_index(drop=True)
        self.diretorio = diretorio_dce
        self.estrategia_2d = estrategia_2d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        paciente_id = linha['pid']
        classe_pcr = np.float32(linha['pCR'])
        nome_base = f"{paciente_id}_spy1_vis1_dce_aqc_1.nii.gz"  # Fase de pós-contraste precoce

        try:
            vol = nib.load(os.path.join(self.diretorio, nome_base)).get_fdata()
            img_2d = extrair_2d_estrategia(vol, self.estrategia_2d)
            tensor_img = torch.from_numpy(img_2d).unsqueeze(0).float()
            return tensor_img, torch.tensor([classe_pcr])
        except:
            return torch.zeros((1, 256, 256)), torch.tensor([classe_pcr])


# =====================================================================
# BLOCO 3: ARQUITETURA DA REDE E TREINAMENTO (Letra D)
# =====================================================================

class CNNCancerMama(nn.Module):
    """
    Classe (Letra D): Arquitetura da Rede Neural Convolucional (CNN).
    Criada com 4 camadas de Convolução (para extrair texturas e bordas do tumor)
    e 2 camadas Densas para a classificação binária final.
    Usa AdaptiveAvgPool2d para esmagar qualquer tamanho de imagem para 4x4,
    impedindo travamentos por excesso de memória nos laboratórios da faculdade.
    """

    def __init__(self, canais_entrada=1):
        super(CNNCancerMama, self).__init__()
        self.conv1 = nn.Conv2d(canais_entrada, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # Logit bruto (A matemática será feita na função de erro)


def treinar_modelo(modelo, dataloader_treino, num_epocas=1):
    """
    Função de Treinamento: Roda o Forward (previsão) e o Backward (correção de peso).
    Aplica pos_weight=2.39 (Letra B) na função BCEWithLogitsLoss para que a rede não
    ignore os casos minoritários de câncer com pCR Presente.
    """
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(disp)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.39]).to(disp))
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)

    print("\n" + "=" * 50)
    print(f"  🧠 RELATÓRIO DE TREINAMENTO DA REDE NEURAL")
    print("=" * 50)

    for epoca in range(num_epocas):
        modelo.train()
        for i, (imgs, rotulos) in enumerate(dataloader_treino):
            imgs, rotulos = imgs.to(disp), rotulos.to(disp)
            otimizador.zero_grad()
            erro = criterio(modelo(imgs), rotulos)
            erro.backward()
            otimizador.step()
            print(f"  Processando Lote {i + 1} - Erro Atual (Loss): {erro.item():.4f}")
            if i == 1: break  # Parada rápida para a simulação na apresentação
    return modelo


# =====================================================================
# BLOCO 4: GRAD-CAM (Letra E)
# =====================================================================

class GradCAM:
    """
    Classe (Letra E): Espião (Explicabilidade Visual).
    Prende "ganchos" virtuais na última camada convolucional da rede. Quando a rede
    chega em uma conclusão, esta classe rouba a matemática dos gradientes e entende
    exatamente em qual pedaço de pixel a Inteligência Artificial focou a sua atenção.
    """

    def __init__(self, modelo, camada_alvo):
        self.modelo = modelo
        self.camada_alvo = camada_alvo
        self.grad = None
        self.ativ = None
        self.camada_alvo.register_forward_hook(self.salvar_ativ)
        self.camada_alvo.register_full_backward_hook(self.salvar_grad)

    def salvar_ativ(self, m, inp, out): self.ativ = out.detach()

    def salvar_grad(self, m, grad_in, grad_out): self.grad = grad_out[0].detach()

    def gerar(self, tensor_img):
        self.modelo.eval()
        previsao = self.modelo(tensor_img)
        self.modelo.zero_grad()
        previsao.backward(retain_graph=True)
        pesos = torch.mean(self.grad, dim=[2, 3], keepdim=True)
        cam = torch.sum(pesos * self.ativ, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)

        probabilidade = torch.sigmoid(previsao).item() * 100
        return cam.squeeze().cpu().numpy(), probabilidade


def plotar_gradcam(img_2d, mapa_calor, probabilidade):
    """Gera o mapa de calor em janela pop-up sobrepondo a cor (vermelho/azul) à ressonância"""
    mapa_redim = cv2.resize(mapa_calor, (img_2d.shape[1], img_2d.shape[0]))
    mapa_color = cv2.applyColorMap(np.uint8(255 * mapa_redim), cv2.COLORMAP_JET)
    img_cinza = np.uint8(255 * (img_2d / np.max(img_2d)))
    img_cinza_rgb = cv2.cvtColor(img_cinza, cv2.COLOR_GRAY2RGB)

    img_final = cv2.addWeighted(img_cinza_rgb, 0.6, mapa_color, 0.4, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_cinza, cmap='gray')
    plt.title("Ressonância 2D Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_final)
    plt.title(f"Grad-CAM (Decisão: {probabilidade:.1f}% pCR)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =====================================================================
# BLOCO 5: INTERFACE GRÁFICA ATUALIZADA (Letra A)
# =====================================================================

class VisualizadorMRI:
    """
    Interface Gráfica: Orquestra tudo. Carrega o exame no eixo Z e
    possui um botão "Cérebro" que chama a rede treinada para classificar
    aquele exato paciente que está na tela, escrevendo o resultado a cores.
    """

    def __init__(self, root, rede_treinada):
        self.root = root
        self.rede = rede_treinada
        self.root.title("Classificação pCR (DCE-MRI) - Processamento")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")
        ttk.Style().theme_use('clam')

        self.vol_dce = None
        self.vol_mask = None

        tk.Label(root, text="Visualizador e Diagnóstico por IA", font=("Segoe UI", 16, "bold"), bg="#2c3e50",
                 fg="white").pack(pady=10)

        frm_btn = tk.Frame(root, bg="#2c3e50")
        frm_btn.pack(side=tk.TOP, pady=5)

        ttk.Button(frm_btn, text="1. Carregar DCE-MRI", command=self.load_dce).grid(row=0, column=0, padx=10)
        ttk.Button(frm_btn, text="2. Carregar Máscara", command=self.load_mask).grid(row=0, column=1, padx=10)

        self.btn_ia = tk.Button(frm_btn, text="🧠 3. Analisar Imagem com IA", font=("Segoe UI", 10, "bold"),
                                bg="#e74c3c", fg="white", command=self.analisar_ia)
        self.btn_ia.grid(row=0, column=2, padx=20)

        self.lbl_diagnostico = tk.Label(root, text="Diagnóstico: Aguardando carregamento...",
                                        font=("Segoe UI", 14, "bold"), bg="#2c3e50", fg="#f1c40f")
        self.lbl_diagnostico.pack(pady=5)

        self.fig, (self.ax_dce, self.ax_mask) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'wspace': 0.1})
        self.fig.patch.set_facecolor('#ecf0f1')
        self.ax_dce.axis('off');
        self.ax_mask.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)

        frm_slider = tk.Frame(root, bg="#2c3e50")
        frm_slider.pack(side=tk.BOTTOM, fill=tk.X, pady=15, padx=50)
        self.slider = ttk.Scale(frm_slider, orient=tk.HORIZONTAL, command=self.atualizar)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.lbl_z = tk.Label(frm_slider, text="Z: 0", bg="#2c3e50", fg="white", font=("Segoe UI", 10, "bold"))
        self.lbl_z.pack(side=tk.RIGHT)

    def load_dce(self):
        c = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if c:
            self.vol_dce = nib.load(c).get_fdata()
            self.lbl_diagnostico.config(text="Diagnóstico: Pronto para análise.", fg="#3498db")
            self.ajustar()

    def load_mask(self):
        c = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if c: self.vol_mask = nib.load(c).get_fdata(); self.ajustar()

    def analisar_ia(self):
        if self.vol_dce is None:
            messagebox.showwarning("Aviso", "Carregue a imagem DCE-MRI primeiro!")
            return

        img_2d = extrair_2d_estrategia(self.vol_dce, estrategia="central")
        tensor_img = torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(0).float()

        cam_explica = GradCAM(self.rede, self.rede.conv4)
        mapa, probabilidade = cam_explica.gerar(tensor_img)

        classe_texto = "pCR PRESENTE" if probabilidade >= 50.0 else "pCR AUSENTE"
        cor = "#2ecc71" if probabilidade >= 50.0 else "#e74c3c"

        self.lbl_diagnostico.config(
            text=f"Diagnóstico da IA: {probabilidade:.1f}% de chance | Rótulo: {classe_texto}",
            fg=cor
        )
        plotar_gradcam(img_2d, mapa, probabilidade)

    def ajustar(self):
        m = self.vol_dce.shape[2] - 1 if self.vol_dce is not None else (
            self.vol_mask.shape[2] - 1 if self.vol_mask is not None else 0)
        self.slider.config(to=m);
        self.slider.set(m // 2);
        self.atualizar(m // 2)

    def atualizar(self, val):
        z = int(float(val))
        self.lbl_z.config(text=f"Z: {z}")
        self.ax_dce.clear();
        self.ax_mask.clear()
        self.ax_dce.axis('off');
        self.ax_mask.axis('off')

        if self.vol_dce is not None and z < self.vol_dce.shape[2]:
            self.ax_dce.imshow(self.vol_dce[:, :, z], cmap='gray', aspect=2.0)
            self.ax_dce.set_title(f"DCE-MRI (Z: {z})")
        if self.vol_mask is not None and z < self.vol_mask.shape[2]:
            self.ax_mask.imshow(self.vol_mask[:, :, z], cmap='gray', aspect=2.0)
            self.ax_mask.set_title(f"Máscara (Z: {z})")
        self.canvas.draw()


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    caminho_csv = "dataset/BreastDCEDL_metadata_min_crop.csv"
    analisar_balanceamento(caminho_csv)

    df = pd.read_csv(caminho_csv)
    dataset_treino = DatasetCancerMama(df, diretorio_dce="dados", estrategia_2d="central")
    dataloader = DataLoader(dataset_treino, batch_size=2, shuffle=True)

    minha_rede = CNNCancerMama(canais_entrada=1)
    minha_rede = treinar_modelo(minha_rede, dataloader, num_epocas=1)

    print("\n✅ Treinamento rápido concluído! Abrindo a Interface Gráfica...\n")

    janela = tk.Tk()
    app = VisualizadorMRI(janela, rede_treinada=minha_rede)
    janela.mainloop()