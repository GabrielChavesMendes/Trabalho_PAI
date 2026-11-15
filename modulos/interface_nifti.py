"""Interface de inspeção dos volumes NIfTI do BreastDCEDL I-SPY1."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modulos.modulo_dataset import (
    FASES,
    carregar_metadados,
    criar_manifesto,
    ler_volume,
    selecionar_corte,
)
from modulos.modulo_balanceamento import resumo_particoes


class InterfaceNIfTI:
    """Abre paciente pelo CSV e sincroniza fases DCE e máscara no mesmo Z."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PAI — Visualizador DCE-MRI e máscara")
        root.geometry("1280x900")

        self.manifesto = {}
        self.imagens = []
        self.mascara = None
        self.paciente_atual = None
        self.z_max = 0

        controles = ttk.Frame(root, padding=8)
        controles.pack(fill=tk.X)

        self.csv_path = tk.StringVar(value="dataset/BreastDCEDL_SPY1_min_crop_metadata.csv")
        self.dataset_path = tk.StringVar(value="dataset/BreastDCEDL_ISPY1_min_crop")
        self.fase = tk.StringVar(value="tres")
        self.estrategia = tk.StringVar(value="maior_tumor")
        self.paciente = tk.StringVar()
        self.status = tk.StringVar(value="Selecione o CSV e a pasta dos volumes para começar.")

        ttk.Label(controles, text="CSV:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controles, textvariable=self.csv_path, width=58).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(controles, text="Procurar", command=self.escolher_csv).grid(row=0, column=2, padx=3)
        ttk.Label(controles, text="Dataset:").grid(row=1, column=0, sticky="w")
        ttk.Entry(controles, textvariable=self.dataset_path, width=58).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(controles, text="Procurar", command=self.escolher_dataset).grid(row=1, column=2, padx=3)
        ttk.Button(controles, text="Carregar pacientes", command=self.carregar_lista).grid(row=0, column=3, rowspan=2, padx=8)

        ttk.Label(controles, text="Paciente:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.combo_paciente = ttk.Combobox(controles, textvariable=self.paciente, state="readonly", width=24)
        self.combo_paciente.grid(row=2, column=1, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(controles, text="Fase:").grid(row=2, column=1, sticky="w", padx=(210, 0), pady=(8, 0))
        ttk.Combobox(controles, textvariable=self.fase, values=(*FASES, "tres"), state="readonly", width=14).grid(row=2, column=1, sticky="w", padx=(250, 0), pady=(8, 0))
        ttk.Label(controles, text="Corte:").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(controles, textvariable=self.estrategia, values=("central", "maior_tumor"), state="readonly", width=16).grid(row=2, column=2, sticky="e", pady=(8, 0))
        ttk.Button(controles, text="Abrir paciente", command=self.abrir_paciente).grid(row=2, column=3, padx=8, pady=(8, 0))
        controles.columnconfigure(1, weight=1)

        ttk.Label(root, textvariable=self.status, wraplength=1200, padding=8).pack(fill=tk.X)
        self.fig, self.axes = plt.subplots(1, 4, figsize=(13, 6))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        rodape = ttk.Frame(root, padding=8)
        rodape.pack(fill=tk.X)
        self.z_label = tk.StringVar(value="Corte Z: —")
        ttk.Label(rodape, textvariable=self.z_label).pack(side=tk.LEFT)
        self.slider = tk.Scale(
            rodape, from_=0, to=0, orient=tk.HORIZONTAL, command=self.atualizar,
            state=tk.DISABLED, length=900,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
        self.renderizar()

    def escolher_csv(self):
        caminho = filedialog.askopenfilename(
            title="Selecione o CSV oficial",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if caminho:
            self.csv_path.set(caminho)

    def escolher_dataset(self):
        caminho = filedialog.askdirectory(title="Selecione a pasta BreastDCEDL_ISPY1_min_crop")
        if caminho:
            self.dataset_path.set(caminho)

    def carregar_lista(self):
        try:
            tabela = carregar_metadados(self.csv_path.get())
            pacientes = criar_manifesto(tabela, self.dataset_path.get())
            self.manifesto = {paciente.identificador: paciente for paciente in pacientes}
            ids = list(self.manifesto)
            self.combo_paciente["values"] = ids
            if ids:
                self.paciente.set(ids[0])
            distribuicao = "; ".join(
                f"{item['particao']}: {item['total']} (pCR 0={item['pCR_0']}, pCR 1={item['pCR_1']})"
                for item in resumo_particoes(tabela)
            )
            self.status.set(f"{len(ids)} pacientes carregados. {distribuicao}")
        except Exception as exc:
            messagebox.showerror("Erro ao carregar CSV", str(exc))

    def abrir_paciente(self):
        paciente = self.manifesto.get(self.paciente.get())
        if paciente is None:
            messagebox.showwarning("Paciente", "Carregue o CSV e selecione um paciente.")
            return
        try:
            mascara, affine_mascara = ler_volume(paciente.caminho_mascara)
            indices = range(3) if self.fase.get() == "tres" else (FASES.index(self.fase.get()),)
            imagens = []
            for indice in indices:
                imagem, affine = ler_volume(paciente.caminhos_dce[indice])
                if imagem.shape != mascara.shape:
                    raise ValueError("A DCE e a máscara possuem dimensões diferentes.")
                if not np.allclose(affine, affine_mascara, rtol=1e-4, atol=1e-3):
                    raise ValueError("A DCE e a máscara não possuem a mesma orientação/posição (affine).")
                imagens.append(imagem)

            z_inicial = selecionar_corte(mascara, self.estrategia.get())
            self.imagens = imagens
            self.mascara = mascara
            self.paciente_atual = paciente
            self.z_max = mascara.shape[0] - 1
            self.slider.configure(to=self.z_max, state=tk.NORMAL)
            self.slider.set(z_inicial)
            self.status.set(
                f"{paciente.identificador} | pCR={paciente.pcr} | fases={len(imagens)} | "
                f"volume [Z,H,W]={mascara.shape} | corte inicial={z_inicial}. "
                "Exibição na ordem de armazenamento do NIfTI; confirme orientação no MRIcro."
            )
            self.renderizar(z_inicial)
        except Exception as exc:
            messagebox.showerror("Erro ao abrir paciente", str(exc))

    def atualizar(self, valor):
        self.renderizar(int(float(valor)))

    def renderizar(self, z=None):
        for eixo in self.axes:
            eixo.clear()
            eixo.axis("off")

        if self.mascara is None:
            self.axes[0].set_title("DCE-MRI")
            self.axes[3].set_title("Máscara")
            self.canvas.draw_idle()
            return

        z = int(self.slider.get()) if z is None and self.slider.winfo_exists() else int(z or 0)
        z = max(0, min(z, self.z_max))
        nomes = FASES if len(self.imagens) == 3 else (self.fase.get(),)
        for indice, (imagem, nome) in enumerate(zip(self.imagens, nomes)):
            self.axes[indice].imshow(imagem[z, :, :], cmap="gray", origin="lower")
            self.axes[indice].set_title(f"{nome} — XY, Z={z}")
            self.axes[indice].axis("off")
            if np.any(self.mascara[z] > 0):
                self.axes[indice].contour(self.mascara[z] > 0, levels=[0.5], colors="red", linewidths=0.8)

        self.axes[3].imshow(self.mascara[z, :, :] > 0, cmap="gray", vmin=0, vmax=1, origin="lower")
        self.axes[3].set_title(f"Máscara — XY, Z={z}")
        self.axes[3].axis("off")
        self.z_label.set(f"Corte Z: {z} / {self.z_max}")
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    InterfaceNIfTI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
