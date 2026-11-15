"""Interface gráfica de inspeção NIfTI para a Issue #1.

Compatível com os volumes I-SPY1 MinCrop [Z,H,W]. A GUI não executa
diagnóstico nem treinamento; é uma ferramenta de inspeção de imagem/máscara.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class InterfaceNIfTI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PAI — Visualizador NIfTI e máscara")
        self.root.geometry("1180x820")
        self.imagem = None
        self.mascara = None
        self.imagem_path = None
        self.mascara_path = None
        self.imagem_affine = None
        self.mascara_affine = None

        barra = ttk.Frame(root, padding=10)
        barra.pack(fill=tk.X)
        ttk.Button(barra, text="Abrir DCE (.nii/.nii.gz)", command=self.abrir_imagem).pack(side=tk.LEFT, padx=4)
        ttk.Button(barra, text="Abrir máscara", command=self.abrir_mascara).pack(side=tk.LEFT, padx=4)
        ttk.Button(barra, text="Abrir DCE + máscara do paciente", command=self.abrir_par).pack(side=tk.LEFT, padx=4)
        self.status = tk.StringVar(value="Selecione um volume NIfTI e, opcionalmente, sua máscara.")
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1100).pack(fill=tk.X)

        self.fig, (self.ax_img, self.ax_mask) = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        rodape = ttk.Frame(root, padding=10)
        rodape.pack(fill=tk.X)
        self.z_label = tk.StringVar(value="Corte Z: —")
        ttk.Label(rodape, textvariable=self.z_label).pack(side=tk.LEFT)
        self.slider = tk.Scale(rodape, from_=0, to=0, orient=tk.HORIZONTAL,
                               command=self.atualizar, state=tk.DISABLED, length=800)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
        self.renderizar()

    @staticmethod
    def _carregar_volume(caminho: str):
        img = nib.load(caminho)
        dados = np.asarray(img.dataobj, dtype=np.float32)
        if dados.ndim == 4 and dados.shape[-1] == 1:
            dados = dados[..., 0]
        if dados.ndim != 3:
            raise ValueError(f"Esperado volume 3D (ou 4D com um único canal); recebido {dados.shape}.")
        if not np.isfinite(dados).all():
            raise ValueError("O volume contém NaN ou infinito.")
        return dados, np.asarray(img.affine)

    def _escolher(self, titulo):
        return filedialog.askopenfilename(
            title=titulo,
            filetypes=[("Arquivos NIfTI", "*.nii *.nii.gz"), ("Todos os arquivos", "*.*")]
        )

    def abrir_imagem(self):
        caminho = self._escolher("Selecione o volume DCE-MRI")
        if not caminho:
            return
        try:
            dados, affine = self._carregar_volume(caminho)
            self.imagem, self.imagem_affine, self.imagem_path = dados, affine, caminho
            self._sincronizar_slider()
            self._validar_par()
            self.status.set(f"DCE: {Path(caminho).name} | dimensões {dados.shape} | orientação affine carregada.")
            self.renderizar()
        except Exception as exc:
            messagebox.showerror("Não foi possível abrir a DCE-MRI", str(exc))

    def abrir_mascara(self):
        caminho = self._escolher("Selecione a máscara do mesmo paciente")
        if not caminho:
            return
        try:
            dados, affine = self._carregar_volume(caminho)
            self.mascara, self.mascara_affine, self.mascara_path = dados, affine, caminho
            self._sincronizar_slider()
            self._validar_par()
            self.status.set(f"Máscara: {Path(caminho).name} | dimensões {dados.shape}.")
            self.renderizar()
        except Exception as exc:
            messagebox.showerror("Não foi possível abrir a máscara", str(exc))

    def abrir_par(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta do paciente (DCE e máscara)")
        if not pasta:
            return
        arquivos = list(Path(pasta).glob("*.nii")) + list(Path(pasta).glob("*.nii.gz"))
        dces = [p for p in arquivos if "_dce_" in p.name]
        masks = [p for p in arquivos if "_mask" in p.name]
        if len(dces) != 1 or len(masks) != 1:
            messagebox.showwarning("Seleção ambígua", "A pasta deve conter exatamente uma DCE e uma máscara do mesmo paciente. Selecione os arquivos individualmente.")
            return
        try:
            imagem, affine_i = self._carregar_volume(str(dces[0]))
            mascara, affine_m = self._carregar_volume(str(masks[0]))
            self.imagem, self.imagem_affine, self.imagem_path = imagem, affine_i, str(dces[0])
            self.mascara, self.mascara_affine, self.mascara_path = mascara, affine_m, str(masks[0])
            self._sincronizar_slider()
            self._validar_par()
            self.status.set(f"Paciente: {dces[0].name} | {mascara.shape} | arquivos carregados.")
            self.renderizar()
        except Exception as exc:
            messagebox.showerror("Erro ao carregar o par", str(exc))

    def _validar_par(self):
        if self.imagem is None or self.mascara is None:
            return
        if self.imagem.shape != self.mascara.shape:
            raise ValueError(f"Dimensões incompatíveis: DCE {self.imagem.shape}, máscara {self.mascara.shape}.")
        if not np.allclose(self.imagem_affine, self.mascara_affine, rtol=1e-4, atol=1e-3):
            raise ValueError("A orientação/posição espacial (affine) da DCE difere da máscara.")
        if self.imagem_path and self.mascara_path:
            # Heurística de segurança: nomes precisam compartilhar identificador.
            id_img = Path(self.imagem_path).name.split("_spy1_")[0]
            id_mask = Path(self.mascara_path).name.split("_spy1_")[0]
            if id_img != id_mask:
                raise ValueError(f"Arquivos parecem pertencer a pacientes diferentes: {id_img} e {id_mask}.")

    def _sincronizar_slider(self):
        volume = self.imagem if self.imagem is not None else self.mascara
        if volume is None:
            return
        self.slider.config(to=volume.shape[0] - 1, state=tk.NORMAL)
        self.slider.set(volume.shape[0] // 2)

    def atualizar(self, valor):
        self.renderizar(int(float(valor)))

    def renderizar(self, z=None):
        volume = self.imagem if self.imagem is not None else self.mascara
        if volume is None:
            self.ax_img.clear(); self.ax_mask.clear()
            self.ax_img.set_title("DCE-MRI — aguardando volume")
            self.ax_mask.set_title("Máscara — aguardando volume")
            self.canvas.draw_idle()
            return
        z = int(self.slider.get()) if z is None and self.slider.winfo_exists() else (z or 0)
        z = max(0, min(z, volume.shape[0] - 1))
        self.ax_img.clear(); self.ax_mask.clear()
        if self.imagem is not None:
            self.ax_img.imshow(self.imagem[z, :, :], cmap="gray", origin="lower")
            self.ax_img.set_title(f"DCE-MRI — plano XY | Z={z}")
        else:
            self.ax_img.set_title("DCE-MRI não carregada")
        if self.mascara is not None:
            self.ax_mask.imshow(self.mascara[z, :, :] > 0, cmap="gray", origin="lower", vmin=0, vmax=1)
            self.ax_mask.set_title(f"Máscara — plano XY | Z={z}")
        else:
            self.ax_mask.set_title("Máscara não carregada")
        self.ax_img.axis("off"); self.ax_mask.axis("off")
        self.z_label.set(f"Corte Z: {z} / {volume.shape[0]-1}")
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    InterfaceNIfTI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
