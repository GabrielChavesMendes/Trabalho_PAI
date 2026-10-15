import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VisualizadorMRI:
    """
    Classe responsável por criar a Interface Gráfica do Usuário (GUI).
    Ela integra botões, a barra de rolagem (slider) do eixo Z e os painéis
    de plotagem (Matplotlib) para visualizar a ressonância e a segmentação lado a lado.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Trabalho PAI - Visualizador DCE-MRI")
        self.root.geometry("900x600")  # Define o tamanho inicial da janela

        # Variáveis para armazenar as matrizes tridimensionais em memória
        self.volume_dce = None
        self.volume_mask = None
        self.max_z = 0

        # 1. FRAME SUPERIOR (Área dos Botões)
        self.frame_botoes = tk.Frame(root)
        self.frame_botoes.pack(side=tk.TOP, pady=10)

        # Botão para carregar a imagem original (Ressonância)
        self.btn_dce = tk.Button(self.frame_botoes, text="1. Carregar Imagem (DCE-MRI)", command=self.carregar_dce,
                                 width=30)
        self.btn_dce.pack(side=tk.LEFT, padx=10)

        # Botão para carregar a máscara (Segmentação)
        self.btn_mask = tk.Button(self.frame_botoes, text="2. Carregar Máscara (Tumor)", command=self.carregar_mask,
                                  width=30)
        self.btn_mask.pack(side=tk.LEFT, padx=10)

        # 2. FRAME CENTRAL (Área das Imagens)
        # Criamos uma figura do Matplotlib com 1 linha e 2 colunas para colocar lado a lado
        self.fig, (self.ax_dce, self.ax_mask) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.patch.set_facecolor('#f0f0f0')  # Cor de fundo combinando com a janela

        # Configuração estética inicial dos eixos
        self.ax_dce.set_title("DCE-MRI (Aguardando imagem...)")
        self.ax_mask.set_title("Máscara de Segmentação (Aguardando...)")
        self.ax_dce.axis('off')  # Esconde réguas numéricas
        self.ax_mask.axis('off')

        # FigureCanvasTkAgg: A 'mágica' que joga o gráfico do Matplotlib dentro da janela Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 3. FRAME INFERIOR (Barra de Navegação Z)
        # Scale é o slider. O comando 'atualizar_telas' é chamado automaticamente toda vez que ele é movido.
        self.slider_z = tk.Scale(root, from_=0, to=0, orient=tk.HORIZONTAL,
                                 label="Navegar no Eixo Z (Profundidade)",
                                 command=self.atualizar_telas, length=500)
        self.slider_z.pack(side=tk.BOTTOM, pady=20)

    def carregar_dce(self):
        """Abre a janela do Windows/Linux para o usuário selecionar o arquivo DCE-MRI"""
        caminho = filedialog.askopenfilename(title="Selecione a DCE-MRI", filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if caminho:
            # Reutilizamos a lógica que já aprendemos: ler com nibabel e pegar matriz float
            self.volume_dce = nib.load(caminho).get_fdata()
            self.ajustar_limites_slider()
            self.atualizar_telas(self.slider_z.get())

    def carregar_mask(self):
        """Abre a janela do Windows/Linux para o usuário selecionar o arquivo da Máscara"""
        caminho = filedialog.askopenfilename(title="Selecione a Máscara", filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if caminho:
            self.volume_mask = nib.load(caminho).get_fdata()
            self.ajustar_limites_slider()
            self.atualizar_telas(self.slider_z.get())

    def ajustar_limites_slider(self):
        """Descobre quantas fatias (Z) a imagem tem e ajusta o tamanho máximo da barra de rolagem"""
        if self.volume_dce is not None:
            self.max_z = self.volume_dce.shape[2] - 1
        elif self.volume_mask is not None:
            self.max_z = self.volume_mask.shape[2] - 1

        self.slider_z.config(to=self.max_z)
        self.slider_z.set(self.max_z // 2)  # Pula automaticamente para o corte central

    def atualizar_telas(self, valor_slider):
        """
        Função principal de renderização.
        Recebe o valor atual da barra de rolagem e desenha as respectivas fatias.
        """
        z_atual = int(valor_slider)

        # Limpa o desenho anterior para evitar sobreposição de memória
        self.ax_dce.clear()
        self.ax_mask.clear()

        # Desenha a fatia da DCE-MRI (se já tiver sido carregada)
        if self.volume_dce is not None and z_atual < self.volume_dce.shape[2]:
            self.ax_dce.imshow(self.volume_dce[:, :, z_atual], cmap='gray')
            self.ax_dce.set_title(f"DCE-MRI - Plano XY (Z: {z_atual})")
        else:
            self.ax_dce.set_title("DCE-MRI não carregada")

        # Desenha a fatia da Máscara (se já tiver sido carregada)
        if self.volume_mask is not None and z_atual < self.volume_mask.shape[2]:
            self.ax_mask.imshow(self.volume_mask[:, :, z_atual], cmap='gray')  # Máscara também em cinza
            self.ax_mask.set_title(f"Máscara - Plano XY (Z: {z_atual})")
        else:
            self.ax_mask.set_title("Máscara não carregada")

        self.ax_dce.axis('off')
        self.ax_mask.axis('off')

        # Força o Matplotlib a atualizar o gráfico na interface
        self.canvas.draw()


# =====================================================================
# INICIALIZAÇÃO DO PROGRAMA
# =====================================================================
if __name__ == "__main__":
    janela_principal = tk.Tk()
    app = VisualizadorMRI(janela_principal)
    janela_principal.mainloop()  # Mantém a janela aberta rodando em loop