import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np


# =====================================================================
# 1. CLASSE DO DATASET (Carrega e converte 3D para 2D dinamicamente)
# =====================================================================
class DatasetCancerMama(Dataset):
    def __init__(self, df, diretorio_dce, transform=None, estrategia_2d="central", canais=1):
        """
        df: DataFrame contendo as anotações do CSV
        diretorio_dce: Pasta onde estão os arquivos .nii.gz
        estrategia_2d: "central" ou "mip" (Letra C do trabalho)
        canais: 1 (usando apenas pré-contraste) ou 3 (combinando fases)
        """
        self.df = df
        self.diretorio = diretorio_dce
        self.transform = transform
        self.estrategia_2d = estrategia_2d
        self.canais = canais

    def __len__(self):
        return len(self.df)

    def extrair_2d(self, volume):
        if self.estrategia_2d == "central":
            z_mid = volume.shape[2] // 2
            return volume[:, :, z_mid]
        else:  # MIP (Maximum Intensity Projection)
            return np.max(volume, axis=2)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        paciente_id = linha['pid']  # Ex: 'ISPY1_1001'
        classe_pcr = np.float32(linha['pCR'])

        # Padrão de nomenclatura baseado na imagem fornecida:
        nome_base = f"{paciente_id}_spy1_vis1_dce_aqc_"

        try:
            if self.canais == 1:
                # Letra D (Estratégia 1): Treinar só com a fase de pós-contraste precoce (aqc_1)
                caminho = os.path.join(self.diretorio, f"{nome_base}1.nii.gz")
                volume_3d = nib.load(caminho).get_fdata()
                imagem_2d = self.extrair_2d(volume_3d)
                tensor_imagem = torch.from_numpy(imagem_2d).unsqueeze(0).float()

            else:
                # Letra D (Estratégia 2): As 3 fases juntas (0, 1 e 2)
                fases_2d = []
                for i in range(3):
                    caminho = os.path.join(self.diretorio, f"{nome_base}{i}.nii.gz")
                    vol = nib.load(caminho).get_fdata()
                    fases_2d.append(self.extrair_2d(vol))

                # Empilha as 3 matrizes criando uma imagem de 3 canais (RGB artificial)
                tensor_imagem = torch.from_numpy(np.stack(fases_2d, axis=0)).float()

            return tensor_imagem, torch.tensor([classe_pcr])

        except FileNotFoundError:
            # Proteção caso algum paciente falte na pasta
            return torch.zeros((self.canais, 256, 256)), torch.tensor([classe_pcr])


# =====================================================================
# 2. O LOOP DE TREINAMENTO (Lidando com Desbalanceamento)
# =====================================================================
def treinar_modelo(modelo, dataloader_treino, num_epocas=10):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)

    # ESTRATÉGIA DE DESBALANCEAMENTO (Letra B)
    # Total de pacientes 0: 1024 | Total de pacientes 1: 428
    # O pos_weight é calculado como: Classe Majoritária / Classe Minoritária
    peso_positivo = torch.tensor([1024 / 428]).to(dispositivo)

    # BCEWithLogitsLoss é a função ideal para classificação binária.
    # Ela já embute a Sigmoide, o que impede problemas matemáticos de instabilidade no Pytorch.
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivo)
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)

    print(f"Iniciando treinamento no dispositivo: {dispositivo}")

    for epoca in range(num_epocas):
        modelo.train()
        perda_acumulada = 0.0

        for imagens, rotulos in dataloader_treino:
            imagens = imagens.to(dispositivo)
            rotulos = rotulos.to(dispositivo)

            otimizador.zero_grad()  # Zera o gradiente da rodada anterior

            previsoes = modelo(imagens)  # Forward
            erro = criterio(previsoes, rotulos)  # Calcula o erro com a penalidade da classe minoritária

            erro.backward()  # Backward (Regra da Cadeia)
            otimizador.step()  # Atualiza os pesos

            perda_acumulada += erro.item()

        print(f"Época [{epoca + 1}/{num_epocas}] - Loss: {perda_acumulada / len(dataloader_treino):.4f}")

    return modelo