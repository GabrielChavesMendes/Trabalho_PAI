import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCancerMama(nn.Module):
    def __init__(self, canais_entrada=1):
        """
        canais_entrada=1: Treinamento usando apenas a fase pré-contraste, pós-precoce ou pós-tardio.
        canais_entrada=3: Treinamento usando as 3 fases sobrepostas (como RGB).
        """
        super(CNNCancerMama, self).__init__()

        # O trabalho exige no mínimo 5 camadas com parâmetros.
        # Usaremos 4 camadas Convolucionais e 2 camadas Densas (Lineares), totalizando 6 camadas.

        # Camada 1: Extração de bordas e texturas locais
        self.conv1 = nn.Conv2d(in_channels=canais_entrada, out_channels=32, kernel_size=3, padding=1)
        # Camada 2: Extração de padrões geométricos
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Camada 3: Padrões complexos do tumor
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Camada 4: Abstração de alto nível
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # AdaptiveAvgPool achata as saídas espaciais da conv4 (seja lá qual for o tamanho do crop)
        # para uma grade fixa de 4x4. Isso previne o estouro de memória (Out Of Memory).
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Camada 5: Totalmente Conectada (O classificador)
        # 256 canais da conv4 * 4 * 4 (tamanho do adaptive pool) = 4096 entradas
        self.fc1 = nn.Linear(4096, 512)

        self.dropout = nn.Dropout(0.5)  # Prevenção de Overfitting

        # Camada 6: Totalmente Conectada (Saída)
        # 1 neurônio de saída pois é classificação binária (pCR 0 ou 1)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Propagação pelas convoluções com função de ativação ReLU e Max Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)  # Achata a matriz 2D para um vetor 1D

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Retornamos os "logits" brutos.
        # A função Sigmoid será aplicada pela função de perda BCEWithLogitsLoss no loop de treino,
        # garantindo maior estabilidade numérica matemática do que aplicar nn.Sigmoid() aqui.
        x = self.fc2(x)
        return x


# =====================================================================
# BLOCO DE TESTE
# =====================================================================
if __name__ == "__main__":
    # Teste 1: Simulando um volume fatiado de apenas 1 fase (1 Canal de Entrada)
    modelo_1_fase = CNNCancerMama(canais_entrada=1)
    tensor_falso_1fase = torch.randn(1, 1, 256, 256)  # (Lote=1, Canais=1, Altura=256, Largura=256)
    saida_1fase = modelo_1_fase(tensor_falso_1fase)
    print(f"Formato da saída (1 Fase): {saida_1fase.shape}")

    # Teste 2: Simulando as 3 fases sobrepostas (3 Canais de Entrada)
    modelo_3_fases = CNNCancerMama(canais_entrada=3)
    tensor_falso_3fases = torch.randn(1, 3, 256, 256)  # (Lote=1, Canais=3, Altura=256, Largura=256)
    saida_3fases = modelo_3_fases(tensor_falso_3fases)
    print(f"Formato da saída (3 Fases): {saida_3fases.shape}")