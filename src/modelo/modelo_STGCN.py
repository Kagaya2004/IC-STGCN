import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """
    Camada de Convolução Temporal: Captura dependências ao longo do tempo.
    Utiliza uma estrutura de Gated Linear Unit (GLU) para filtrar informações.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        # A saída é dobrada (out_channels * 2) para a operação de gate (GLU)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        # x shape: (Batch, Channels, Timesteps, Nodes)
        x = self.conv(x)
        # Divide o tensor ao meio na dimensão dos canais para aplicar o gate
        p, q = torch.split(x, x.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatialConv(nn.Module):
    """
    Camada de Convolução Espacial: Captura a topologia da malha viária.
    Utiliza a Matriz de Adjacência (W) injetada durante o treino.
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)
        self.adj = None  # Será injetada pelo script de treinamento

    def forward(self, x):
        # x shape: (Batch, Channels, Timesteps, Nodes)
        # Permuta para (Batch, Timesteps, Nodes, Channels) para facilitar a multiplicação
        x = x.permute(0, 2, 3, 1)
        
        # Agregação baseada no Grafo: X_vizinho = W * X
        if self.adj is not None:
            # Multiplicação matricial da vizinhança
            x = torch.matmul(self.adj, x)
        
        # Transformação Linear (Pesos treináveis da GCN)
        x = self.theta(x)
        x = F.relu(x)
        
        # Retorna para o formato original (Batch, Channels, Timesteps, Nodes)
        return x.permute(0, 3, 1, 2)

class STConvBlock(nn.Module):
    """
    Bloco Espaço-Temporal "Sanduíche": Temporal -> Espacial -> Temporal.
    Inclui conexões residuais e LayerNorm para estabilidade na RTX 3050.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)
        
        # Conexão residual para evitar desvanecimento de gradiente
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.tmp_conv1(x)
        x = self.spa_conv(x)
        x = self.tmp_conv2(x)
        
        # Normalização e Resíduo
        # x shape: (B, C, T, N) -> LayerNorm espera canais no final
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        
        return x + res

class STGCN_TrafficModel(nn.Module):
    """
    Modelo STGCN Principal.
    Configurado para: 60 nós, 2 features de entrada e predição de 3 horas.
    """
    def __init__(self, num_nodes=60, num_features=2, num_timesteps_input=12, num_timesteps_output=12):
        super(STGCN_TrafficModel, self).__init__()
        
        # Camada de entrada para projetar as 2 features (Velocidade + Dia) para 64 canais
        self.input_conv = nn.Conv2d(num_features, 64, 1)
        
        # Blocos Espaço-Temporais
        self.block1 = STConvBlock(64, 32, 64)
        self.block2 = STConvBlock(64, 32, 64)
        
        # Camada de Saída (Predição Multi-Step)
        # Reduz os Timesteps de entrada para a dimensão de saída desejada
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 128, (num_timesteps_input, 1)),
            nn.ReLU(),
            nn.Conv2d(128, num_timesteps_output, 1)
        )

    def forward(self, x):
        # Entrada x: (Batch, Features=2, Timesteps=12, Nodes=60)
        x = self.input_conv(x)
        
        x = self.block1(x)
        x = self.block2(x)
        
        # Saída: (Batch, Timesteps_Output=12, 1, Nodes=60)
        x = self.output_layer(x)
        
        # Ajusta shape para (Batch, Timesteps, Nodes) para o cálculo da Loss (MSE)
        return x.squeeze(2)