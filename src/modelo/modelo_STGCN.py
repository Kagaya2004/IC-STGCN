import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """
    Captura a evolução do tráfego no tempo usando Gated Linear Units (GLU).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        # Saída dobrada para permitir a operação de Gate (GLU)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        # x shape: (Batch, Canais, Tempo, Nós)
        x = self.conv(x)
        # GLU: Divide os canais ao meio. Metade vira filtro, metade vira porta (sigmoid)
        p, q = torch.split(x, x.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatialConv(nn.Module):
    """
    Captura a influência das ruas vizinhas usando Convolução em Grafo (GCN).
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x shape: (Batch, Canais, Tempo, Nós)
        # adj: Matriz de Adjacência Ponderada (N, N)
        
        # 1. Agregação de Vizinhança via Grafo
        # Permuta para (Batch, Tempo, Canais, Nós) para multiplicar pelo grafo
        x_perm = x.permute(0, 2, 1, 3)
        # Multiplicação matricial: Cada nó recebe a média ponderada dos seus vizinhos
        out = torch.matmul(x_perm, adj)
        # 2. Transformação Linear (Aprendizado de Pesos)
        out = self.theta(out.permute(0, 1, 3, 2))
        return out.permute(0, 3, 1, 2)

class STConvBlock(nn.Module):
    """
    Bloco Espaço-Temporal: Temporal -> Espacial -> Temporal.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Conexão Residual para evitar desvanecimento do gradiente
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj):
        res = self.residual(x)
        x = self.tmp_conv1(x)
        x = self.spa_conv(x, adj)
        x = self.tmp_conv2(x)
        x = self.dropout(x)
        
        # Layer Normalization
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        
        return x + res

class STGCN_TrafficModel(nn.Module):
    """
    Arquitetura STGCN V2.0 para Chicago.
    Entrada: (Batch, 2 features, 12 timesteps, 60 nós)
    Saída: (Batch, 1 feature, 12 timesteps, 60 nós)
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(STGCN_TrafficModel, self).__init__()
        
        # 1. Projeção de Entrada (Expande as 2 features para 64 canais internos)
        self.input_layer = nn.Conv2d(num_features, 64, 1)
        
        # 2. Empilhamento de Blocos Espaço-Temporais
        self.block1 = STConvBlock(64, 32, 64)
        self.block2 = STConvBlock(64, 32, 64)
        
        # 3. Camada de Saída (Regressão para o Horizonte de Predição)
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 128, (num_timesteps_input, 1)),
            nn.ReLU(),
            nn.Conv2d(128, num_timesteps_output, 1)
        )

    def forward(self, x, adj):
        # Injeção de dados
        x = self.input_layer(x)
        # Processamento Híbrido
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        # Predição final
        x = self.output_layer(x)
        # Reshape para (Batch, 1, Output_Steps, Nodes)
        return x.unsqueeze(1)