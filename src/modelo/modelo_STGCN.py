import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """
    Captura a evolucao do trafego no tempo usando Gated Linear Units (GLU).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        # Saida dobrada para permitir a operacao de Gate (filtro + porta)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        # x shape: [Batch, Canais, Tempo, Nos]
        x = self.conv(x)
        # GLU: Divide os canais ao meio para aplicar a sigmoid como gate
        p, q = torch.split(x, x.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatialConv(nn.Module):
    """
    Implementacao de Convolucao em Grafo (GCN).
    Equacao: Z = A * X * Theta
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x shape: [Batch, Canais, Tempo, Nos]
        batch_size, c_in, t_steps, num_nodes = x.shape
        
        # Reorganiza para multiplicacao de matrizes: [Batch * Tempo, Nos, Canais]
        x = x.permute(0, 2, 3, 1).reshape(-1, num_nodes, c_in)
        
        # Agregacao espacial baseada na Matriz de Adjacencia Ponderada
        x = torch.matmul(adj, x)
        
        # Transformacao linear (Theta)
        x = self.theta(x)
        
        # Retorna para o formato: [Batch, Canais, Tempo, Nos]
        x = x.reshape(batch_size, t_steps, num_nodes, -1)
        return x.permute(0, 3, 1, 2)

class STConvBlock(nn.Module):
    """
    Bloco Espaco-Temporal (Temporal -> Spatial -> Temporal).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        
        # LayerNorm e Dropout para estabilizacao do treino
        self.ln = nn.LayerNorm([out_channels, 12, num_nodes])
        self.dropout = nn.Dropout(0.2)
        
        # Conexao Residual
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj):
        res = self.residual(x)
        x = self.tmp_conv1(x)
        x = self.spa_conv(x, adj)
        x = self.tmp_conv2(x)
        x = self.dropout(x)
        return self.ln(x + res)

class STGCN_TrafficModel(nn.Module):
    """
    Arquitetura STGCN V2.0 otimizada para a malha central de Chicago.
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(STGCN_TrafficModel, self).__init__()
        
        # Expansao de features (Velocidade + Seno da Hora) para canais internos
        self.input_layer = nn.Conv2d(num_features, 64, 1)
        
        self.block1 = STConvBlock(64, 32, 64, num_nodes)
        self.block2 = STConvBlock(64, 32, 64, num_nodes)
        
        # AJUSTE: O kernel (num_timesteps_input, 1) colapsa a dimensao temporal de entrada
        # Isso garante que a predicao nao arraste dimensoes extras indesejadas
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 128, (num_timesteps_input, 1)),
            nn.ReLU(),
            nn.Conv2d(128, num_timesteps_output, 1)
        )

    def forward(self, x, adj):
        # x vindo do DataLoader: [Batch, Tempo(12), Nos(179), Canais(2)]
        # Permuta para o padrao da convolucao: [Batch, Canais, Tempo, Nos]
        x = x.permute(0, 3, 1, 2)
        
        x = self.input_layer(x)
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        
        # Aplicacao da camada de predicao
        x = self.output_layer(x)
        
        # Saida atual: [Batch, Horizon(12), 1, Nos(179)]
        # Removemos a dimensao 2 (tempo residual) para bater com o Ground Truth [Batch, Horizon, Nos]
        return x.squeeze(2)