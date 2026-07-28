import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """ Captura a evolução do tráfego no tempo usando Gated Linear Units (GLU). """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        # Saída dobrada para permitir a operação de Gate (filtro + porta)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        # x shape: [Batch, Canais, Tempo, Nós]
        x = self.conv(x)
        # GLU: Divide os canais ao meio para aplicar a sigmoid como gate
        p, q = torch.split(x, x.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatialConv(nn.Module):
    """ 
    Implementação de Convolução em Grafo (GCN). 
    Equação fundamental adaptada: Z = A * X * Theta 
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x shape: [Batch, Canais, Tempo, Nós]
        # adj shape: [Nós, Nós]
        batch_size, channels, timesteps, num_nodes = x.shape
        
        # Permuta para permitir a multiplicação matricial espacial: [Batch, Tempo, Nós, Canais]
        x_perm = x.permute(0, 2, 3, 1)
        
        # Propagação de mensagens no grafo através da matriz de adjacência espacial
        # [Batch, Tempo, Nós, Nós] x [Batch, Tempo, Nós, Canais] -> [Batch, Tempo, Nós, Canais]
        out = torch.einsum('ij,btjk->btik', adj, x_perm)
        
        # Transformação linear dos atributos espaciais combinados
        out = self.theta(out)
        
        # Retorna ao padrão do PyTorch: [Batch, Canais, Tempo, Nós]
        return out.permute(0, 3, 1, 2)

class STConvBlock(nn.Module):
    """ Bloco Espaço-Temporal Sanduíche (Temporal -> Spatial -> Temporal). """
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, adj):
        x = self.tmp_conv1(x)
        x = self.spa_conv(x, adj)
        x = self.tmp_conv2(x)
        x = self.batch_norm(x)
        return F.relu(x)

class STGCN_TrafficModel(nn.Module):
    """
    Arquitetura STGCN V2.5 otimizada para a malha viária de Chicago (N=1047).
    Suporta passagem dinâmica da matriz de adjacência no forward pass.
    """
    def __init__(self, in_channels=1, hidden_channels=64, out_channels=6, num_nodes=1047):
        super(STGCN_TrafficModel, self).__init__()
        
        # Camada de projeção inicial para expansão de canais de feature (1 canal -> 64 canais)
        self.input_layer = nn.Conv2d(in_channels, 64, 1)
        
        # Dois blocos convolucionais espaço-temporais profundos intercalados
        self.block1 = STConvBlock(64, hidden_channels // 2, 64, num_nodes)
        self.block2 = STConvBlock(64, hidden_channels // 2, 64, num_nodes)
        
        # Camada de saída para colapsar a dimensão temporal (12 passos) e projetar o horizonte futuro
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 128, (12, 1)),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x, adj):
        # x vindo do DataLoader: [Batch, Tempo(12), Nós(1047), Canais(1)]
        # Permuta para o padrão de mapas de recursos do PyTorch: [Batch, Canais, Tempo, Nós]
        x = x.permute(0, 3, 1, 2)
        
        # Projeção inicial de atributos
        x = self.input_layer(x)
        
        # Processamento através dos blocos espaço-temporais
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        
        # Redução temporal e projeção final do horizonte de previsão
        x = self.output_layer(x) # Shape resultante: [Batch, Horizonte(12), 1, Nós]
        
        # Acomoda o shape removendo a dimensão unitária colapsada para casar com a Loss
        x = x.squeeze(2) # Shape final síncrono: [Batch, Horizonte(12), Nós]
        
        return x