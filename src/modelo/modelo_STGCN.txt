import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """ Captura a evolucao do trafego no tempo usando Gated Linear Units (GLU). """
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
    Equacao fundamental: Z = A * X * Theta 
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x shape: [Batch, Canais, Tempo, Nos]
        # adj shape: [Nos, Nos]
        batch_size, c_in, t_steps, num_nodes = x.shape
        
        # Reorganiza para multiplicacao matricial: [Batch * Tempo, Nos, Canais]
        x_perm = x.permute(0, 2, 3, 1).reshape(-1, num_nodes, c_in)
        
        # 1. Agregacao Espacial: Multiplicacao pela Matriz de Adjacencia Hibrida (A * X)
        # adj[None, :, :] força o broadcasting para o tamanho do batch temporal
        x_space = torch.bmm(adj.unsqueeze(0).expand(x_perm.shape[0], -1, -1), x_perm)
        
        # 2. Transformacao Linear: Aplicacao do filtro linear (Theta)
        z = self.theta(x_space)
        
        # Restaura o formato original: [Batch, Canais_Out, Tempo, Nos]
        z = z.reshape(batch_size, t_steps, num_nodes, -1).permute(0, 3, 1, 2)
        return F.relu(z)

class STConvBlock(nn.Module):
    """ Bloco Espaco-Temporal Sanduiche (Temporal -> Spatial -> Temporal). """
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, adj):
        # Preserva identidade para conexao residual
        residual = x 
        
        # Camada Temporal 1
        x = self.tmp_conv1(x)
        # Camada Espacial (GCN) injetando a estrutura do grafo
        x = self.spa_conv(x, adj)
        # Camada Temporal 2
        x = self.tmp_conv2(x)
        
        # Normalizacao e conexao residual para estabilizar o gradiente
        x = self.batch_norm(x)
        return x + residual[:, :, :x.shape[2], :]

class STGCN_TrafficModel(nn.Module):
    """
    Arquitetura STGCN V2.0 otimizada para a malha central de Chicago.
    Instancia a topologia do grafo de forma estatica via PyTorch Buffer.
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, adj_mx):
        super(STGCN_TrafficModel, self).__init__()
        
        # Registra a matriz de adjacencia como buffer permanente do modelo
        self.register_buffer('adj_mx', adj_mx)
        
        # Camada de projecao inicial para expansao de canais de feature
        self.input_layer = nn.Conv2d(num_features, 64, 1)
        
        # Dois blocos convolucionais espaco-temporais profundos
        self.block1 = STConvBlock(64, 32, 64, num_nodes)
        self.block2 = STConvBlock(64, 32, 64, num_nodes)
        
        # Camada de saída para colapsar a dimensao temporal e prever o horizonte futuro
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 128, (num_timesteps_input, 1)),
            nn.ReLU(),
            nn.Conv2d(128, num_timesteps_output, 1)
        )

    def forward(self, x):
        # x vindo do DataLoader: [Batch, Tempo(12), Nos(178), Canais(2)]
        # Permuta para o padrao da convolucao do PyTorch: [Batch, Canais, Tempo, Nos]
        x = x.permute(0, 3, 1, 2)
        
        # Projecao de atributos
        x = self.input_layer(x)
        
        # Propagacao espaco-temporal utilizando o buffer interno estatico
        x = self.block1(x, self.adj_mx)
        x = self.block2(x, self.adj_mx)
        
        # Geracao das predicoes lineares de velocidade
        out = self.output_layer(x)
        
        # Remove a dimensao temporal residual para retornar [Batch, Horizonte(12), Nos(178)]
        return out.squeeze(2)