import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        x_conv = self.conv(x)
        p, q = torch.split(x_conv, x_conv.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        self.theta = nn.Linear(in_channels, out_channels)
        self.adj = None

    def forward(self, x, edge_index, edge_weight):
        batch, c, t, n = x.shape
        if self.adj is None or self.adj.shape[0] != n:
            self.adj = torch.zeros((n, n), device=x.device)
            self.adj[edge_index[0], edge_index[1]] = edge_weight

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, n, c)
        out = torch.matmul(self.adj, x_flat)
        out = self.theta(out)
        return F.relu(out.reshape(batch, t, n, -1).permute(0, 3, 1, 2))

class STConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(in_channels, hidden_channels)
        self.spa_conv = SpatialConv(hidden_channels, hidden_channels)
        self.tmp_conv2 = TemporalConv(hidden_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, edge_weight):
        res = self.residual(x)
        x = self.tmp_conv1(x)
        x = self.spa_conv(x, edge_index, edge_weight)
        x = self.tmp_conv2(x)
        x = F.relu(x + res[:, :, :x.shape[2], :]) 
        batch, c, t, n = x.shape
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class STGCN_TrafficModel(nn.Module):
    def __init__(self, num_nodes=207, num_features=5, num_timesteps_input=12, num_timesteps_output=12):
        super(STGCN_TrafficModel, self).__init__()
        self.input_conv = nn.Conv2d(num_features, 64, kernel_size=(1, 1))
        self.block1 = STConvBlock(64, 32, 64)
        self.block2 = STConvBlock(64, 32, 64)
        self.fully_connected = nn.Sequential(
            nn.Linear(64 * num_timesteps_input, 128),
            nn.ReLU(),
            nn.Linear(128, num_timesteps_output)
        )

    def forward(self, x, edge_index, edge_weight):
        x = x.permute(0, 3, 1, 2) 
        x = self.input_conv(x)
        x = self.block1(x, edge_index, edge_weight)
        x = self.block2(x, edge_index, edge_weight)
        batch, c, t, n = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, n, -1)
        out = self.fully_connected(x) 
        return out.permute(0, 2, 1) # [Batch, 12, 207]