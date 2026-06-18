import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class ChicagoTrafficDataset(Dataset):
    def __init__(self, data_path, mask_path, window_size=12, horizon=12):
        """
        Dataset especializado para a malha viaria de Chicago.
        Args:
            data_path: Caminho para 'data_chicago.npy' (T, N, C)
            mask_path: Caminho para 'mask_chicago.npy' (T, N)
            window_size: Tamanho da janela de entrada (ex: 12 passos = 2 horas)
            horizon: Horizonte de previsao (ex: 12 passos = 2 horas)
        """
        # Carregamento dos dados em float32 para otimizacao de VRAM
        self.data = np.load(data_path).astype(np.float32)
        self.mask = np.load(mask_path).astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon
        
        # Filtragem de janelas validas para evitar cruzamento de gaps temporais extensos
        self.indices = self._gerar_indices_validos()

    def _gerar_indices_validos(self):
        """Identifica pontos de inicio de janela que nao contenham dados nulos."""
        valid_indices = []
        total_steps = self.data.shape[0]
        
        # Varredura para garantir que a janela completa (input + target) seja valida
        # O limite e (Total - entrada - saida)
        for i in range(total_steps - self.window_size - self.horizon):
            # Verifica se existe algum valor NaN no canal de velocidade (canal 0)
            if not np.isnan(self.data[i : i + self.window_size + self.horizon, :, 0]).any():
                valid_indices.append(i)
        
        print(f"Indices temporais processados. Janelas validas encontradas: {len(valid_indices)}")
        return valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # X: Janela de entrada (Passos, Nos, Canais)
        x = self.data[start_idx : start_idx + self.window_size]
        
        # Y: Alvo da previsao (apenas canal 0: velocidade)
        y = self.data[start_idx + self.window_size : start_idx + self.window_size + self.horizon, :, 0]
        
        # M: Mascara correspondente ao Y (usada para ignorar perdas em dados ausentes)
        m = self.mask[start_idx + self.window_size : start_idx + self.window_size + self.horizon]
        
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

def preparar_loaders_chicago(data_dir, batch_size=32, train_split=0.7, val_split=0.15):
    """
    Cria os objetos DataLoader para treino, validacao e teste.
    """
    caminho_dados = os.path.join(data_dir, "data_chicago.npy")
    caminho_mask = os.path.join(data_dir, "mask_chicago.npy")
    
    dataset_completo = ChicagoTrafficDataset(caminho_dados, caminho_mask)
    
    # Calculo de tamanhos para divisao do dataset
    total = len(dataset_completo)
    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset_completo, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Configuracao de performance para RTX 3050 (num_workers e pin_memory)
    params = {
        'batch_size': batch_size,
        'num_workers': 2,
        'pin_memory': True
    }
    
    train_loader = DataLoader(train_set, shuffle=True, **params)
    val_loader = DataLoader(val_set, shuffle=False, **params)
    test_loader = DataLoader(test_set, shuffle=False, **params)
    
    return train_loader, val_loader, test_loader