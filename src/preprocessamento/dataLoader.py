import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class ChicagoTrafficDataset(Dataset):
    def __init__(self, data_path, mask_path, window_size=12, horizon=12):
        """
        Dataset especializado para Chicago V2.0.
        Args:
            data_path: Caminho para 'data_chicago.npy' (T, N, C)
            mask_path: Caminho para 'mask_chicago.npy' (T, N)
        """
        self.data = np.load(data_path)
        self.mask = np.load(mask_path)
        self.window_size = window_size
        self.horizon = horizon
        
        # Identificar apenas os índices que possuem dados contínuos
        self.valid_indices = self._check_continuity()

    def _check_continuity(self):
        """Descarta janelas que cruzam o gap de 9 meses ou períodos sem dados."""
        valid = []
        T = self.data.shape[0]
        
        print(f"🔍 Validando integridade de {T} blocos temporais...")
        for i in range(T - self.window_size - self.horizon):
            # Se houver NaN em qualquer parte da janela (Input + Output), a sequência é inválida
            window_slice = self.data[i : i + self.window_size + self.horizon]
            if not np.isnan(window_slice).any():
                valid.append(i)
        
        print(f"✅ {len(valid)} sequências íntegras identificadas para treino.")
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # X: (Canais, Tempo, Nós)
        x = self.data[start_idx : start_idx + self.window_size]
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        
        # Y: (Tempo, Nós) - Alvo é a Velocidade (Canal 0)
        y = self.data[start_idx + self.window_size : start_idx + self.window_size + self.horizon, :, 0]
        y = torch.from_numpy(y).float()
        
        # M: Máscara correspondente ao Y para cálculo da Loss
        m = self.mask[start_idx + self.window_size : start_idx + self.window_size + self.horizon]
        m = torch.from_numpy(m).float()
        
        return x, y, m

def preparar_loaders_v2(base_path, batch_size=32):
    """Configura os DataLoaders para a Iniciação Científica."""
    d_path = os.path.join(base_path, "data_chicago.npy")
    m_path = os.path.join(base_path, "mask_chicago.npy")
    
    full_dataset = ChicagoTrafficDataset(d_path, m_path)
    
    # Divisão 80/10/10 para rigor acadêmico
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # pin_memory=True é essencial para performance em GPUs como a RTX 3050
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, pin_memory=True),
        DataLoader(test_set, batch_size=batch_size, pin_memory=True)
    )

if __name__ == "__main__":
    # Teste rápido
    path = r"C:\IC\Aplication\data\tratados"
    tr, vl, ts = preparar_loaders_v2(path)
    print("🚀 DataLoaders V2.0 prontos para o treinamento.")