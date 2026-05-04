import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, data_path, window_size=12, horizon=12):
        """
        Args:
            data_path: Caminho para o 'data_chicago_final.npy' (Shape: T, 2, 60)
            window_size: Histórico de entrada (12 passos = 3 horas)
            horizon: Horizonte de predição (12 passos = 3 horas)
        """
        # Carrega o tensor binário (Timesteps, Features, Nodes)
        self.data = np.load(data_path)
        self.window_size = window_size
        self.horizon = horizon
        
        # Gerar apenas índices que não cruzam a lacuna de dados (NaNs)
        self.indices = self._filtrar_janelas_validas()

    def _filtrar_janelas_validas(self):
        """
        Varre os dados e remove índices onde a janela de entrada ou saída 
        conteria buracos de dados (provenientes da lacuna de 6 meses).
        """
        valid_indices = []
        T = self.data.shape[0]
        
        print(f"Analisando {T} timesteps para identificar janelas válidas...")
        
        # Percorre os dados verificando se há NaNs no bloco (janela + horizonte)
        for i in range(T - self.window_size - self.horizon):
            bloco = self.data[i : i + self.window_size + self.horizon, 0, :]
            if not np.isnan(bloco).any():
                valid_indices.append(i)
        
        print(f"Filtro concluído: {len(valid_indices)} janelas prontas para treino.")
        return valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # X: (Canais, Tempo, Nós) -> Entrada: Velocidade + Dia da Semana
        # Pegamos o bloco (window, 2, 60) e transpomos para (2, window, 60)
        x = self.data[start_idx : start_idx + self.window_size, :, :]
        x = np.transpose(x, (1, 0, 2)) 
        
        # Y: (Tempo, Nós) -> Alvo: Apenas Velocidade (Canal 0)
        y = self.data[start_idx + self.window_size : start_idx + self.window_size + self.horizon, 0, :]
        
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def preparar_loaders(base_path, batch_size=32):
    """
    Organiza a divisão dos dados para a Iniciação Científica (80/10/10)
    """
    data_path = os.path.join(base_path, "data_chicago_final.npy")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}. Execute o dataCleaning.py primeiro.")

    full_dataset = TrafficDataset(data_path)
    
    # Divisão estatística para o artigo
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Reprodutibilidade científica
    )
    
    # Configuração para RTX 3050: pin_memory=True acelera transferência CPU -> GPU
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"Loaders criados: Treino={len(train_loader)} | Val={len(val_loader)} | Teste={len(test_loader)} (batches)")
    return train_loader, val_loader, test_loader