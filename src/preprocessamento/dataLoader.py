import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Subset

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
        """Identifica pontos de inicio de janela que nao contenham dados nulos massivos."""
        valid_indices = []
        total_steps = self.data.shape[0]
        
        # Varredura para garantir que a janela completa (input + target) seja valida
        # O limite e (Total - entrada - saida)
        for i in range(total_steps - self.window_size - self.horizon):
            # Simplificacao logica: se a mascara indicar que mais de 50% dos sensores
            # cairam ao mesmo tempo na janela inteira, ignoramos esse recorte.
            # Isso acelera a criacao do dataset e protege a GCN de matrizes ruidosas.
            window_mask = self.mask[i : i + self.window_size + self.horizon]
            
            # Se a taxa de dados validos for maior que 80%, aceita a janela
            valid_ratio = window_mask.mean()
            if valid_ratio > 0.8:
                valid_indices.append(i)
                
        print(f"Dataset Temporal: {len(valid_indices)} recortes validos gerados a partir de {total_steps} passos.")
        return valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # X: (Janela de Entrada, Nos, Features)
        x = self.data[start_idx : start_idx + self.window_size]
        
        # Y: (Horizonte Futuro, Nos) -> Prever apenas a velocidade (Canal 0)
        y = self.data[start_idx + self.window_size : start_idx + self.window_size + self.horizon, :, 0]
        
        # M: Mascara correspondente ao Y (usada para ignorar perdas em dados imputados artificialmente)
        m = self.mask[start_idx + self.window_size : start_idx + self.window_size + self.horizon]
        
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

def preparar_loaders_chicago(data_dir, batch_size=32, train_split=0.7, val_split=0.15):
    """
    Cria os objetos DataLoader para treino, validacao e teste com DIVISAO CRONOLOGICA.
    """
    caminho_dados = os.path.join(data_dir, "data_chicago.npy")
    caminho_mask = os.path.join(data_dir, "mask_chicago.npy")
    
    dataset_completo = ChicagoTrafficDataset(caminho_dados, caminho_mask)
    
    # 1. Calculo rigoroso dos limites cronologicos
    total = len(dataset_completo)
    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val
    
    # 2. Divisao Sequencial (Evitando Vazamento do Futuro para o Passado)
    indices_totais = list(range(total))
    
    idx_train = indices_totais[:n_train]
    idx_val = indices_totais[n_train : n_train + n_val]
    idx_test = indices_totais[n_train + n_val :]
    
    train_set = Subset(dataset_completo, idx_train)
    val_set = Subset(dataset_completo, idx_val)
    test_set = Subset(dataset_completo, idx_test)
    
    print(f"Splits Cronológicos Criados: Treino ({len(train_set)}), Val ({len(val_set)}), Teste ({len(test_set)})")

    # Configuracao de performance para RTX 3050 (num_workers=2-4 dependendo do CPU, pin_memory acelera transf. para GPU)
    num_workers = 2 
    
    # shuffle=True no Treino é permitido pois os conjuntos temporais já estão isolados
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader