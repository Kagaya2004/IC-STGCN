import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class ChicagoTrafficDataset(Dataset):
    def __init__(self, data_path, mask_path, window_size=12, horizon=12):
        """ 
        Dataset especializado para a malha viária de Chicago. 
        Realiza a imputação matemática segura para evitar NaN Poisoning.
        """
        raw_data = np.load(data_path).astype(np.float32)
        
        # CORREÇÃO CRÍTICA (Zero-Mean Imputation):
        # Como usamos StandardScaler, a média estatística dos dados é 0.0.
        # Substituímos os sensores offline (NaN) pela média (0.0) para não envenenar 
        # a matemática convolucional da STGCN.
        self.data = np.nan_to_num(raw_data, nan=0.0)
        
        self.mask = np.load(mask_path).astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size : idx + self.window_size + self.horizon]
        mask_y = self.mask[idx + self.window_size : idx + self.window_size + self.horizon]
        
        return torch.tensor(x), torch.tensor(y), torch.tensor(mask_y)

def preparar_loaders_chicago(batch_size=32, window_size=12, horizon=12):
    """ Cria os DataLoaders lendo diretamente os splits físicos (Abordagem B). """
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    loaders = {}
    splits = ["treinamento", "validacao", "teste"]
    
    print("\nInicializando DataLoaders (PyTorch)...")
    
    for split in splits:
        caminho_dados = data_dir / f"data_{split}.npy"
        caminho_mask = data_dir / f"mask_{split}.npy"
        
        if not caminho_dados.exists() or not caminho_mask.exists():
            raise FileNotFoundError(f"Erro Crítico: Tensores para '{split}' não encontrados em {data_dir}.")
            
        dataset = ChicagoTrafficDataset(caminho_dados, caminho_mask, window_size, horizon)
        
        shuffle = True if split == "treinamento" else False
        
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        print(f" -> DataLoader [{split.upper()}] pronto. Batches: {len(loaders[split])}")

    return loaders["treinamento"], loaders["validacao"], loaders["teste"]