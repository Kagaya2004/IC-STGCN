import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

# Importações do projeto
from src.modelo.modelo_STGCN import STGCN_TrafficModel

def masked_mae_loss(y_pred, y_true, mask):
    """
    Calcula o MAE ignorando falhas de sensores.
    Garante que as dimensões batam antes da operação.
    """
    # FIX: Força o alinhamento de dimensões para evitar RuntimeError
    y_pred = y_pred.reshape(y_true.shape)
    mask = mask.reshape(y_true.shape)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def create_windows(data, mask, window=12, horizon=12):
    """Gera janelas de tempo respeitando a integridade dos dados."""
    x, y, m = [], [], []
    for i in range(len(data) - window - horizon):
        slice_data = data[i : i+window+horizon]
        # Só adiciona se não houver NaNs (evita o gap de 9 meses)
        if not np.isnan(slice_data).any():
            x.append(data[i : i+window])
            y.append(data[i+window : i+window+horizon, :, 0]) # Alvo: Velocidade
            m.append(mask[i+window : i+window+horizon])
    return np.array(x), np.array(y), np.array(m)

def treinar_modelo():
    # 1. Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    model_path = os.path.join(base_dir, "src", "modelo", "best_stgcn_chicago.pt")
    
    # 2. Carregar Dados
    print("📂 Carregando tensores e matriz de adjacência...")
    adj_mx = np.load(os.path.join(data_dir, "adj_mx_chicago.npy"))
    adj_mx = torch.from_numpy(adj_mx).float().to(device)
    
    data = np.load(os.path.join(data_dir, "data_chicago.npy"))
    mask = np.load(os.path.join(data_dir, "mask_chicago.npy"))
    
    # 3. Preparar Janelas
    x_raw, y_raw, m_raw = create_windows(data, mask)
    
    if len(x_raw) == 0:
        print("❌ Nenhuma janela válida encontrada. Verifique o dataCleaning.")
        return

    # Split 80/20
    split = int(0.8 * len(x_raw))
    # Input X no formato (Batch, Canais, Tempo, Nós) -> (B, 2, 12, 60)
    train_x = torch.tensor(x_raw[:split]).permute(0, 3, 1, 2)
    val_x = torch.tensor(x_raw[split:]).permute(0, 3, 1, 2)
    
    train_loader = DataLoader(TensorDataset(train_x, torch.tensor(y_raw[:split]), torch.tensor(m_raw[:split])), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, torch.tensor(y_raw[split:]), torch.tensor(m_raw[split:])), batch_size=32)

    # 4. Iniciar Modelo
    model = STGCN_TrafficModel(num_nodes=60, num_features=2, num_timesteps_input=12, num_timesteps_output=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Loop de Treino
    print(f"🚀 Iniciando treinamento em {device}...")
    best_loss = float('inf')

    for epoch in range(50):
        model.train()
        train_loss = 0
        for x_batch, y_batch, m_batch in train_loader:
            x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
            
            optimizer.zero_grad()
            
            # Execução do modelo
            output = model(x_batch, adj_mx)
            
            # FIX: Redimensiona a saída para (Batch, Tempo, Nós) para bater com o y_batch
            y_pred = output.reshape(y_batch.shape)
            
            loss = masked_mae_loss(y_pred, y_batch, m_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_v, y_v, m_v in val_loader:
                x_v, y_v, m_v = x_v.to(device), y_v.to(device), m_v.to(device)
                out_v = model(x_v, adj_mx)
                y_pred_v = out_v.reshape(y_v.shape)
                val_loss += masked_mae_loss(y_pred_v, y_v, m_v).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), model_path)
            print("⭐ Modelo salvo!")

if __name__ == "__main__":
    treinar_modelo()