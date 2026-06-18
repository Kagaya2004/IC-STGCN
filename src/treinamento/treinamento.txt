import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pandas as pd
from datetime import timedelta

# Importacoes dos modulos internos
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader import preparar_loaders_chicago

def masked_mae_loss(y_pred, y_true, mask):
    """
    Calcula o Erro Absoluto Medio (MAE) ignorando os dados ausentes via mascara.
    """
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def treinar_modelo():
    # 1. Configuracoes de Ambiente e Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    model_dir = os.path.join(base_dir, "src", "modelo")
    log_dir = os.path.join(base_dir, "data", "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Carregamento da Topologia e Loaders
    adj_path = os.path.join(data_dir, "adj_mx_chicago.npy")
    adj_mx = torch.from_numpy(np.load(adj_path)).float().to(device)
    num_nodes = adj_mx.shape[0]

    train_loader, val_loader, _ = preparar_loaders_chicago(data_dir, batch_size=32)

    # 3. Instanciacao do Modelo
    model = STGCN_TrafficModel(
        num_nodes=num_nodes, 
        num_features=2, 
        num_timesteps_input=12, 
        num_timesteps_output=12
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 4. Inicios do Monitoramento de Tempo
    print(f"Iniciando treinamento: {num_nodes} nos | Dispositivo: {device}")
    tempo_inicio_total = time.time()
    
    epochs = 50
    best_val_loss = float('inf')
    history = []

    for epoch in range(epochs):
        tempo_inicio_epoca = time.time()
        model.train()
        train_loss = 0
        
        for x_batch, y_batch, m_batch in train_loader:
            x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch, adj_mx)
            loss = masked_mae_loss(output, y_batch, m_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validacao
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_v, y_v, m_v in val_loader:
                x_v, y_v, m_v = x_v.to(device), y_v.to(device), m_v.to(device)
                out_v = model(x_v, adj_mx)
                val_loss += masked_mae_loss(out_v, y_v, m_v).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        tempo_fim_epoca = time.time()
        duracao_epoca = tempo_fim_epoca - tempo_inicio_epoca
        
        print(f"Epoca {epoch+1:02d}/{epochs} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | Tempo: {duracao_epoca:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_stgcn_chicago.pt"))

        history.append({
            'epoch': epoch+1, 
            'train_loss': train_loss, 
            'val_loss': val_loss,
            'duration_sec': duracao_epoca
        })

    # 5. Finalizacao e Relatorio de Tempo Total
    tempo_fim_total = time.time()
    duracao_total_segundos = tempo_fim_total - tempo_inicio_total
    tempo_formatado = str(timedelta(seconds=round(duracao_total_segundos)))

    print("-" * 30)
    print(f"Treinamento Concluido")
    print(f"Tempo Total: {tempo_formatado}")
    print("-" * 30)

    # 6. Salvamento dos Logs de Performance
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(log_dir, "history_stgcn_v2.csv"), index=False)
    
    # Salva resumo tecnico em arquivo de texto
    with open(os.path.join(log_dir, "performance_summary.txt"), "w") as f:
        f.write(f"Resumo de Performance Computacional\n")
        f.write(f"Modelo: STGCN V2.0\n")
        f.write(f"Total de Nos: {num_nodes}\n")
        f.write(f"Epocas: {epochs}\n")
        f.write(f"Tempo Total de Execucao: {tempo_formatado}\n")
        f.write(f"Media por Epoca: {duracao_total_segundos/epochs:.2f} segundos\n")
        f.write(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

if __name__ == "__main__":
    treinar_modelo()