import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

# Importações do seu projeto
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader import preparar_loaders

def treinar_stgcn():
    # 1. Configurações de Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Treinando em: {device}")

    # 2. Definição de Caminhos
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    logs_dir = os.path.join(base_dir, "data", "logs")
    model_save_dir = os.path.join(base_dir, "src", "modelos", "pesos")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # 3. Carregar Estrutura do Grafo
    adj_path = os.path.join(data_dir, "adj_mat_chicago.npy")
    adj_mat = torch.from_numpy(np.load(adj_path)).to(device)
    
    # 4. Inicializar Modelos e Hiperparâmetros
    # IMPORTANTE: num_features=2 (Velocidade + Dia da Semana)
    model = STGCN_TrafficModel(
        num_nodes=60, 
        num_features=2, 
        num_timesteps_input=12, 
        num_timesteps_output=12
    ).to(device)
    
    # Passa a matriz de adjacência para as camadas espaciais do modelo
    for block in [model.block1, model.block2]:
        block.spa_conv.adj = adj_mat

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Preparar Dados (Batch 32 para RTX 3050)
    train_loader, val_loader, _ = preparar_loaders(data_dir, batch_size=32)

    # 6. Loop de Treinamento
    epochs = 50
    history = []
    best_val_loss = float('inf')
    
    print("\nIniciando Treinamento (50 Épocas)...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validação
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_out = model(x_val)
                val_loss = criterion(val_out, y_val)
                val_losses.append(val_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Salvar o melhor modelo (Best Weights)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_stgcn_chicago.pt"))

        # Log de progresso
        print(f"Época [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        history.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

    total_time = (time.time() - start_time) / 60
    print(f"\nTreinamento Finalizado em {total_time:.2f} minutos.")

    # 7. Salvar Histórico para o Artigo
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(logs_dir, "history_stgcn.csv"), index=False)
    print(f"Histórico salvo em: logs/history_stgcn.csv")

if __name__ == "__main__":
    treinar_stgcn()