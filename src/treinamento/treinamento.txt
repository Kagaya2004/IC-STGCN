import torch
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
    """ Calcula o Erro Absoluto Medio (MAE) ignorando os dados ausentes via mascara. """
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
    print(f"Iniciando treinamento. Dispositivo alocado: {device}")
    
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    model_dir = os.path.join(base_dir, "src", "modelo")
    log_dir = os.path.join(base_dir, "data", "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Carregamento de Dados e Grafo
    print("Carregando Matriz de Adjacencia e DataLoaders...")
    # Suposicao: O modelo STGCN exige a matriz de adjacencia na inicializacao
    adj_mx = np.load(os.path.join(data_dir, "adj_mx.npy"))
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)

    batch_size = 32
    train_loader, val_loader, test_loader = preparar_loaders_chicago(data_dir, batch_size=batch_size)

    # 3. Inicializacao do Modelo (Baseado na filtragem de 178 nos e 2 features)
    model = STGCN_TrafficModel(
        num_nodes=178, 
        num_features=2, 
        num_timesteps_input=12, 
        num_timesteps_output=12,
        adj_mx=adj_tensor # Passando a matriz estrutural calculada
    ).to(device)

    # 4. Hiperparametros Otimizados
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Scheduler: Corta o LR pela metade se nao houver melhoria na validacao por 4 epocas
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    # Parametros de Early Stopping
    paciencia_early_stop = 12
    epocas_sem_melhoria = 0
    best_val_loss = float('inf')
    
    history = []
    tempo_inicio_total = time.time()

    # 5. Loop de Treinamento
    for epoch in range(epochs):
        tempo_inicio_epoca = time.time()
        
        # --- FASE DE TREINO ---
        model.train()
        train_loss_total = 0.0
        
        for batch_idx, (x, y, mask) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x)
            loss = masked_mae_loss(y_pred, y, mask)
            
            # Backward pass
            loss.backward()
            
            # Contencao de Explosao de Gradiente (Gradient Clipping)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss_total += loss.item()
            
            # Liberacao explicita de memoria do grafo computacional
            del x, y, mask, y_pred, loss

        train_loss_media = train_loss_total / len(train_loader)

        # --- FASE DE VALIDACAO ---
        model.eval()
        val_loss_total = 0.0
        
        with torch.no_grad(): # Desliga o rastreamento de gradiente (Economia massiva de VRAM)
            for x, y, mask in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                
                y_pred = model(x)
                loss = masked_mae_loss(y_pred, y, mask)
                val_loss_total += loss.item()
                
                del x, y, mask, y_pred, loss

        val_loss_media = val_loss_total / len(val_loader)
        
        # Atualiza o Scheduler
        scheduler.step(val_loss_media)
        
        duracao_epoca = time.time() - tempo_inicio_epoca

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoca [{epoch+1}/{epochs}] | Treino MAE: {train_loss_media:.4f} | Val MAE: {val_loss_media:.4f} | LR: {current_lr:.6f} | Tempo: {duracao_epoca:.1f}s")

        # --- EARLY STOPPING & MODEL CHECKPOINT ---
        if val_loss_media < best_val_loss:
            best_val_loss = val_loss_media
            epocas_sem_melhoria = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_stgcn_chicago.pt"))
            print(">>> Novo melhor modelo salvo!")
        else:
            epocas_sem_melhoria += 1
            if epocas_sem_melhoria >= paciencia_early_stop:
                print(f"\n--- Early Stopping ativado na epoca {epoch+1}. Treinamento interrompido. ---")
                break

        history.append({
            'epoch': epoch+1, 
            'train_loss': train_loss_media, 
            'val_loss': val_loss_media,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Limpeza agressiva de cache entre epocas para GPU limitada
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. Finalizacao e Salvamento dos Logs
    tempo_fim_total = time.time()
    tempo_formatado = str(timedelta(seconds=round(tempo_fim_total - tempo_inicio_total)))

    print("\n" + "="*40)
    print(f"TREINAMENTO FINALIZADO")
    print(f"Tempo Total: {tempo_formatado}")
    print(f"Melhor Perda de Validacao: {best_val_loss:.4f}")
    print("="*40)

    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(log_dir, "history_stgcn_v2.csv"), index=False)

if __name__ == "__main__":
    treinar_modelo()