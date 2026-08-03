import torch
import torch.optim as optim
import numpy as np
import time
import os
from pathlib import Path
from datetime import timedelta

# Importações internas
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader import preparar_loaders_chicago

def masked_mae_loss(y_pred, y_true, mask):
    """ Calcula o Erro Absoluto Médio (MAE) ignorando os dados ausentes via máscara. """
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def treinar_modelo():
    # 1. Configurações Dinâmicas e Hardware
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    model_dir = base_dir / "src" / "modelo"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando ambiente PyTorch. Dispositivo alocado: {device.type.upper()}")
    
    # 2. Carrega os DataLoaders (Horizonte 6 passos / 60 min)
    HORIZON = 6
    WINDOW_SIZE = 12
    batch_size = 32
    
    loaders = preparar_loaders_chicago(batch_size=batch_size, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_loader = loaders['treinamento']
    val_loader = loaders['validacao']
    
    caminho_meta = data_dir / "node_metadata.csv"
    caminho_adj = data_dir / "adj_mx.npy"
    
    import pandas as pd
    meta_df = pd.read_csv(caminho_meta) if caminho_meta.exists() else None
    num_nodes = len(meta_df) if meta_df is not None else 60
    
    adj_mx = np.load(caminho_adj)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)
    
    # 3. Instancia o Modelo, Otimizador e Scheduler
    model = STGCN_TrafficModel(in_channels=1, hidden_channels=64, out_channels=HORIZON, num_nodes=num_nodes).to(device)
    epochs = 150
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Caminhos dos arquivos de salvamento
    caminho_pesos_salvar = model_dir / "stgcn_pesos_chicago.pth"
    caminho_checkpoint = model_dir / "stgcn_checkpoint.pth"
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 4. LÓGICA DE RETOMADA DE TREINAMENTO (FULL CHECKPOINTING)
    if caminho_checkpoint.exists():
        print(f"\n🔄 ARQUIVO DE CHECKPOINT DETECTADO!")
        print("Restaurando a 'memória' da Inteligência Artificial...")
        checkpoint = torch.load(caminho_checkpoint, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"-> Treinamento retomado a partir da Época {start_epoch+1}.")
        print(f"-> Melhor Loss de Validação anterior: {best_val_loss:.4f}\n")
    else:
        print(f"Instanciando STGCN do zero (Nós: {num_nodes} | Horizonte: {HORIZON*10} min)...")
        print("\n================== TREINAMENTO INICIADO ==================")
    
    # 5. O Loop de Treinamento
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss_total = 0.0
            start_epoch_time = time.time()
            
            for i, (x_batch, y_batch, mask_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).squeeze(-1)
                mask_batch = mask_batch.to(device).squeeze(-1)
                
                optimizer.zero_grad()
                y_pred = model(x_batch, adj_tensor)
                
                loss = masked_mae_loss(y_pred, y_batch, mask_batch)
                loss.backward()
                optimizer.step()
                
                train_loss_total += loss.item()
                
                if i % 1000 == 0:
                    print(f" Época {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss Atual: {loss.item():.4f}", end='\r')

            train_loss_avg = train_loss_total / len(train_loader)

            # Validação
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for x_val, y_val, mask_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device).squeeze(-1)
                    mask_val = mask_val.to(device).squeeze(-1)

                    y_pred_val = model(x_val, adj_tensor)
                    val_loss = masked_mae_loss(y_pred_val, y_val, mask_val)
                    val_loss_total += val_loss.item()

            val_loss_avg = val_loss_total / len(val_loader)
            scheduler.step()
            
            tempo_epoch = time.time() - start_epoch_time
            print(f"\n[Época {epoch+1}/{epochs}] Concluída em {tempo_epoch:.1f}s | Train MAE: {train_loss_avg:.4f} | Val MAE: {val_loss_avg:.4f}")

            # Salva o PESO FINAL (Apenas Inferência) se for um recorde
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(model.state_dict(), caminho_pesos_salvar)
                print(f" -> Novo recorde! Pesos limpos salvos em: {caminho_pesos_salvar.name}")
                
            # Salva o CHECKPOINT DE ESTADO COMPLETO (Sempre, ao fim de toda época)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, caminho_checkpoint)

    except KeyboardInterrupt:
        print("\n\n⚠️ Treinamento INTERROMPIDO manualmente pelo usuário (Ctrl+C).")
        print("Fique tranquilo! O estado da última época concluída está a salvo no arquivo de checkpoint.")
        print("Basta executar o script novamente para continuar de onde parou.\n")

if __name__ == "__main__":
    treinar_modelo()