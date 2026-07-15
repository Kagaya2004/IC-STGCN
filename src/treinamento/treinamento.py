import torch
import torch.optim as optim
import numpy as np
import time
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
    
    # Otimização para GPUs NVIDIA (Acelera as convoluções)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 2. Carregamento da Matriz Topológica
    caminho_adj = data_dir / "adj_mx.npy"
    if not caminho_adj.exists():
        raise FileNotFoundError("Matriz de adjacência não encontrada. Rode matriz_adjacencia.py primeiro.")
    
    adj_mx = np.load(caminho_adj)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)
    num_nodes = adj_mx.shape[0]

    # 3. Inicialização dos DataLoaders (Proteção de VRAM: Batch baixo)
    # Lote reduzido para 16 visando proteger a limitação de 6GB de VRAM
    batch_size = 16 
    train_loader, val_loader, test_loader = preparar_loaders_chicago(
        batch_size=batch_size, window_size=12, horizon=12
    )

    # 4. Inicialização do Modelo
    # Parâmetros padrão ajustados para a STGCN
    print(f"Instanciando STGCN (Nós: {num_nodes})...")
    model = STGCN_TrafficModel(
        in_channels=1, 
        hidden_channels=64, # Reduza para 32 se houver estouro de VRAM
        out_channels=12,    # Previsão dos próximos 12 passos (2 horas)
        num_nodes=num_nodes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 5. Loop de Treinamento
    epochs = 50
    best_val_loss = float('inf')
    caminho_pesos = model_dir / "stgcn_pesos_chicago.pth"

    print("\n================== TREINAMENTO INICIADO ==================")
    start_train_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        start_epoch_time = time.time()

        for batch_idx, (x, y, mask) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            mask = mask.to(device).squeeze(-1)
            
            optimizer.zero_grad()
            
            # Forward Pass: A STGCN exige os dados e a matriz de adjacência
            y_pred = model(x, adj_tensor)
            
            # Assegura que o output tenha a mesma dimensão para a loss
            loss = masked_mae_loss(y_pred, y, mask)
            loss.backward()
            
            # Prevenção contra explosão de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss_total += loss.item()
            
            if batch_idx % 50 == 0:
                print(f" Época {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss Atual: {loss.item():.4f}", end='\r')

        train_loss_avg = train_loss_total / len(train_loader)

        # 6. Validação
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

        # Checkpoint: Salva apenas se o modelo melhorou (Early Stopping base)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), caminho_pesos)
            print(f" -> Novo recorde! Pesos salvos em: {caminho_pesos.name}")

    total_time = time.time() - start_train_time
    print(f"\n================ TREINAMENTO FINALIZADO ================")
    print(f"Tempo total: {timedelta(seconds=int(total_time))}")
    print(f"Melhor MAE de Validação: {best_val_loss:.4f}")

if __name__ == "__main__":
    treinar_modelo()