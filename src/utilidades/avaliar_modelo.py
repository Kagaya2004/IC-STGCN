import torch
import numpy as np
import pandas as pd
import os
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader import preparar_loaders

def calcular_metricas(y_true, y_pred):
    """Calcula métricas padrão para artigos de predição de tráfego."""
    # MAE: Erro Absoluto Médio
    mae = np.mean(np.abs(y_true - y_pred))
    # RMSE: Raiz do Erro Quadrático Médio
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # MAPE: Erro Percentual Médio Absoluto (evitando divisão por zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return mae, rmse, mape

def avaliar_stgcn_final():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Avaliando no dispositivo: {device}")

    # 1. Configuração de Caminhos
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    model_path = os.path.join(base_dir, "src", "modelos", "pesos", "best_stgcn_chicago.pt")
    
    # 2. Carregar Estatísticas de Normalização (Crucial para o Artigo)
    stats = pd.read_csv(os.path.join(data_dir, "scaler_stats.csv"))
    mean_speed = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_speed = stats.loc[stats['feature'] == 'speed', 'std'].values[0]

    # 3. Preparar Grafo e Modelo
    adj_path = os.path.join(data_dir, "adj_mat_chicago.npy")
    adj_mat = torch.from_numpy(np.load(adj_path)).to(device)
    
    model = STGCN_TrafficModel(num_nodes=60, num_features=2).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Reinjetar a matriz de adjacência nas camadas espaciais
    for block in [model.block1, model.block2]:
        block.spa_conv.adj = adj_mat
    
    model.eval()

    # 4. Carregar Dados de Teste (Loader)
    # O test_loader contém os 10% de dados que não foram usados no treino nem na val
    _, _, test_loader = preparar_loaders(data_dir, batch_size=32)

    all_preds = []
    all_targets = []

    print(f"Realizando predições em {len(test_loader)} batches de teste...")
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            
            # Inferência
            output = model(x_batch) # Shape: (B, 12, 60)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())

    # Concatenar todos os resultados
    y_pred_norm = np.concatenate(all_preds, axis=0)
    y_true_norm = np.concatenate(all_targets, axis=0)

    # 5. DESNORMALIZAÇÃO: Converter de Z-Score para MPH
    # Fórmula: Real = (Z * Std) + Média
    y_pred_real = (y_pred_norm * std_speed) + mean_speed
    y_true_real = (y_true_norm * std_speed) + mean_speed

    # 6. Cálculo das Métricas de Performance
    mae, rmse, mape = calcular_metricas(y_true_real, y_pred_real)

    print("\n" + "="*40)
    print("      RESULTADOS FINAIS (UNIDADES REAIS)")
    print("="*40)
    print(f"MAE (Erro Médio):   {mae:.4f} MPH")
    print(f"RMSE (Desvio):      {rmse:.4f} MPH")
    print(f"MAPE (Erro %):      {mape:.2f}%")
    print("="*40)

    # 7. Guardar resultados para a tabela do Artigo
    metrics_df = pd.DataFrame({
        'Modelo': ['STGCN_Chicago'],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'Hardware': ['RTX 3050']
    })
    save_path = os.path.join(base_dir, "data", "logs", "metrics_final_real.csv")
    metrics_df.to_csv(save_path, index=False)
    print(f"Métricas reais guardadas em: logs/metrics_final_real.csv")

if __name__ == "__main__":
    avaliar_stgcn_final()