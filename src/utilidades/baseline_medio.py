import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from src.modelo.modelo_STGCN import STGCN_TrafficModel

def extrair_comparativo_cientifico():
    print("Iniciando Extração de Métricas Científicas (Auditoria Multicritério)...")
    
    data_dir = base_dir / "data" / "tratados"
    modelo_dir = base_dir / "src" / "modelo"
    
    caminho_dados_teste = data_dir / "data_teste.npy"
    caminho_mask_teste = data_dir / "mask_teste.npy"
    caminho_pesos = modelo_dir / "stgcn_pesos_chicago.pth"
    caminho_meta = data_dir / "node_metadata.csv"
    caminho_adj = data_dir / "adj_mx.npy"
    caminho_stats = data_dir / "scaler_stats.csv"
    
    # 1. Carregamento das Estatísticas para Desnormalização (MPH)
    stats_df = pd.read_csv(caminho_stats)
    std_speed = float(stats_df.loc[stats_df['feature'] == 'speed', 'std'].values[0])
    mean_speed = float(stats_df.loc[stats_df['feature'] == 'speed', 'mean'].values[0])
    print(f"-> Estatísticas carregadas: Média = {mean_speed:.2f} MPH | Desvio = {std_speed:.2f} MPH")

    HORIZON = 6
    WINDOW_SIZE = 12
    
    # Carregamento em memória dos tensores brutos de teste
    data_teste = np.load(caminho_dados_teste).astype(np.float32)
    mask_teste = np.load(caminho_mask_teste).astype(np.float32)
    adj_mx = np.load(caminho_adj).astype(np.float32)
    meta_df = pd.read_csv(caminho_meta)
    num_nodes = len(meta_df)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Dispositivo de cálculo: {device.type.upper()}")
    
    model = STGCN_TrafficModel(in_channels=1, hidden_channels=64, out_channels=HORIZON, num_nodes=num_nodes).to(device)
    model.load_state_dict(torch.load(caminho_pesos, map_location=device, weights_only=True))
    model.eval()
    
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)
    
    total_janelas = len(data_teste) - WINDOW_SIZE - HORIZON + 1
    print(f"-> Avaliando {total_janelas} janelas temporais...")
    
    registros = []
    
    # Processamento em lote para acelerar a GPU
    BATCH_SIZE = 64
    
    with torch.no_grad():
        for b_start in range(0, total_janelas, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, total_janelas)
            b_size = b_end - b_start
            
            x_batch_list, y_batch_list, mask_y_list, mask_x_list = [], [], [], []
            
            for idx in range(b_start, b_end):
                x = data_teste[idx : idx + WINDOW_SIZE]
                y = data_teste[idx + WINDOW_SIZE : idx + WINDOW_SIZE + HORIZON]
                m_x = mask_teste[idx : idx + WINDOW_SIZE]
                m_y = mask_teste[idx + WINDOW_SIZE : idx + WINDOW_SIZE + HORIZON]
                
                x_batch_list.append(x)
                y_batch_list.append(y)
                mask_x_list.append(m_x)
                mask_y_list.append(m_y)
                
            x_b = torch.tensor(np.array(x_batch_list), dtype=torch.float32).to(device)
            y_b = np.array(y_batch_list).squeeze(-1)       # (B, H, N)
            m_y_b = np.array(mask_y_list).squeeze(-1)      # (B, H, N)
            x_b_np = np.array(x_batch_list).squeeze(-1)    # (B, W, N)
            m_x_b = np.array(mask_x_list).squeeze(-1)      # (B, W, N)
            
            # --- MODELO 1: STGCN ---
            y_pred_stgcn = model(x_b, adj_tensor).cpu().numpy() # (B, H, N)
            
            # --- MODELO 2: Média Móvel (Usando máscara real m_x_b) ---
            soma_x = np.sum(x_b_np * m_x_b, axis=1) # (B, N)
            qtd_x = np.sum(m_x_b, axis=1)           # (B, N)
            media_via = np.divide(soma_x, qtd_x, out=np.zeros_like(soma_x), where=qtd_x != 0)
            y_pred_ma = np.repeat(media_via[:, np.newaxis, :], HORIZON, axis=1) # (B, H, N)
            
            # --- MODELO 3: Média Histórica Global (HA) ---
            y_pred_ha = np.zeros_like(y_pred_ma)
            
            # Cálculo de perdas mascaradas por janela
            for i in range(b_size):
                m_i = m_y_b[i]
                soma_mask = np.sum(m_i)
                if soma_mask == 0:
                    continue
                
                loss_stgcn = np.sum(np.abs(y_pred_stgcn[i] - y_b[i]) * m_i) / soma_mask
                loss_ma = np.sum(np.abs(y_pred_ma[i] - y_b[i]) * m_i) / soma_mask
                loss_ha = np.sum(np.abs(y_pred_ha[i] - y_b[i]) * m_i) / soma_mask
                
                # Erro no horizonte de 60 min (passo 5 = índice final)
                m_i_60 = m_i[-1]
                soma_mask_60 = np.sum(m_i_60)
                loss_stgcn_60 = np.sum(np.abs(y_pred_stgcn[i, -1] - y_b[i, -1]) * m_i_60) / soma_mask_60 if soma_mask_60 > 0 else loss_stgcn
                loss_ma_60 = np.sum(np.abs(y_pred_ma[i, -1] - y_b[i, -1]) * m_i_60) / soma_mask_60 if soma_mask_60 > 0 else loss_ma
                
                # Velocidade média real observada na janela para classificar tráfego
                leituras_validas = x_b_np[i][m_x_b[i] == 1]
                vel_media_janela = np.mean(leituras_validas) if len(leituras_validas) > 0 else 0.0
                
                registros.append({
                    "mae_stgcn": loss_stgcn,
                    "mae_ma": loss_ma,
                    "mae_ha": loss_ha,
                    "mae_stgcn_60min": loss_stgcn_60,
                    "mae_ma_60min": loss_ma_60,
                    "vel_media": vel_media_janela
                })
                
    df = pd.DataFrame(registros)
    limiar_pico = np.percentile(df["vel_media"], 20)
    
    df_pico = df[df["vel_media"] <= limiar_pico]
    df_livre = df[df["vel_media"] > limiar_pico]
    
    def formatar_linha(nome, subset):
        ha_z = subset["mae_ha"].mean()
        ma_z = subset["mae_ma"].mean()
        st_z = subset["mae_stgcn"].mean()
        
        # Conversão direta para Milhas por Hora (MPH)
        ha_mph = ha_z * std_speed
        ma_mph = ma_z * std_speed
        st_mph = st_z * std_speed
        return f"{nome:<20} | {ha_mph:<16.2f} MPH | {ma_mph:<16.2f} MPH | {st_mph:<16.2f} MPH"

    print("\n" + "="*90)
    print("🔬 RELATÓRIO CIENTÍFICO DE DESEMPENHO (ERRO ABSOLUTO MÉDIO EM MPH)")
    print("="*90)
    print(f"{'CENÁRIO':<20} | {'MÉDIA GLOBAL (HA)':<20} | {'MÉDIA MÓVEL (MA)':<20} | {'STGCN (IA)'}")
    print("-" * 90)
    print(formatar_linha("Trânsito Livre", df_livre))
    print(formatar_linha("Pico/Congestionado", df_pico))
    print("-" * 90)
    print(formatar_linha("MÉDIA GERAL", df))
    print("="*90)
    
    # Auditoria de Longo Prazo (Horizonte de 60 min)
    ma_60_mph = df["mae_ma_60min"].mean() * std_speed
    st_60_mph = df["mae_stgcn_60min"].mean() * std_speed
    print(f"\n📌 ANÁLISE DO HORIZONTE DE 60 MINUTOS:")
    print(f" -> MAE Média Móvel (60 min): {ma_60_mph:.2f} MPH")
    print(f" -> MAE STGCN (60 min):       {st_60_mph:.2f} MPH")
    print(f" -> Vantagem do STGCN:        {ma_60_mph - st_60_mph:+.2f} MPH")

if __name__ == "__main__":
    extrair_comparativo_cientifico()