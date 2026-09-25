import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.modelo.modelo_STGCN import STGCN_TrafficModel

def avaliar_modelos_unificado():
    print("Iniciando Avaliação Científica Unificada (STGCN vs Média Móvel vs Média Global)...")
    
    # 1. Configuração de Caminhos via Pathlib
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    modelo_dir = base_dir / "src" / "modelo"
    
    caminho_dados_teste = data_dir / "data_teste.npy"
    caminho_mask_teste = data_dir / "mask_teste.npy"
    caminho_pesos = modelo_dir / "stgcn_pesos_chicago.pth"
    caminho_meta = data_dir / "node_metadata.csv"
    caminho_adj = data_dir / "adj_mx.npy"
    caminho_stats = data_dir / "scaler_stats.csv"
    
    # 2. Carregamento das Estatísticas para Desnormalização (MPH)
    stats_df = pd.read_csv(caminho_stats)
    std_speed = float(stats_df.loc[stats_df['feature'] == 'speed', 'std'].values[0])
    mean_speed = float(stats_df.loc[stats_df['feature'] == 'speed', 'mean'].values[0])
    print(f"-> Estatísticas carregadas: Média = {mean_speed:.2f} | Desvio = {std_speed:.2f} (Base original em MPH)")

    HORIZON = 6
    WINDOW_SIZE = 12
    
    # 3. Carregamento dos Tensores em Memória
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
    print(f"-> Avaliando {total_janelas} janelas temporais em lotes (Batch Processing)...")
    
    registros = []
    BATCH_SIZE = 64
    FATOR_KMH = 1.60934
    
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
            
            # Inferência STGCN
            y_pred_stgcn = model(x_b, adj_tensor).cpu().numpy() # (B, H, N)
            
            # Baseline Média Móvel (MA)
            soma_x = np.sum(x_b_np * m_x_b, axis=1)
            qtd_x = np.sum(m_x_b, axis=1)
            media_via = np.divide(soma_x, qtd_x, out=np.zeros_like(soma_x), where=qtd_x != 0)
            y_pred_ma = np.repeat(media_via[:, np.newaxis, :], HORIZON, axis=1)
            
            # Baseline Histórico Global (HA)
            y_pred_ha = np.zeros_like(y_pred_ma)
            
            # Cálculo de perdas mascaradas por janela no lote
            for i in range(b_size):
                m_i = m_y_b[i]
                soma_mask = np.sum(m_i)
                if soma_mask == 0:
                    continue
                
                loss_stgcn = np.sum(np.abs(y_pred_stgcn[i] - y_b[i]) * m_i) / soma_mask
                loss_ma = np.sum(np.abs(y_pred_ma[i] - y_b[i]) * m_i) / soma_mask
                loss_ha = np.sum(np.abs(y_pred_ha[i] - y_b[i]) * m_i) / soma_mask
                
                # Erro no horizonte de 60 min (último passo da predição)
                m_i_60 = m_i[-1]
                soma_mask_60 = np.sum(m_i_60)
                loss_stgcn_60 = np.sum(np.abs(y_pred_stgcn[i, -1] - y_b[i, -1]) * m_i_60) / soma_mask_60 if soma_mask_60 > 0 else loss_stgcn
                loss_ma_60 = np.sum(np.abs(y_pred_ma[i, -1] - y_b[i, -1]) * m_i_60) / soma_mask_60 if soma_mask_60 > 0 else loss_ma
                
                # Velocidade média real observada na janela para categorizar tráfego
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
            
            if (b_start // BATCH_SIZE) % 25 == 0:
                print(f"Progresso: lote {b_start}/{total_janelas} processado...")

    df_metricas = pd.DataFrame(registros)
    
    # 4. Segregação por Percentil de Trânsito (20% mais lentos = Pico/Congestionado)
    limiar_pico = np.percentile(df_metricas["vel_media"], 20)
    df_pico = df_metricas[df_metricas["vel_media"] <= limiar_pico]
    df_livre = df_metricas[df_metricas["vel_media"] > limiar_pico]
    
    def calcular_sumario(df):
        if len(df) == 0: return 0.0, 0.0, 0.0, 0
        ha_kmh = df["mae_ha"].mean() * std_speed * FATOR_KMH
        ma_kmh = df["mae_ma"].mean() * std_speed * FATOR_KMH
        st_kmh = df["mae_stgcn"].mean() * std_speed * FATOR_KMH
        return ha_kmh, ma_kmh, st_kmh, len(df)

    gb_l, ma_l, st_l, n_l = calcular_sumario(df_livre)
    gb_p, ma_p, st_p, n_p = calcular_sumario(df_pico)
    gb_g, ma_g, st_g, n_g = calcular_sumario(df_metricas)

    print("\n" + "="*92)
    print("🔬 RELATÓRIO CIENTÍFICO DE DESEMPENHO (MÉTRICAS EM KM/H)")
    print("="*92)
    print(f"{'CENÁRIO':<20} | {'GLOBAL BASELINE (HA)':<20} | {'MÉDIA MÓVEL (MA)':<20} | {'STGCN (IA)'}")
    print("-" * 92)
    print(f"{'Trânsito Livre':<20} | {gb_l:<16.2f} km/h | {ma_l:<16.2f} km/h | {st_l:<16.2f} km/h")
    print(f"{'Pico/Congestionado':<20} | {gb_p:<16.2f} km/h | {ma_p:<16.2f} km/h | {st_p:<16.2f} km/h")
    print("-" * 92)
    print(f"{'MÉDIA GERAL':<20} | {gb_g:<16.2f} km/h | {ma_g:<16.2f} km/h | {st_g:<16.2f} km/h")
    print("="*92)
    
    # Auditoria de Longo Prazo (60 minutos)
    ma_60_kmh = df_metricas["mae_ma_60min"].mean() * std_speed * FATOR_KMH
    st_60_kmh = df_metricas["mae_stgcn_60min"].mean() * std_speed * FATOR_KMH
    print(f"\n📌 ANÁLISE DO HORIZONTE DE 60 MINUTOS:")
    print(f" -> MAE Média Móvel (60 min): {ma_60_kmh:.2f} km/h")
    print(f" -> MAE STGCN (60 min):       {st_60_kmh:.2f} km/h")
    print(f" -> Vantagem do STGCN:        {ma_60_kmh - st_60_kmh:+.2f} km/h")

    # 5. Exportação dos Resultados para a pasta de logs
    log_dir = base_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = log_dir / "metricas_finais_kmh.csv"
    
    resumo_df = pd.DataFrame({
        'Cenario': ['Transito Livre', 'Pico/Congestionado', 'Media Geral'],
        'MAE_HA_KMH': [gb_l, gb_p, gb_g],
        'MAE_MA_KMH': [ma_l, ma_p, ma_g],
        'MAE_STGCN_KMH': [st_l, st_p, st_g],
        'Amostras': [n_l, n_p, n_g]
    })
    resumo_df.to_csv(save_path, index=False)
    print(f"\n✅ Relatório científico consolidado salvo em: {save_path}")

if __name__ == "__main__":
    avaliar_modelos_unificado()