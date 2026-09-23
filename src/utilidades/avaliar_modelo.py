import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os

from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader import ChicagoTrafficDataset

def masked_mae_numpy(y_pred, y_true, mask):
    """Calcula o Erro Absoluto Médio (MAE) ignorando os dados ausentes via máscara."""
    mask = mask.astype(np.float32)
    mask_sum = np.sum(mask)
    if mask_sum == 0:
        return 0.0
    loss = np.abs(y_pred - y_true) * mask
    return np.sum(loss) / mask_sum

def avaliar_modelos():
    print("Iniciando Avaliação Unificada de Modelos (STGCN vs Baselines)...")

    # 1. Configuração de Caminhos via Pathlib
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    modelo_dir = base_dir / "src" / "modelo"

    caminho_dados_teste = data_dir / "data_teste.npy"
    caminho_mask_teste = data_dir / "mask_teste.npy"
    caminho_pesos = modelo_dir / "stgcn_pesos_chicago.pth"
    caminho_meta = data_dir / "node_metadata.csv"
    caminho_adj = data_dir / "adj_mx.npy"

    # 2. Carregar Estatísticas para Desnormalização
    stats = pd.read_csv(data_dir / "scaler_stats.csv")
    mean_speed = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_speed = stats.loc[stats['feature'] == 'speed', 'std'].values[0]
    print(f"-> Parâmetros Originais de Desnormalização: Média={mean_speed:.2f} MPH, Desvio={std_speed:.2f} MPH")

    HORIZON = 6
    WINDOW_SIZE = 12

    dataset_teste = ChicagoTrafficDataset(caminho_dados_teste, caminho_mask_teste, window_size=WINDOW_SIZE, horizon=HORIZON)
    meta_df = pd.read_csv(caminho_meta)
    num_nodes = len(meta_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN_TrafficModel(in_channels=1, hidden_channels=64, out_channels=HORIZON, num_nodes=num_nodes).to(device)
    model.load_state_dict(torch.load(caminho_pesos, map_location=device, weights_only=True))
    model.eval()

    adj_mx = np.load(caminho_adj)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)

    metricas_janelas = []

    print(f"Calculando inferências e aplicando conversão para KM/H em {len(dataset_teste)} janelas...")

    with torch.no_grad():
        for i in range(len(dataset_teste)):
            x_tensor, y_tensor, mask_tensor = dataset_teste[i]
            x_input = x_tensor.unsqueeze(0).to(device)
            
            # --- MODELO 1: STGCN (IA) ---
            y_pred_stgcn_norm = model(x_input, adj_tensor).squeeze(0).cpu().numpy()
            
            # --- MODELO 2: Média Móvel (MA) ---
            x_numpy = x_tensor.numpy().squeeze(-1) 
            mask_x = (~np.isclose(x_numpy, 0.0)).astype(np.float32)
            elementos_validos = x_numpy[mask_x.astype(bool)]
            
            soma_historica = np.sum(x_numpy, axis=0)
            contagem_validos = np.sum(mask_x, axis=0)
            media_historica_por_via = np.divide(soma_historica, contagem_validos, out=np.zeros_like(soma_historica), where=contagem_validos!=0)
            y_pred_ma_norm = np.tile(media_historica_por_via, (HORIZON, 1)) 
            
            # --- MODELO 3: Média Histórica Global (HA) ---
            y_pred_global_norm = np.zeros_like(y_pred_ma_norm)
            
            # Alvos e Máscaras
            y_true_norm = y_tensor.numpy().squeeze(-1)
            mask_y_numpy = mask_tensor.numpy().squeeze(-1)
            
            # --- DESNORMALIZAÇÃO PARA UNIDADES REAIS E CONVERSÃO PARA KM/H ---
            FATOR_KMH = 1.60934
            
            y_pred_stgcn_kmh = ((y_pred_stgcn_norm * std_speed) + mean_speed) * FATOR_KMH
            y_pred_ma_kmh = ((y_pred_ma_norm * std_speed) + mean_speed) * FATOR_KMH
            y_pred_global_kmh = ((y_pred_global_norm * std_speed) + mean_speed) * FATOR_KMH
            y_true_kmh = ((y_true_norm * std_speed) + mean_speed) * FATOR_KMH
            
            velocidade_media_janela_kmh = np.mean((x_numpy[mask_x.astype(bool)] * std_speed + mean_speed) * FATOR_KMH) if len(elementos_validos) > 0 else (mean_speed * FATOR_KMH)
            
            # --- CÁLCULO DO MAE MASCARADO EM KM/H ---
            mae_stgcn = masked_mae_numpy(y_pred_stgcn_kmh, y_true_kmh, mask_y_numpy)
            mae_ma = masked_mae_numpy(y_pred_ma_kmh, y_true_kmh, mask_y_numpy)
            mae_global = masked_mae_numpy(y_pred_global_kmh, y_true_kmh, mask_y_numpy)
            
            metricas_janelas.append({
                "mae_stgcn_kmh": mae_stgcn,
                "mae_ma_kmh": mae_ma,
                "mae_global_kmh": mae_global,
                "vel_media_kmh": velocidade_media_janela_kmh
            })
            
            if i % 1000 == 0 and i > 0:
                print(f"Progresso: {i}/{len(dataset_teste)} janelas...")

    df_metricas = pd.DataFrame(metricas_janelas)

    # Separação por Percentil de Trânsito (20% mais lentos = Pico/Congestionado)
    limiar_pico = np.percentile(df_metricas["vel_media_kmh"], 20)
    df_pico = df_metricas[df_metricas["vel_media_kmh"] <= limiar_pico]
    df_livre = df_metricas[df_metricas["vel_media_kmh"] > limiar_pico]

    def calcular_sumario(df):
        if len(df) == 0: return 0.0, 0.0, 0.0, 0
        return df["mae_global_kmh"].mean(), df["mae_ma_kmh"].mean(), df["mae_stgcn_kmh"].mean(), len(df)

    gb_g, ma_g, st_g, n_g = calcular_sumario(df_metricas)
    gb_p, ma_p, st_p, n_p = calcular_sumario(df_pico)
    gb_l, ma_l, st_l, n_l = calcular_sumario(df_livre)

    print("\n" + "="*85)
    print("🔬 RELATÓRIO FINAL DE DESEMPENHO (MÉTRICAS EM KM/H)")
    print("="*85)
    print(f"{'CENÁRIO':<20} | {'GLOBAL BASELINE (HA)':<20} | {'MÉDIA MÓVEL (MA)':<20} | {'STGCN (IA)'}")
    print("-" * 85)
    print(f"{'Trânsito Livre':<20} | {gb_l:<20.4f} | {ma_l:<20.4f} | {st_l:<20.4f}")
    print(f"{'Pico/Congestionado':<20} | {gb_p:<20.4f} | {ma_p:<20.4f} | {st_p:<20.4f}")
    print("-" * 85)
    print(f"{'MÉDIA GERAL':<20} | {gb_g:<20.4f} | {ma_g:<20.4f} | {st_g:<20.4f}")
    print("="*85)

    # Salvar para o artigo
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
    print(f"\n✅ Resultados unificados salvos em: {save_path}")

if __name__ == "__main__":
    avaliar_modelos()