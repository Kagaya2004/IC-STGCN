import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def processar_tensores_modulares():
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    splits = ["treinamento", "validacao", "teste"]
    caminho_meta = data_dir / "node_metadata.csv"
    
    if not caminho_meta.exists():
        raise FileNotFoundError("⚠️ node_metadata.csv não encontrado. Rode filtragem_regiao.py primeiro.")
        
    print(f"-> Carregando lista canônica de nós de: {caminho_meta}")
    meta_df = pd.read_csv(caminho_meta)
    nos_globais = np.sort(meta_df['SEGMENT_ID'].unique())
    num_nodes_global = len(nos_globais)
    mapa_nos = {no: idx for idx, no in enumerate(nos_globais)}
    print(f"-> Cardinalidade Topológica Fixada: N = {num_nodes_global} segmentos viários.\n")
    
    scaler = StandardScaler()
    
    for split in splits:
        caminho_csv = data_dir / f"{split}.csv"
        
        if not caminho_csv.exists():
            continue
            
        print(f"[{split.upper()}] Carregando dados e agregando 10 min...")
        df = pd.read_csv(caminho_csv, engine='c', low_memory=False)
        
        df['SPEED'] = df['SPEED'].replace(-1, np.nan)
        df['TIME'] = pd.to_datetime(df['TIME']).dt.floor('10min')
        
        df_valido = df[df['SEGMENT_ID'].isin(mapa_nos)].copy()
        df_valido = df_valido.groupby(['TIME', 'SEGMENT_ID'], as_index=False)['SPEED'].mean()
        
        # ==============================================================
        # CORREÇÃO 2: Criação de linha do tempo contínua (Sem gaps)
        # ==============================================================
        tempo_min = df_valido['TIME'].min()
        tempo_max = df_valido['TIME'].max()
        tempos_continuos = pd.date_range(start=tempo_min, end=tempo_max, freq='10min')
        
        num_timesteps = len(tempos_continuos)
        mapa_tempos = {t: idx for idx, t in enumerate(tempos_continuos)}
        
        print(f"[{split.upper()}] Dimensões: Tempos Contínuos (T) = {num_timesteps}, Nós (N) = {num_nodes_global}")
        
        tensor_dados = np.full((num_timesteps, num_nodes_global, 1), np.nan, dtype=np.float32)
        
        indices_t = df_valido['TIME'].map(mapa_tempos).values
        indices_n = df_valido['SEGMENT_ID'].map(mapa_nos).values
        valores_v = df_valido['SPEED'].values
        
        # Filtro de segurança (Garante que só escreve indices mapeados)
        mask_validos_idx = ~np.isnan(indices_t)
        indices_t = indices_t[mask_validos_idx].astype(int)
        indices_n = indices_n[mask_validos_idx].astype(int)
        valores_v = valores_v[mask_validos_idx]

        tensor_dados[indices_t, indices_n, 0] = valores_v
        tensor_mascara = (~np.isnan(tensor_dados)).astype(np.float32)
        
        print(f"[{split.upper()}] Aplicando Standard Scaler sobre valores válidos...")
        
        if split == "treinamento":
            valores_treino_validos = tensor_dados[tensor_mascara == 1].reshape(-1, 1)
            scaler.fit(valores_treino_validos)
            
            mean_speed = float(scaler.mean_[0])
            std_speed = float(scaler.scale_[0])
            
            pd.DataFrame({'feature': ['speed'], 'mean': [mean_speed], 'std': [std_speed]}).to_csv(
                data_dir / "scaler_stats.csv", index=False
            )
            print(f"[{split.upper()}] Estatísticas reais: Média={mean_speed:.2f} MPH, Desvio={std_speed:.2f} MPH")
        
        # Normalização mantendo proteção dos nulos originais
        dados_flat = tensor_dados.flatten().reshape(-1, 1)
        mask_flat = (tensor_mascara.flatten() == 1)
        
        dados_norm = np.copy(dados_flat)
        dados_norm[mask_flat] = scaler.transform(dados_flat[mask_flat])
        tensor_dados_norm = dados_norm.reshape(tensor_dados.shape)
        
        # ==============================================================
        # CORREÇÃO 3: Imputação com zero para evitar NaN no PyTorch
        # ==============================================================
        tensor_dados_norm = np.nan_to_num(tensor_dados_norm, nan=0.0)
        
        np.save(data_dir / f"data_{split}.npy", tensor_dados_norm)
        np.save(data_dir / f"mask_{split}.npy", tensor_mascara)
        print(f"[{split.upper()}] Tensores salvos com sucesso.\n")

if __name__ == "__main__":
    processar_tensores_modulares()