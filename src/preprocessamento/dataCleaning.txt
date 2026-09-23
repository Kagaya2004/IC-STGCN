import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def processar_tensores_modulares():
    # 1. Configurações Dinâmicas de Caminho via Pathlib
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    splits = ["treinamento", "validacao", "teste"]
    caminho_meta = data_dir / "node_metadata.csv"
    
    # 2. Definição Canônica e Global dos Nós (Garantia de N_train == N_val == N_test)
    if caminho_meta.exists():
        print(f"-> Carregando lista canônica de nós de: {caminho_meta}")
        meta_df = pd.read_csv(caminho_meta)
        nos_globais = np.sort(meta_df['SEGMENT_ID'].unique())
    else:
        # Fallback: extrai a união estrita a partir do treino caso o metadata não exista
        print("⚠️ node_metadata.csv não encontrado. Extraindo nós de treinamento.csv...")
        df_treino_temp = pd.read_csv(data_dir / "treinamento.csv", usecols=['SEGMENT_ID'])
        nos_globais = np.sort(df_treino_temp['SEGMENT_ID'].unique())
        
    num_nodes_global = len(nos_globais)
    mapa_nos = {no: idx for idx, no in enumerate(nos_globais)}
    print(f"-> Cardinalidade Topológica Fixada: N = {num_nodes_global} segmentos viários.\n")
    
    # 3. Inicialização do Scaler (Ajustado estritamente no treino)
    scaler = StandardScaler()
    
    print("Iniciando processamento de tensores STGCN...\n")
    
    for split in splits:
        caminho_csv = data_dir / f"{split}.csv"
        
        if not caminho_csv.exists():
            print(f"⚠️ Aviso: Arquivo base não encontrado: {caminho_csv}")
            continue
            
        print(f"[{split.upper()}] Carregando dados...")
        df = pd.read_csv(caminho_csv, engine='c', low_memory=False)
        
        # Tratamento de leituras inválidas da CTA (-1 = ausência de leitura)
        df['SPEED'] = df['SPEED'].replace(-1, np.nan)
        
        # Sincronização temporal em blocos de 10 minutos
        print(f"[{split.upper()}] Sincronizando relógios temporais (blocos de 10min)...")
        df['TIME'] = pd.to_datetime(df['TIME'])
        df['TIME'] = df['TIME'].dt.floor('10min')
        
        # Agregação de redundâncias espaciais/temporais
        df = df.groupby(['TIME', 'SEGMENT_ID'], as_index=False)['SPEED'].mean()
        
        # Extração das Dimensões Temporais do split
        tempos_unicos = np.sort(df['TIME'].unique())
        num_timesteps = len(tempos_unicos)
        mapa_tempos = {t: idx for idx, t in enumerate(tempos_unicos)}
        
        print(f"[{split.upper()}] Dimensões: Tempos (T) = {num_timesteps}, Nós Fixos (N) = {num_nodes_global}")
        
        # Alocação de memória (float32 para respeitar teto de 6GB VRAM)
        tensor_dados = np.full((num_timesteps, num_nodes_global, 1), np.nan, dtype=np.float32)
        
        # Filtra registros que pertencem ao conjunto canônico de nós
        df_valido = df[df['SEGMENT_ID'].isin(mapa_nos)].copy()
        
        indices_t = df_valido['TIME'].map(mapa_tempos).values
        indices_n = df_valido['SEGMENT_ID'].map(mapa_nos).values
        valores_v = df_valido['SPEED'].values
        
        tensor_dados[indices_t, indices_n, 0] = valores_v

        # Geração da Máscara Binária (1.0 = dado válido observado; 0.0 = ausente/sensor falho)
        tensor_mascara = (~np.isnan(tensor_dados)).astype(np.float32)
        
        # Normalização com Proteção contra Vazamento e Data Poisoning
        print(f"[{split.upper()}] Aplicando Standard Scaler sobre valores válidos...")
        valores_validos = tensor_dados[tensor_mascara == 1].reshape(-1, 1)
        
        if split == "treinamento":
            scaler.fit(valores_validos)
            
            mean_speed = float(scaler.mean_[0])
            std_speed = float(scaler.scale_[0])
            
            stats_df = pd.DataFrame({
                'feature': ['speed'],
                'mean': [mean_speed],
                'std': [std_speed]
            })
            caminho_stats = data_dir / "scaler_stats.csv"
            stats_df.to_csv(caminho_stats, index=False)
            print(f"[{split.upper()}] ✅ Estatísticas reais salvas: Média={mean_speed:.2f} MPH, Desvio={std_speed:.2f} MPH")
        
        # Aplica transformação mantendo integridade nos NaNs
        dados_flat = np.nan_to_num(tensor_dados.flatten()).reshape(-1, 1)
        dados_norm = scaler.transform(dados_flat)
        tensor_dados_norm = dados_norm.reshape(tensor_dados.shape)
        
        # Restaura os NaNs nas posições de sensores ausentes
        tensor_dados_norm = np.where(tensor_mascara == 1, tensor_dados_norm, np.nan)
        
        # Persistência dos tensores em disco
        caminho_dados_npy = data_dir / f"data_{split}.npy"
        caminho_mask_npy = data_dir / f"mask_{split}.npy"
        
        np.save(caminho_dados_npy, tensor_dados_norm)
        np.save(caminho_mask_npy, tensor_mascara)
        print(f"[{split.upper()}] Tensores salvos com sucesso.\n")

if __name__ == "__main__":
    processar_tensores_modulares()