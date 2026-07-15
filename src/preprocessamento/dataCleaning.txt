import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def processar_tensores_modulares():
    # 1. Configurações Dinâmicas de Caminho com Pathlib
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    splits = ["treinamento", "validacao", "teste"]
    
    # 2. Inicialização do Scaler (Prevenção de Data Leakage)
    scaler = StandardScaler()
    
    print("Iniciando processamento de tensores STGCN...\n")
    
    for split in splits:
        caminho_csv = data_dir / f"{split}.csv"
        
        if not caminho_csv.exists():
            print(f"Erro: Arquivo base não encontrado: {caminho_csv}")
            continue
            
        print(f"[{split.upper()}] Carregando dados...")
        df = pd.read_csv(caminho_csv, engine='c', low_memory=False)
        
        # 1. Correção Matemática Crítica: Arredondamento Temporal
        print(f"[{split.upper()}] Sincronizando relógios para janelas de 10 minutos...")
        df['TIME'] = pd.to_datetime(df['TIME'])
        df['TIME'] = df['TIME'].dt.floor('10min')
        
        # 2. Agrupamento de redundâncias (Média das velocidades no mesmo bloco de 10 min)
        print(f"[{split.upper()}] Agregando possíveis leituras redundantes...")
        df = df.groupby(['TIME', 'SEGMENT_ID'], as_index=False)['SPEED'].mean()
        
        # Extração das Dimensões Reais (Agora sincronizadas e ordenadas com NumPy)
        tempos_unicos = np.sort(df['TIME'].unique())
        num_timesteps = len(tempos_unicos)
        
        nos_unicos = np.sort(df['SEGMENT_ID'].unique())
        num_nodes = len(nos_unicos)
        
        print(f"[{split.upper()}] Dimensões corrigidas: Tempos (T) = {num_timesteps}, Nós (N) = {num_nodes}")
        
        # 3. Mapeamento Rápido em Dicionários
        mapa_tempos = {t: idx for idx, t in enumerate(tempos_unicos)}
        mapa_nos = {no: idx for idx, no in enumerate(nos_unicos)}
        
        # Construção dos Tensores - Alocação Direta na Memória
        print(f"[{split.upper()}] Preenchendo tensor direto na memória (Bypass de RAM)...")
        tensor_dados = np.full((num_timesteps, num_nodes, 1), np.nan, dtype=np.float32)
        
        # Extração de índices e valores
        indices_t = df['TIME'].map(mapa_tempos).values
        indices_n = df['SEGMENT_ID'].map(mapa_nos).values
        valores_v = df['SPEED'].values
        
        # Preenchimento vetorizado instantâneo
        tensor_dados[indices_t, indices_n, 0] = valores_v

        # Criação da Máscara (1 para dado real, 0 para NaN)
        print(f"[{split.upper()}] Gerando máscara de ausência...")
        tensor_mascara = (~np.isnan(tensor_dados)).astype(np.float32)
        
        # Normalização com Proteção de Leakage
        print(f"[{split.upper()}] Aplicando Standard Scaler...")
        dados_flat = np.nan_to_num(tensor_dados.flatten()).reshape(-1, 1)
        
        if split == "treinamento":
            dados_norm = scaler.fit_transform(dados_flat)
        else:
            dados_norm = scaler.transform(dados_flat)
            
        tensor_dados_norm = dados_norm.reshape(tensor_dados.shape)
        tensor_dados_norm = np.where(tensor_mascara == 1, tensor_dados_norm, np.nan)
        
        # Salvamento
        caminho_dados_npy = data_dir / f"data_{split}.npy"
        caminho_mask_npy = data_dir / f"mask_{split}.npy"
        
        np.save(caminho_dados_npy, tensor_dados_norm)
        np.save(caminho_mask_npy, tensor_mascara)
        
        print(f"[{split.upper()}] Concluído. Tensores salvos.\n")

if __name__ == "__main__":
    processar_tensores_modulares()