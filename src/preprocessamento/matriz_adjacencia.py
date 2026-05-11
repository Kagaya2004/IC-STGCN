import pandas as pd
import numpy as np
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância em metros entre dois pontos (Haversine)."""
    R = 6371000  # Raio da Terra em metros
    phi1, phi2 = np.radians(lat1), np.numeric(lat2) if not isinstance(lat2, str) else np.radians(float(lat2))
    # Simplificação para uso com arrays numpy
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def gerar_estrutura_grafo_ponderada(caminho_dataset, output_dir):
    print("🚀 Iniciando geração da estrutura de grafo ponderado...")
    
    # 1. Carregar o dataset unificado
    df = pd.read_csv(caminho_dataset)
    df['TIME'] = pd.to_datetime(df['TIME'])

    # 2. MELHORIA 1: Interseção de IDs entre Anos
    # Garante que só usaremos sensores que sobreviveram ao "gap" de 9 meses
    sensores_2023 = set(df[df['TIME'].dt.year == 2023]['SEGMENT_ID'].unique())
    sensores_2024 = set(df[df['TIME'].dt.year >= 2024]['SEGMENT_ID'].unique())
    sensores_validos = sorted(list(sensores_2023.intersection(sensores_2024)))
    
    print(f"📊 Sensores em 2023: {len(sensores_2023)} | 2024+: {len(sensores_2024)}")
    print(f"🔗 Interseção (sensores persistentes): {len(sensores_validos)}")

    # Filtrar metadados apenas para os sensores que existem em ambos os blocos
    df_nós = df[df['SEGMENT_ID'].isin(sensores_validos)].drop_duplicates('SEGMENT_ID').copy()
    df_nós = df_nós.sort_values('SEGMENT_ID').reset_index(drop=True)
    df_nós['NODE_INDEX'] = df_nós.index
    
    # Salvar Metadados Atualizados
    df_metadata = df_nós[['SEGMENT_ID', 'NODE_INDEX', 'START_LATITUDE', 'START_LONGITUDE', 'STREET']]
    df_metadata.to_csv(os.path.join(output_dir, "node_metadata.csv"), index=False)
    
    # 3. Inicializar Matriz W (Ponderada)
    num_nós = len(df_metadata)
    id_para_idx = dict(zip(df_metadata['SEGMENT_ID'], df_metadata['NODE_INDEX']))
    W = np.zeros((num_nós, num_nós))

    # 4. MELHORIA 2 e 3: Conectividade Direcionada e Gaussian Kernel
    # sigma (distância padrão) para o kernel gaussiano (aprox. 10km na literatura de tráfego)
    sigma2 = 10000**2 
    epsilon = 0.1 # Threshold para evitar matrizes muito esparsas ou densas

    print("🛰️ Calculando adjacência espacial...")
    
    # Lógica de cruzamento e fluxo (Direcionado)
    # Se o fim do segmento A (TO_STREET) é a rua do segmento B (STREET), há fluxo de A para B
    if 'TO_STREET' in df.columns:
        for _, row in df_nós.iterrows():
            idx_i = id_para_idx[row['SEGMENT_ID']]
            rua_destino = row['TO_STREET']
            
            # Vizinhos que estão na rua para onde o tráfego aponta
            vizinhos = df_nós[df_nós['STREET'] == rua_destino]
            
            for _, v_row in vizinhos.iterrows():
                idx_j = id_para_idx[v_row['SEGMENT_ID']]
                
                # Calcular distância física para o peso
                dist = haversine_distance(
                    row['START_LATITUDE'], row['START_LONGITUDE'],
                    v_row['START_LATITUDE'], v_row['START_LONGITUDE']
                )
                
                # Gaussian Kernel: W_ij = exp(-d^2 / sigma^2)
                peso = np.exp(-(dist**2) / sigma2)
                
                if peso >= epsilon:
                    W[idx_i, idx_j] = peso # Grafo direcionado: A influencia B

    # 5. Salvar Matriz NumPy (.npy)
    np.save(os.path.join(output_dir, "adj_mx_chicago.npy"), W)
    print(f"✅ Sucesso! Matriz ponderada {W.shape} gerada com {np.count_nonzero(W)} conexões.")

if __name__ == "__main__":
    base = r"C:\IC\Aplication"
    gerar_estrutura_grafo_ponderada(
        os.path.join(base, "data", "tratados", "dataset_unificado_stgcn.csv"),
        os.path.join(base, "data", "tratados")
    )