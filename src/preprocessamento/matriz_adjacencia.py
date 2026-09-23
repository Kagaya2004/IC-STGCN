import pandas as pd
import numpy as np
import math
from pathlib import Path

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância geodésica em metros utilizando a curvatura da Terra.
    """
    R = 6371000 # Raio médio da Terra em metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def gerar_matriz_adjacencia(threshold=0.5):
    """
    Gera a matriz de adjacência espacial aplicando um Kernel Gaussiano
    sobre as distâncias reais de Haversine. Garante sincronia (N) com os tensores.
    """
    print("Iniciando formulação da Matriz de Adjacência Geodésica...")
    
    # 1. Configurações Dinâmicas de Caminho
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    caminho_meta = data_dir / "node_metadata.csv"
    
    if not caminho_meta.exists():
        raise FileNotFoundError(f"⚠️ Erro fatal: Arquivo canônico não encontrado em {caminho_meta}.")
        
    # 2. Definição Global e Ordenada dos Nós
    print("-> Carregando metadados espaciais...")
    meta_df = pd.read_csv(caminho_meta)
    
    # A ordenação é CRÍTICA para que o índice coincida com o do dataCleaning.py
    meta_df = meta_df.sort_values(by='SEGMENT_ID').reset_index(drop=True)
    num_nodes = len(meta_df)
    print(f"-> Grafo topológico fixado: N = {num_nodes} segmentos viários.")
    
    # 3. Extração Espacial
    lats = meta_df['START_LATITUDE'].values
    lons = meta_df['START_LONGITUDE'].values
    
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # 4. Cálculo Físico de Distância (O(N^2))
    print("-> Calculando distâncias de Haversine (metros)...")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = calcular_distancia_haversine(lats[i], lons[i], lats[j], lons[j])
                
    # 5. Aplicação do Kernel Gaussiano
    # Utiliza o desvio padrão das distâncias válidas (ruas distintas) como Sigma para equilibrar a decaída
    distancias_validas = dist_matrix[dist_matrix > 0]
    sigma = np.std(distancias_validas) if len(distancias_validas) > 0 else 1000.0
    print(f"-> Aplicando Kernel Gaussiano (Sigma Espacial: {sigma:.2f}m)...")
    
    # W_ij = exp( -(dist_ij / sigma)^2 )
    adj_matrix = np.exp(-np.square(dist_matrix / sigma))
    
    # 6. Esparsidade (Corte Threshold)
    # Zera conexões muito fracas (ruas muito distantes) para reduzir o custo computacional na STGCN
    adj_matrix[adj_matrix < threshold] = 0.0
    
    # 7. Auto-Conexão (Identidade)
    # A GCN exige que o nó passe informação para si mesmo
    np.fill_diagonal(adj_matrix, 1.0)
    
    caminho_adj = data_dir / "adj_mx.npy"
    np.save(caminho_adj, adj_matrix.astype(np.float32))
    print(f"✅ Matriz de Adjacência Científica salva em: {caminho_adj}")

if __name__ == "__main__":
    gerar_matriz_adjacencia()