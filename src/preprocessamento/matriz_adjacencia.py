import pandas as pd
import numpy as np
from pathlib import Path

def gerar_matriz_adjacencia(sigma=0.1, threshold=0.5):
    """
    Gera a matriz de adjacência espacial para os nós de Chicago.
    Garante sincronia total com os tensores gerados no dataCleaning (N=1047).
    """
    # 1. Configurações Dinâmicas de Caminho com Pathlib
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    caminho_csv = data_dir / "treinamento.csv"
    
    caminho_meta_saida = data_dir / "node_metadata.csv"
    caminho_matrix_saida = data_dir / "adj_mx.npy"

    if not caminho_csv.exists():
        print(f"Erro Crítico: {caminho_csv} não encontrado. Execute o uniao_dataset primeiro.")
        return

    print("Carregando dados de treinamento para mapeamento topológico...")
    df = pd.read_csv(caminho_csv)

    # 2. Isolamento de Metadados Únicos por Segmento (Nós do Grafo)
    print("Consolidando metadados dos nós (SEGMENT_ID)...")
    df_nodes = df.groupby('SEGMENT_ID').agg({
        'STREET': 'first',
        'DIRECTION': 'first',
        'FROM_STREET': 'first',
        'TO_STREET': 'first',
        'START_LATITUDE': 'first',
        'START_LONGITUDE': 'first'
    }).reset_index()

    # ORDENAÇÃO CRUCIAL: Deve ser idêntica à do dataCleaning para preservar o alinhamento de índices
    df_nodes = df_nodes.sort_values('SEGMENT_ID').reset_index(drop=True)
    df_nodes['NODE_INDEX'] = df_nodes.index
    num_nodes = len(df_nodes)
    print(f"-> Total de nós indexados e sincronizados: {num_nodes}")

    # 3. Construção da Matriz de Pesos via Kernel Gaussiano (Padrão STGCN)
    print("Calculando distâncias espaciais e aplicando Kernel Gaussiano...")
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    latitudes = df_nodes['START_LATITUDE'].values
    longitudes = df_nodes['START_LONGITUDE'].values

    # Matriz de distância vetorizada para economia de memória
    for i in range(num_nodes):
        # Distância euclidiana local aproximada baseada nas coordenadas
        dist_sq = (latitudes - latitudes[i])**2 + (longitudes - longitudes[i])**2
        
        # Cálculo de proximidade espacial
        weights = np.exp(-dist_sq / (sigma ** 2))
        
        # Força esparsidade zerando conexões fracas abaixo do limiar
        weights[weights < threshold] = 0.0
        adj_matrix[i, :] = weights

    # 4. Injeção de Self-Loops Matemáticos (Proteção contra divisões por zero na GCN)
    print("Injetando Self-Loops na diagonal principal...")
    np.fill_diagonal(adj_matrix, 1.0)

    # 5. Salvamento Sincronizado dos Artefatos
    df_nodes.to_csv(caminho_meta_saida, index=False)
    np.save(caminho_matrix_saida, adj_matrix)
    
    print(f"\n[SUCESSO] Mapeamento concluído com rigor dimensional:")
    print(f" -> Arquivo de Metadados: {caminho_meta_saida} ({len(df_nodes)} nós)")
    print(f" -> Matriz de Adjacência Guardada: {caminho_matrix_saida} com formato {adj_matrix.shape}")

if __name__ == "__main__":
    gerar_matriz_adjacencia()