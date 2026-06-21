import pandas as pd
import numpy as np
import networkx as nx
import os

def gerar_matriz_adjacencia():
    # 1. Configurações de Caminho
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    caminho_csv = os.path.join(data_dir, "dataset_filtrado.csv")
    
    caminho_meta_saida = os.path.join(data_dir, "node_metadata.csv")
    caminho_matrix_saida = os.path.join(data_dir, "adj_mx.npy")

    if not os.path.exists(caminho_csv):
        print("Erro: dataset_filtrado.csv não encontrado. Execute o filtro primeiro.")
        return

    print("Carregando dados filtrados para mapeamento de nós...")
    df = pd.read_csv(caminho_csv)

    # 2. Isolamento de Metadados Únicos por Segmento (Nós do Grafo)
    print("Consolidando metadados iniciais dos nós (SEGMENT_ID)...")
    df_nodes = df.groupby('SEGMENT_ID').agg({
        'STREET': 'first',
        'DIRECTION': 'first',
        'FROM_STREET': 'first',
        'TO_STREET': 'first',
        'START_LATITUDE': 'first',
        'START_LONGITUDE': 'first'
    }).reset_index()

    # Ordenação determinística inicial
    df_nodes = df_nodes.sort_values('SEGMENT_ID').reset_index(drop=True)
    num_nodes_inicial = len(df_nodes)

    # 3. Cálculo Vetorizado da Distância Haversine (Física)
    print("Calculando matriz de distâncias físicas via NumPy Broadcasting...")
    lat = np.radians(df_nodes['START_LATITUDE'].to_numpy())
    lon = np.radians(df_nodes['START_LONGITUDE'].to_numpy())

    lat_i, lat_j = lat[:, np.newaxis], lat[np.newaxis, :]
    lon_i, lon_j = lon[:, np.newaxis], lon[np.newaxis, :]

    dlat = lat_j - lat_i
    dlon = lon_j - lon_i

    a = np.sin(dlat / 2.0)**2 + np.cos(lat_i) * np.cos(lat_j) * np.sin(dlon / 2.0)**2
    dist_matrix = 2 * 6371.0 * np.arcsin(np.sqrt(a)) # Distância em quilômetros

    # Aplicação do decaimento Gaussiano (constante sigma = 10km)
    sigma = 10.0
    weight_matrix = np.exp(- (dist_matrix ** 2) / (sigma ** 2))

    # 4. Construção Vetorizada da Conectividade Lógica
    print("Construindo restrições lógicas de cruzamento de vias...")
    idx = np.arange(num_nodes_inicial)
    idx_i, idx_j = np.meshgrid(idx, idx, indexing='ij')

    street_arr = df_nodes['STREET'].to_numpy()
    direction_arr = df_nodes['DIRECTION'].to_numpy()
    from_arr = df_nodes['FROM_STREET'].to_numpy()
    to_arr = df_nodes['TO_STREET'].to_numpy()

    # Regra Lógica 1: Continuidade na mesma via (Destino de i alimenta Origem de j)
    mask_continuidade = (street_arr[idx_i] == street_arr[idx_j]) & \
                        (direction_arr[idx_i] == direction_arr[idx_j]) & \
                        (to_arr[idx_i] == from_arr[idx_j])

    # Regra Lógica 2: Interseção de vias (A rua onde i termina cruza a rua principal de j)
    mask_intersecao = (to_arr[idx_i] == street_arr[idx_j]) | \
                      (from_arr[idx_j] == street_arr[idx_i])

    logical_adjacency = (mask_continuidade | mask_intersecao).astype(np.float32)

    # Fusão de Hadamard (Pesos Físicos * Filtro Lógico)
    adj_matrix_bruta = weight_matrix * logical_adjacency
    np.fill_diagonal(adj_matrix_bruta, 0.0)

    # 5. ETAPA NOVA: Extração do Maior Componente Conexo (LCC)
    print("Analisando conectividade global para eliminar ilhas topológicas (LCC)...")
    # Instancia o grafo direcionado no NetworkX a partir da matriz bruta para análise
    G_temp = nx.from_numpy_array(adj_matrix_bruta, create_using=nx.DiGraph)
    
    # Encontra as componentes fracamente conexas (garante conectividade urbana arterial)
    componentes = list(nx.weakly_connected_components(G_temp))
    maior_componente = max(componentes, key=len)
    nos_validos_indices = sorted(list(maior_componente))
    
    num_nodes_final = len(nos_validos_indices)
    print(f"-> Malha original: {num_nodes_inicial} nós.")
    print(f"-> Nós retidos no componente principal: {num_nodes_final}")
    print(f"-> Nós órfãos removidos: {num_nodes_inicial - num_nodes_final}")

    # 6. Re-indexação e Filtragem Definitiva
    df_nodes_limpos = df_nodes.iloc[nos_validos_indices].reset_index(drop=True)
    df_nodes_limpos['NODE_INDEX'] = df_nodes_limpos.index
    
    # Reduz a matriz NxN para conter apenas as linhas e colunas do componente conectado
    adj_matrix_final = adj_matrix_bruta[nos_validos_indices][:, nos_validos_indices]

    # 7. Salvamento Sincronizado
    df_nodes_limpos.to_csv(caminho_meta_saida, index=False)
    np.save(caminho_matrix_saida, adj_matrix_final)
    
    print(f"\nSucesso! Arquivos salvos em {data_dir}:")
    print(f"- Metadados limpos: 'node_metadata.csv' (Formato {df_nodes_limpos.shape})")
    print(f"- Matriz de Adjacência: 'adj_mx.npy' (Formato {adj_matrix_final.shape})")

if __name__ == "__main__":
    gerar_matriz_adjacencia()