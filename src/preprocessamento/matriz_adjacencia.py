import pandas as pd
import numpy as np
import os

def gerar_matriz_adjacencia():
    # 1. Configuracoes de Caminho
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    caminho_csv = os.path.join(data_dir, "dataset_filtrado.csv")
    
    caminho_meta_saida = os.path.join(data_dir, "node_metadata.csv")
    caminho_matrix_saida = os.path.join(data_dir, "adj_mx.npy")

    if not os.path.exists(caminho_csv):
        print("Erro: dataset_filtrado.csv nao encontrado. Execute o filtro primeiro.")
        return

    print("Carregando dados filtrados para mapeamento de nos...")
    df = pd.read_csv(caminho_csv)

    # 2. Isolamento de Metadados Unicos por Segmento (Nos do Grafo)
    # Agrupa por SEGMENT_ID para obter os atributos espaciais estaticos
    print("Consolidando metadados dos nos (SEGMENT_ID)...")
    df_nodes = df.groupby('SEGMENT_ID').agg({
        'STREET': 'first',
        'DIRECTION': 'first',
        'FROM_STREET': 'first',
        'TO_STREET': 'first',
        'START_LATITUDE': 'first',
        'START_LONGITUDE': 'first'
    }).reset_index()

    # Ordena de forma deterministica e cria o indice sequencial para as matrizes
    df_nodes = df_nodes.sort_values('SEGMENT_ID').reset_index(drop=True)
    df_nodes['NODE_INDEX'] = df_nodes.index
    
    # Salva o arquivo de metadados exigido pelo dataCleaning.py
    df_nodes.to_csv(caminho_meta_saida, index=False)
    print(f"Metadados salvos em: {caminho_meta_saida}. Total de nos (N): {len(df_nodes)}")

    num_nodes = len(df_nodes)
    
    # 3. Calculo Vetorizado da Distancia Haversine (Fisica)
    print("Calculando matriz de distancias fisicas via NumPy Broadcasting...")
    lat = np.radians(df_nodes['START_LATITUDE'].to_numpy())
    lon = np.radians(df_nodes['START_LONGITUDE'].to_numpy())

    # Expansao de eixos para forcar a operacao matricial NxN
    lat_i, lat_j = lat[:, np.newaxis], lat[np.newaxis, :]
    lon_i, lon_j = lon[:, np.newaxis], lon[np.newaxis, :]

    dlat = lat_j - lat_i
    dlon = lon_j - lon_i

    # Formula Haversine estruturada em matrizes
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_i) * np.cos(lat_j) * np.sin(dlon / 2.0)**2
    dist_matrix = 2 * 6371.0 * np.arcsin(np.sqrt(a)) # Distancia em quilometros

    # Aplicacao do decaimento Gaussiano (Parametro sigma regulável, ex: 10km)
    sigma = 10.0
    weight_matrix = np.exp(- (dist_matrix ** 2) / (sigma ** 2))

    # 4. Construcao Vetorizada da Conectividade Logica (Restricoes de Chicago)
    print("Construindo restricoes logicas de cruzamento de vias...")
    
    # Criamos uma matriz de indices para cruzamento
    idx = np.arange(num_nodes)
    idx_i, idx_j = np.meshgrid(idx, idx, indexing='ij')

    # Extracao dos vetores de texto para alinhamento matricial
    street_arr = df_nodes['STREET'].to_numpy()
    direction_arr = df_nodes['DIRECTION'].to_numpy()
    from_arr = df_nodes['FROM_STREET'].to_numpy()
    to_arr = df_nodes['TO_STREET'].to_numpy()

    # Regra Lógica 1: Continuidade na mesma via (Destino de i alimenta Origem de j)
    mask_continuidade = (street_arr[idx_i] == street_arr[idx_j]) & \
                        (direction_arr[idx_i] == direction_arr[idx_j]) & \
                        (to_arr[idx_i] == from_arr[idx_j])

    # Regra Lógica 2: Intersecao de vias (A rua onde i termina cruza ou vira a rua principal de j)
    mask_intersecao = (to_arr[idx_i] == street_arr[idx_j]) | \
                      (from_arr[idx_j] == street_arr[idx_i])

    # Fusao das regras logicas (Matriz Binaria de Conectividade)
    logical_adjacency = (mask_continuidade | mask_intersecao).astype(np.float32)

    # 5. Multiplicacao de Hadamard (Fusao Hibrida) e Remocao de Self-Loops
    print("Fusao das restricoes e salvamento da matriz de adjacencia...")
    adj_matrix = weight_matrix * logical_adjacency
    
    # Garante que a diagonal principal seja zero (nos nao influenciam a si mesmos na convolucao espacial)
    np.fill_diagonal(adj_matrix, 0.0)

    # Salvamento do array binario
    np.save(caminho_matrix_saida, adj_matrix)
    print(f"Matriz de Adjacencia NxN salva com sucesso em: {caminho_matrix_saida} com formato {adj_matrix.shape}")

if __name__ == "__main__":
    gerar_matriz_adjacencia()