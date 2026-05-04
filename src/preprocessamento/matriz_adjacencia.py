import pandas as pd
import numpy as np
import os

def gerar_estrutura_grafo(caminho_dataset, output_dir):
    print("Iniciando geração da estrutura do grafo...")
    
    # 1. Carregar o dataset unificado
    df = pd.read_csv(caminho_dataset)
    
    # 2. Criar o arquivo de METADADOS dos Nós
    # Pegamos apenas um registro por SEGMENT_ID para ter as coordenadas fixas
    df_nós = df.drop_duplicates('SEGMENT_ID').copy()
    df_nós = df_nós.sort_values('SEGMENT_ID').reset_index(drop=True)
    
    # Adicionamos o índice numérico (0 a 59) que a STGCN usará
    df_nós['NODE_INDEX'] = df_nós.index
    
    # Selecionar as colunas para os metadados dos nós
    metadata_cols = ['SEGMENT_ID', 'NODE_INDEX', 'START_LATITUDE', 'START_LONGITUDE', 'LENGTH', 'STREET']
    df_metadata = df_nós[metadata_cols]
    
    # Salvar Metadados (Vital para OR-Tools e Mapas)
    df_metadata.to_csv(os.path.join(output_dir, "node_metadata.csv"), index=False)
    
    # 3. Preparar mapeamento para a Matriz W
    id_para_idx = dict(zip(df_metadata['SEGMENT_ID'], df_metadata['NODE_INDEX']))
    num_nós = len(df_metadata)
    W = np.zeros((num_nós, num_nós), dtype=np.float32)

    # 4. Lógica de Conectividade Urbana (Topologia)
    print(f"Construindo conexões para {num_nós} nós...")
    
    # Conexão por Rua (Segmentos na mesma via)
    for nome_rua, grupo in df_nós.groupby('STREET'):
        segmentos = grupo['SEGMENT_ID'].tolist()
        for i in range(len(segmentos)):
            for j in range(i + 1, len(segmentos)):
                idx_i, idx_j = id_para_idx[segmentos[i]], id_para_idx[segmentos[j]]
                W[idx_i, idx_j] = 1.0
                W[idx_j, idx_i] = 1.0

    # Lógica de Cruzamento (Interseções reais)
    # Se o fim de um segmento (TO_STREET) é o nome da rua de outro, eles se cruzam
    for _, row in df_nós.iterrows():
        idx_origem = id_para_idx[row['SEGMENT_ID']]
        rua_cruzamento = row['TO_STREET']
        
        # Encontra nós que pertencem à rua de cruzamento
        vizinhos = df_nós[df_nós['STREET'] == rua_cruzamento]
        for _, v_row in vizinhos.iterrows():
            idx_destino = id_para_idx[v_row['SEGMENT_ID']]
            W[idx_origem, idx_destino] = 1.0
            W[idx_destino, idx_origem] = 1.0

    # 5. Salvar Matriz Binária (.npy)
    np.save(os.path.join(output_dir, "adj_mat_chicago.npy"), W)
    
    print("-" * 30)
    print("ESTRUTURA DO GRAFO GERADA!")
    print(f"Matriz W: {W.shape} salva em adj_mat_chicago.npy")
    print(f"Metadados: salvos em node_metadata.csv")
    print("-" * 30)
    
    return W

if __name__ == "__main__":
    caminho_csv = r"C:\IC\Aplication\data\tratados\dataset_unificado_stgcn.csv"
    pasta_saida = r"C:\IC\Aplication\data\tratados"
    gerar_estrutura_grafo(caminho_csv, pasta_saida)