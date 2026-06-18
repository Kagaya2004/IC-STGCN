import pandas as pd
import numpy as np
import os

def gerar_matriz_adjacencia():
    # 1. Configuracoes de Caminho
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    caminho_csv = os.path.join(data_dir, "dataset_filtrado.csv")
    
    caminho_adj = os.path.join(data_dir, "adj_mx_chicago.npy")
    caminho_meta = os.path.join(data_dir, "node_metadata.csv")

    if not os.path.exists(caminho_csv):
        print(f"Erro: Arquivo {caminho_csv} nao encontrado.")
        return

    print("Carregando dados para extracao de topologia...")
    cols_topo = ['SEGMENT_ID', 'STREET', 'START_LATITUDE', 'START_LONGITUDE']
    df = pd.read_csv(caminho_csv, usecols=cols_topo)

    # 2. Limpeza e Conversao de Coordenadas
    # Garante que latitude e longitude sejam tratadas como numeros (float)
    for col in ['START_LATITUDE', 'START_LONGITUDE']:
        # Remove possiveis espacos, troca virgula por ponto e converte para numerico
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove linhas onde a conversao falhou (valores nulos)
    df = df.dropna(subset=['START_LATITUDE', 'START_LONGITUDE'])

    # 3. Identificacao de Nos Unicos
    nos = df.drop_duplicates(subset=['SEGMENT_ID']).copy()
    nos = nos.sort_values('SEGMENT_ID').reset_index(drop=True)
    nos['NODE_INDEX'] = nos.index
    num_nos = len(nos)
    
    print(f"Total de nos identificados: {num_nos}")

    # 4. Exportacao de Metadados
    nos.to_csv(caminho_meta, index=False)
    print(f"Metadados salvos em: {caminho_meta}")

    # 5. Calculo da Matriz de Adjacencia Ponderada
    adj_mx = np.zeros((num_nos, num_nos), dtype=np.float32)
    coords = nos[['START_LATITUDE', 'START_LONGITUDE']].values

    print("Calculando pesos de adjacencia (Gaussian Kernel)...")
    
    # Hiperparametros do Grafo
    d_threshold = 0.01 
    sigma_sq = 0.0001 

    for i in range(num_nos):
        for j in range(num_nos):
            if i == j:
                continue
            
            # Agora os valores sao numericos, permitindo a operacao de subtracao vetorial
            dist = np.linalg.norm(coords[i] - coords[j])
            
            if dist <= d_threshold:
                peso = np.exp(-(dist**2) / sigma_sq)
                adj_mx[i, j] = peso

    # 6. Salvamento
    np.save(caminho_adj, adj_mx)
    
    print(f"Sucesso: Matriz de adjacencia {num_nos}x{num_nos} gerada.")
    print(f"Arquivo binario salvo em: {caminho_adj}")

if __name__ == "__main__":
    gerar_matriz_adjacencia()