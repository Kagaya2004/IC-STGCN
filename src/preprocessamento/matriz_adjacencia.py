import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pathlib import Path

def gerar_matriz_adjacencia(caminho_csv, caminho_adj, caminho_meta):
    print("--- FASE 1: GERAÇÃO DA TOPOLOGIA ESPACIAL ---")
    if not caminho_csv.exists():
        print(f"Erro: Arquivo {caminho_csv} não encontrado.")
        return False

    # Leitura com as colunas espaciais exigidas
    cols_req = ['SEGMENT_ID', 'LENGTH', 'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE']
    try:
        df_bruto = pd.read_csv(caminho_csv, usecols=cols_req)
    except ValueError as e:
        print(f"Erro fatal: Colunas necessárias ausentes. Detalhe: {e}")
        return False

    # Limpeza e preenchimento de nulos com a mediana global
    # Limpeza, Isolamento e Coerção de Tipos
    nos = df_bruto.drop_duplicates(subset=['SEGMENT_ID']).copy()
    nos = nos.sort_values('SEGMENT_ID').reset_index(drop=True)
    num_nos = len(nos)
    
    # Tratamento crítico: Garantir que todas as variáveis espaciais sejam números (float)
    colunas_espaciais = ['LENGTH', 'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE']
    for col in colunas_espaciais:
        # Converte para string, troca vírgula por ponto (padrão americano), remove espaços
        nos[col] = nos[col].astype(str).str.replace(',', '.').str.strip()
        # Força a conversão para numérico; o que não for número vira NaN
        nos[col] = pd.to_numeric(nos[col], errors='coerce')
    
    # Agora o cálculo da mediana funcionará perfeitamente, pois LENGTH é float
    mediana_length = nos['LENGTH'].replace(0, np.nan).median()
    nos['LENGTH'] = nos['LENGTH'].replace([0, np.nan], mediana_length)
    
    nos[['SEGMENT_ID']].to_csv(caminho_meta, index=False)

    # Inferência de Arestas por cKDTree
    G = nx.DiGraph()
    G.add_nodes_from(nos['SEGMENT_ID'])

    start_coords = nos[['START_LATITUDE', 'START_LONGITUDE']].values
    end_coords = nos[['END_LATITUDE', 'END_LONGITUDE']].values
    segment_ids = nos['SEGMENT_ID'].values
    lengths = nos['LENGTH'].values

    tree_starts = cKDTree(start_coords)
    tolerancia_graus = 0.00015 # Snapping de GPS (~15 metros)
    vizinhos = tree_starts.query_ball_point(end_coords, r=tolerancia_graus)

    arestas = []
    for i, lista_vizinhos_j in enumerate(vizinhos):
        origem = segment_ids[i]
        peso_distancia = lengths[i] 
        for j in lista_vizinhos_j:
            destino = segment_ids[j]
            if origem != destino:
                arestas.append((origem, destino, {'weight': peso_distancia}))

    G.add_edges_from(arestas)

    # Cálculo da Matriz de Adjacência e Kernel Gaussiano
    adj_mx = nx.to_numpy_array(G, nodelist=nos['SEGMENT_ID'], weight='weight', dtype=np.float32)
    distancias_validas = adj_mx[adj_mx > 0]
    
    # Sigma dinâmico baseado no desvio padrão real da amostra
    sigma = np.std(distancias_validas) if len(distancias_validas) > 0 else 1.0
    sigma_sq = sigma ** 2

    mask = adj_mx > 0
    adj_mx[mask] = np.exp(-(adj_mx[mask] ** 2) / sigma_sq)

    # Esparsidade e Diagonal
    epsilon = 0.1
    adj_mx[adj_mx < epsilon] = 0.0
    np.fill_diagonal(adj_mx, 1.0) 

    # Exportação
    np.save(caminho_adj, adj_mx)
    print(f"Matriz de adjacência {num_nos}x{num_nos} gerada e salva em: {caminho_adj}\n")
    return True


def auditar_matriz_adjacencia(caminho_matriz):
    print("--- FASE 2: PROTOCOLO DE AUDITORIA TOPOLÓGICA ---")
    adj = np.load(caminho_matriz)
    num_nos = adj.shape[0]
    
    # 1. Cálculo de Esparsidade Absoluta
    total_elementos = adj.size
    zeros = np.sum(adj == 0)
    esparsidade = (zeros / total_elementos) * 100
    
    # 2. Remoção de Auto-conexões para métricas reais
    adj_sem_diagonal = adj.copy()
    np.fill_diagonal(adj_sem_diagonal, 0)
    
    # 3. Análise de Grau de Saída (Out-degree)
    graus_saida = np.sum(adj_sem_diagonal > 0, axis=1)
    saida_media = np.mean(graus_saida)
    saida_mediana = np.median(graus_saida)
    saida_max = np.max(graus_saida)
    
    # 4. Análise de Grau de Entrada (In-degree)
    graus_entrada = np.sum(adj_sem_diagonal > 0, axis=0)
    entrada_media = np.mean(graus_entrada)
    
    # 5. Diagnóstico de Desconexão
    isolados_saida = np.sum(graus_saida == 0)
    isolados_entrada = np.sum(graus_entrada == 0)
    
    print(f"Total de Nós (Ruas): {num_nos}")
    print(f"Esparsidade da Matriz: {esparsidade:.2f}% (Alvo > 99%)")
    print(f"\n[Direcionalidade de Saída - 'Para onde o tráfego vai']")
    print(f"Grau Médio: {saida_media:.2f} | Mediana: {saida_mediana:.2f} | Máximo: {saida_max}")
    print(f"Nós sem saída (Dead ends): {isolados_saida} ({(isolados_saida/num_nos)*100:.2f}%)")
    
    print(f"\n[Direcionalidade de Entrada - 'De onde o tráfego vem']")
    print(f"Grau Médio: {entrada_media:.2f}")
    print(f"Nós sorvedouros (Sem entrada): {isolados_entrada} ({(isolados_entrada/num_nos)*100:.2f}%)")
    print("-" * 50)

if __name__ == "__main__":
    # Resolução dinâmica de caminhos
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    arquivo_csv = data_dir / "dataset_unificado.csv"
    arquivo_adj = data_dir / "adj_mx_chicago.npy"
    arquivo_meta = data_dir / "ordered_nodes.csv"
    
    data_dir.mkdir(parents=True, exist_ok=True)

    # Executa o pipeline de forma sequencial
    sucesso_geracao = gerar_matriz_adjacencia(arquivo_csv, arquivo_adj, arquivo_meta)
    
    if sucesso_geracao:
        auditar_matriz_adjacencia(arquivo_adj)