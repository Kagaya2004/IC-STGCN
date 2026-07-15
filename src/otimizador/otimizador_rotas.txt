import torch
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path
from scipy.spatial import KDTree
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Silencia logs e ativa cache em disco para evitar bloqueios da API do OSM
ox.settings.log_console = False
ox.settings.use_cache = True

def obter_grafo_osmnx(data_dir):
    """ Carrega a malha geométrica real do The Loop em Chicago. """
    filepath = Path(data_dir) / "chicago_map.graphml"
    
    if filepath.exists():
        print("-> Malha viária (OSMnx) carregada do cache.")
        return ox.load_graphml(filepath)
    else:
        print("-> Baixando malha viária do Centro de Chicago (OSMnx). Isso pode levar 1 minuto...")
        # Correção OSMnx 2.0+: Tupla OBRIGATÓRIA na ordem (West, South, East, North)
        bbox_chicago = (-87.6400, 41.8600, -87.6100, 41.8900)
        G = ox.graph_from_bbox(bbox=bbox_chicago, network_type='drive')
        ox.save_graphml(G, filepath)
        return G

def construir_grafo_hibrido(G_osm, meta_df, pred_speeds_zscore):
    """
    Injeta as predições espaço-temporais da STGCN nas arestas do OpenStreetMap.
    Utiliza KDTree para mapeamento espacial ultra-rápido.
    """
    print("-> Sincronizando previsões da IA com ruas reais (KDTree)...")
    
    # 1. Preparação das Coordenadas dos Sensores (N=60)
    coords_sensores = np.column_stack((meta_df['START_LATITUDE'], meta_df['START_LONGITUDE']))
    arvore_kdtree = KDTree(coords_sensores)
    
    # 2. Desnormalização heurística (Z-Score -> km/h)
    # Como não salvamos o StandardScaler original, aplicamos uma heurística urbana.
    # Média histórica do trânsito de Chicago no centro ~ 25 km/h, Desvio ~ 10 km/h
    pred_speeds_kmh = (pred_speeds_zscore * 10.0) + 25.0
    pred_speeds_kmh = np.clip(pred_speeds_kmh, 5.0, 60.0) # Limites reais (ninguém anda a -5km/h ou 200km/h no centro)

    # 3. Injeção de Pesos nas Arestas (Ruas)
    for u, v, key, data in G_osm.edges(keys=True, data=True):
        # Acha a coordenada média da rua
        if 'geometry' in data:
            lat_rua = data['geometry'].centroid.y
            lon_rua = data['geometry'].centroid.x
        else:
            lat_rua = G_osm.nodes[u]['y']
            lon_rua = G_osm.nodes[u]['x']
            
        # Acha o sensor de trânsito (N=60) mais próximo desta rua
        _, indice_sensor = arvore_kdtree.query([lat_rua, lon_rua])
        velocidade_ia = pred_speeds_kmh[indice_sensor]
        
        # Calcula o Tempo de Viagem (Custo) = (Distância_km / Velocidade_kmh) * 3600 (para segundos)
        length_km = data.get('length', 0) / 1000.0
        tempo_segundos = (length_km / velocidade_ia) * 3600.0
        
        data['weight'] = tempo_segundos
        data['speed_kmh'] = velocidade_ia
        data['distance_km'] = length_km

    return G_osm

def gerar_matriz_custos_granular(model, data_path, meta_path, adj_tensor, device, coordenadas_reais_usuario):
    """
    Motor Híbrido Principal: Faz inferência na STGCN, funde com o OSMnx 
    e calcula a matriz de custos (segundos) entre os pontos de entrega solicitados.
    """
    print("\n--- MOTOR HÍBRIDO DE ROTEAMENTO ---")
    
    # 1. Carrega Metadados e Grafo Real
    meta_df = pd.read_csv(meta_path)
    G_osm = obter_grafo_osmnx(data_path.parent)
    
    # 2. Captura o estado atual do trânsito (Simulado pegando os últimos 12 passos do teste)
    tensor_historico = np.load(data_path).astype(np.float32)
    # Pega a última janela disponível [T-12:T] e preenche NaNs com a média (0.0)
    x_input = np.nan_to_num(tensor_historico[-12:, :, :], nan=0.0) 
    
    # Adiciona dimensão de Batch [1, 12, 60, 1] e manda para GPU
    x_tensor = torch.tensor(x_input).unsqueeze(0).to(device)

    # 3. Inferência da Inteligência Artificial (Forward Pass)
    model.eval()
    with torch.no_grad():
        print("-> IA STGCN processando futuro do trânsito...")
        y_pred = model(x_tensor, adj_tensor) # Saída: [1, 12, 60]
        
    # Agrega a previsão (Média da velocidade prevista para os próximos 120 minutos)
    previsao_agregada_zscore = y_pred[0].mean(dim=0).cpu().numpy() # Vetor de 60 posições

    # 4. Fusão: IA + Grafo Físico
    G_hibrido = construir_grafo_hibrido(G_osm, meta_df, previsao_agregada_zscore)

    # 5. Cálculo da Matriz de Custos Exata (Dijkstra) entre os Pontos de Entrega
    num_pontos = len(coordenadas_reais_usuario)
    matriz_custos = np.zeros((num_pontos, num_pontos), dtype=int)
    
    print(f"-> Calculando Dijkstra Otimizado para {num_pontos} endereços...")
    
    # Mapeia lat/lon do usuário para os nós do grafo OSMnx
    nos_entrega = []
    for lat, lon in coordenadas_reais_usuario:
        no_osm = ox.distance.nearest_nodes(G_hibrido, X=lon, Y=lat)
        nos_entrega.append(no_osm)
        
    for i in range(num_pontos):
        for j in range(num_pontos):
            if i != j:
                try:
                    # Menor caminho usando o 'weight' gerado pela IA (tempo em segundos)
                    tempo_segundos = nx.shortest_path_length(G_hibrido, nos_entrega[i], nos_entrega[j], weight='weight')
                    matriz_custos[i][j] = int(tempo_segundos)
                except nx.NetworkXNoPath:
                    matriz_custos[i][j] = 999999 # Custo proibitivo se não houver caminho
                    
    return matriz_custos, nos_entrega, G_hibrido

def resolver_open_tsp(matriz_custo):
    """
    Resolve o Problema do Caixeiro Viajante Aberto usando Google OR-Tools.
    O entregador não precisa voltar à base (Dummy Node inserido matematicamente).
    """
    num_nodes = len(matriz_custo)
    if num_nodes < 2:
        return []
        
    print("-> Google OR-Tools: Calculando Rota Global Ótima...")
    # Cria uma matriz maior para acomodar o Nó Fantasma (Dummy)
    matriz_com_dummy = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
    matriz_com_dummy[:num_nodes, :num_nodes] = matriz_custo
    
    # Custo zero para ir do final para o nó fantasma e vice-versa
    for i in range(num_nodes):
        matriz_com_dummy[i, num_nodes] = 0
        matriz_com_dummy[num_nodes, i] = 0
        
    manager = pywrapcp.RoutingIndexManager(num_nodes + 1, 1, num_nodes)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matriz_com_dummy[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        index = routing.Start(0)
        rota_otimizada_indices = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != num_nodes: # Ignora o nó fantasma na resposta
                rota_otimizada_indices.append(node_index)
            index = solution.Value(routing.NextVar(index))
            
        print(f"-> Rota encontrada! Sequência de parada: {rota_otimizada_indices}")
        return rota_otimizada_indices
    else:
        print("-> Falha: Nenhuma rota viável encontrada pelo solver.")
        return []