import torch
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import os
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
        bbox_chicago = (-87.6400, 41.8600, -87.6100, 41.8900)
        G = ox.graph_from_bbox(bbox=bbox_chicago, network_type='drive')
        ox.save_graphml(G, filepath)
        return G

def construir_grafo_hibrido(G_osm, meta_df, pred_speeds_zscore, modo):
    """ Injeta as predições espaço-temporais e aplica engarrafamentos físicos. """
    print("-> Sincronizando previsões da IA com ruas reais (KDTree)...")
    
    coords_sensores = np.column_stack((meta_df['START_LATITUDE'], meta_df['START_LONGITUDE']))
    arvore_kdtree = KDTree(coords_sensores)
    
    pred_speeds_kmh = (pred_speeds_zscore * 12.0) + 30.0
    pred_speeds_kmh = np.clip(pred_speeds_kmh, 5.0, 60.0)

    for u, v, key, data in G_osm.edges(keys=True, data=True):
        if 'geometry' in data:
            lat_rua = data['geometry'].centroid.y
            lon_rua = data['geometry'].centroid.x
        else:
            lat_rua = G_osm.nodes[u]['y']
            lon_rua = G_osm.nodes[u]['x']
            
        _, indice_sensor = arvore_kdtree.query([lat_rua, lon_rua])
        velocidade_ia = pred_speeds_kmh[indice_sensor]

        # --- A JOGADA DE MESTRE: GARGALO ESPACIAL (GRIDLOCK) ---
        if modo == "Pico Histórico (Stress Test)":
            # Se o botão de Stress for ativado, cria um colapso brutal no meio exato da cidade
            if 41.8760 < lat_rua < 41.8840 and -87.6320 < lon_rua < -87.6240:
                velocidade_ia = 3.0  # Trânsito travado (velocidade de caminhada)
            else:
                velocidade_ia = 45.0 # Periferia vazia para forçar o algoritmo a desviar do centro
        
        length_km = data.get('length', 0) / 1000.0
        tempo_segundos = (length_km / velocidade_ia) * 3600.0
        
        data['weight'] = tempo_segundos
        data['speed_kmh'] = velocidade_ia
        data['distance_km'] = length_km

    return G_osm

def gerar_matriz_custos_granular(model, data_path, meta_path, adj_tensor, device, coordenadas_reais_usuario, modo="Tempo Real"):
    print(f"\n--- MOTOR HÍBRIDO DE ROTEAMENTO ({modo}) ---")
    
    meta_df = pd.read_csv(meta_path)
    G_osm = obter_grafo_osmnx(data_path.parent)
    
    tensor_historico = np.load(data_path).astype(np.float32)
    tensor_limpo = np.nan_to_num(tensor_historico, nan=0.0)
    
    if modo == "Pico Histórico (Stress Test)":
        somas_por_passo = np.sum(tensor_limpo, axis=(1, 2))
        somas_janelas = np.convolve(somas_por_passo, np.ones(12, dtype=int), 'valid')
        indice_pior_transito = np.argmin(somas_janelas)
        print(f"-> [Caos detetado] Janela histórica do pior tráfego (Índice {indice_pior_transito}).")
        x_input = tensor_limpo[indice_pior_transito : indice_pior_transito + 12, :, :] 
    else:
        print("-> [Tempo Real] Carregando a janela de dados mais recente...")
        x_input = tensor_limpo[-12:, :, :]
    
    x_tensor = torch.tensor(x_input).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor, adj_tensor) 
        
    previsao_agregada_zscore = y_pred.squeeze().mean(dim=0).cpu().numpy()

    # Passamos a variável `modo` para aplicar a Barreira Geográfica corretamente
    G_hibrido = construir_grafo_hibrido(G_osm, meta_df, previsao_agregada_zscore, modo)

    num_pontos = len(coordenadas_reais_usuario)
    matriz_custos = np.zeros((num_pontos, num_pontos), dtype=int)
    
    print(f"-> Calculando Dijkstra Otimizado para {num_pontos} endereços...")
    
    nos_entrega = []
    for lat, lon in coordenadas_reais_usuario:
        no_osm = ox.distance.nearest_nodes(G_hibrido, X=lon, Y=lat)
        nos_entrega.append(no_osm)
        
    for i in range(num_pontos):
        for j in range(num_pontos):
            if i != j:
                try:
                    tempo_segundos = nx.shortest_path_length(G_hibrido, nos_entrega[i], nos_entrega[j], weight='weight')
                    matriz_custos[i][j] = int(tempo_segundos)
                except nx.NetworkXNoPath:
                    matriz_custos[i][j] = 999999 
                    
    return matriz_custos, nos_entrega, G_hibrido

def resolver_open_tsp(matriz_custo):
    num_nodes = len(matriz_custo)
    if num_nodes < 2: return []
        
    print("-> Google OR-Tools: Calculando Rota Global Ótima...")
    matriz_com_dummy = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
    matriz_com_dummy[:num_nodes, :num_nodes] = matriz_custo
    
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
            if node_index != num_nodes: 
                rota_otimizada_indices.append(node_index)
            index = solution.Value(routing.NextVar(index))
            
        print(f"-> Rota encontrada! Sequência de paradas: {rota_otimizada_indices}")
        return rota_otimizada_indices
    else:
        print("-> Falha: Nenhuma rota viável encontrada pelo solver.")
        return []