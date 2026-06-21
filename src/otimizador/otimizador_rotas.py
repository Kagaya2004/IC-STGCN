import torch
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import os
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Silencia mensagens de depuração em console e ativa cache em disco do OpenStreetMap
ox.settings.log_console = False
ox.settings.use_cache = True

def obter_grafo_osmnx(data_dir):
    """Carrega a malha geométrica real de Chicago salva em disco ou realiza o download."""
    filepath = os.path.join(data_dir, "chicago_map.graphml")
    if os.path.exists(filepath):
        return ox.load_graphml(filepath)
    else:
        print("Baixando malha viária real de Chicago (OSMnx). Isso pode levar 1 minuto...")
        G = ox.graph_from_point((41.8781, -87.6298), dist=4000, network_type='drive')
        ox.save_graphml(G, filepath)
        return G

def construir_grafo_hibrido(G_osm, meta_df, pred_speeds_kmh):
    """Injeta as predições espaço-temporais da STGCN nas arestas equivalentes do OpenStreetMap."""
    for u, v, key, data in G_osm.edges(keys=True, data=True):
        length_km = data.get('length', 0) / 1000.0
        data['weight'] = (length_km / 25.0) * 60.0  # Velocidade padrão em vias locais: 25 km/h
        data['distance_km'] = length_km

    lats = meta_df['START_LATITUDE'].values
    lons = meta_df['START_LONGITUDE'].values
    nearest_edges = ox.distance.nearest_edges(G_osm, X=lons, Y=lats)
    
    for i, (u, v, key) in enumerate(nearest_edges):
        speed_kmh = max(pred_speeds_kmh[i], 5.0)
        length_km = G_osm[u][v][key].get('length', 0) / 1000.0
        G_osm[u][v][key]['weight'] = (length_km / speed_kmh) * 60.0
        
    return G_osm

def gerar_matriz_custos_granular(model, data_path, stats_path, meta_path, device, delivery_indices, coordenadas_reais_usuario, hora_alvo=12, minuto_alvo=0):
    """
    Motor Híbrido V2.3: Combina predição da IA com roteamento OSMnx e aplica
    fatiamento temporal dinâmico com base no horário do sistema ou do simulador.
    """
    model.eval()
    data_dir = os.path.dirname(data_path)
    
    # 1. Processamento e desnormalização Z-Score das velocidades da IA
    data = np.load(data_path).astype(np.float32)
    stats = pd.read_csv(stats_path)
    meta_df = pd.read_csv(meta_path)
    
    # --- LÓGICA DE INDICIAMENTO TEMPORAL ---
    passos_por_hora = 6
    passos_por_dia = 144
    
    # Converte HH:MM para o passo equivalente dentro do dia (Passos de 10 min)
    step_do_dia = (hora_alvo * passos_por_hora) + (minuto_alvo // 10)
    
    # Seleciona um bloco estável no meio do dataset histórico para fatiamento seguro
    dia_base = 10 
    indice_alvo = (dia_base * passos_por_dia) + step_do_dia
    
    # Garante que temos 12 passos de histórico para trás, senão aplica fallback
    if indice_alvo < 12:
        x_input = data[-12:, :, :]
    else:
        x_input = data[indice_alvo-12 : indice_alvo, :, :]
    # --------------------------------------
    
    x_tensor = torch.tensor(x_input).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_zscore = model(x_tensor)
        
    pred_mean_zscore = pred_zscore[0].mean(dim=0).cpu().numpy()
    pred_speed_kmh = ((pred_mean_zscore * stats['std'].values[0]) + stats['mean'].values[0]) * 1.60934
    
    # 2. Geração da malha topológica enriquecida
    G_osm_base = obter_grafo_osmnx(data_dir)
    G_hibrido = construir_grafo_hibrido(G_osm_base, meta_df, pred_speed_kmh)
    
    coordenadas_reais_usuario = np.array(coordenadas_reais_usuario)
    delivery_lats = coordenadas_reais_usuario[:, 0]
    delivery_lons = coordenadas_reais_usuario[:, 1]
    
    osm_nodes = ox.distance.nearest_nodes(G_hibrido, X=delivery_lons, Y=delivery_lats)
    
    num_deliveries = len(delivery_indices)
    matriz_custos = np.zeros((num_deliveries, num_deliveries), dtype=int)
    
    rotas_coords = {}
    metricas = {}
    
    for idx_i, start_osm in enumerate(osm_nodes):
        for idx_j, end_osm in enumerate(osm_nodes):
            if start_osm == end_osm:
                matriz_custos[idx_i, idx_j] = 0
            else:
                try:
                    custo_minutos = nx.shortest_path_length(G_hibrido, source=start_osm, target=end_osm, weight='weight')
                    distancia_km = nx.shortest_path_length(G_hibrido, source=start_osm, target=end_osm, weight='distance_km')
                    caminho_osm = nx.shortest_path(G_hibrido, source=start_osm, target=end_osm, weight='weight')
                    
                    matriz_custos[idx_i, idx_j] = int(custo_minutos * 100)
                    
                    coords_via = [[G_hibrido.nodes[n]['y'], G_hibrido.nodes[n]['x']] for n in caminho_osm]
                    coord_real_estabelecimento_origem = [delivery_lats[idx_i], delivery_lons[idx_i]]
                    coord_real_estabelecimento_destino = [delivery_lats[idx_j], delivery_lons[idx_j]]
                    
                    rotas_coords[(delivery_indices[idx_i], delivery_indices[idx_j])] = {
                        'asfalto': coords_via,
                        'acesso_origem': [coord_real_estabelecimento_origem, coords_via[0]],
                        'acesso_destino': [coords_via[-1], coord_real_estabelecimento_destino]
                    }
                    metricas[(delivery_indices[idx_i], delivery_indices[idx_j])] = (custo_minutos, distancia_km)
                    
                except nx.NetworkXNoPath:
                    matriz_custos[idx_i, idx_j] = 999999
                    
    return matriz_custos, rotas_coords, metricas

def resolver_open_tsp(matriz_custo, delivery_indices):
    num_nodes = len(matriz_custo)
    if num_nodes == 0:
        return []
    matriz_com_dummy = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
    matriz_com_dummy[:num_nodes, :num_nodes] = matriz_custo
    for i in range(num_nodes):
        matriz_com_dummy[i, num_nodes] = 0
        matriz_com_dummy[num_nodes, i] = 9999999
    matriz_com_dummy[num_nodes, 0] = 0
    
    manager = pywrapcp.RoutingIndexManager(num_nodes + 1, 1, num_nodes)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(matriz_com_dummy[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        rota_ordenada_sub = []
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx < num_nodes:
                rota_ordenada_sub.append(node_idx)
            index = solution.Value(routing.NextVar(index))
        return [delivery_indices[i] for i in rota_ordenada_sub]
    return None