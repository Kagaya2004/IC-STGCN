import torch
import numpy as np
import pandas as pd
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import os
from src.modelo.modelo_STGCN import STGCN_TrafficModel

def gerar_matriz_custos(model, data_path, meta_path, stats_path, device, delivery_indices):
    """Gera matriz de tempos previstos entre os pontos selecionados."""
    model.eval()
    
    # 1. Preparação de Dados
    data = np.load(data_path)
    ultimo_bloco = data[-12:, :, :] 
    x_input = torch.from_numpy(ultimo_bloco).permute(1, 0, 2).unsqueeze(0).float().to(device)
    
    meta = pd.read_csv(meta_path)
    for col in ['START_LATITUDE', 'START_LONGITUDE', 'LENGTH']:
        meta[col] = meta[col].astype(str).str.replace(',', '.').astype(float)
    
    stats = pd.read_csv(stats_path)
    mean_speed = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_speed = stats.loc[stats['feature'] == 'speed', 'std'].values[0]

    # 2. Predição Espaço-Temporal
    with torch.no_grad():
        pred_norm = model(x_input)
        velocidades_previstas = (pred_norm[0, 0, :].cpu().numpy() * std_speed) + mean_speed
        velocidades_previstas = np.clip(velocidades_previstas, a_min=5.0, a_max=80.0)
        v_media_mph = np.mean(velocidades_previstas)

    # 3. Construção do Grafo de Chicago
    adj_mat = np.load(os.path.join(os.path.dirname(meta_path), "adj_mat_chicago.npy"))
    G = nx.DiGraph()
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            if adj_mat[i, j] > 0:
                distancia = float(meta.loc[meta['NODE_INDEX'] == j, 'LENGTH'].values[0])
                v_rua = float(velocidades_previstas[j])
                # Tempo em segundos
                tempo_segundos = (distancia / v_rua) * 3600
                G.add_edge(i, j, weight=tempo_segundos)

    # 4. Matriz de Custos (Tempo entre Paradas)
    num_pontos = len(delivery_indices)
    matriz_tempo_escalada = np.zeros((num_pontos, num_pontos), dtype=int)
    
    for idx_a, node_a in enumerate(delivery_indices):
        for idx_b, node_b in enumerate(delivery_indices):
            if idx_a == idx_b:
                matriz_tempo_escalada[idx_a, idx_b] = 0
            else:
                try:
                    tempo = nx.shortest_path_length(G, source=node_a, target=node_b, weight='weight')
                    matriz_tempo_escalada[idx_a, idx_b] = max(int(tempo * 100), 1)
                except nx.NetworkXNoPath:
                    matriz_tempo_escalada[idx_a, idx_b] = 1000000 
    
    return matriz_tempo_escalada, v_media_mph, G

def resolver_tsp(matriz_custo, delivery_indices, v_media):
    """Resolve a sequência ótima de paradas."""
    manager = pywrapcp.RoutingIndexManager(len(matriz_custo), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def transit_callback(from_index, to_index):
        return int(matriz_custo[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    
    callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        rota_indices = []
        while not routing.IsEnd(index):
            rota_indices.append(delivery_indices[manager.IndexToNode(index)])
            index = solution.Value(routing.NextVar(index))
        rota_indices.append(delivery_indices[manager.IndexToNode(routing.Start(0))])
        
        tempo_total = (solution.ObjectiveValue() / 100) / 60
        return rota_indices, tempo_total, v_media
    return [], 0, 0