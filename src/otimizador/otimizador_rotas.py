import torch
import numpy as np
import pandas as pd
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import requests
import os

def criar_grafo_ponderado_ia(adj_mx, pred_speeds):
    """
    Transforma a matriz de adjacência num grafo NetworkX onde o peso 
    das arestas é o tempo de viagem (Distância / Velocidade prevista).
    """
    G = nx.DiGraph()
    num_nodes = adj_mx.shape[0]
    
    # Suposição baseada na Matriz de Adjacência: 
    # adj_mx[i,j] costuma ser exp(-dist^2). Vamos recuperar uma métrica de custo.
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight_adj = adj_mx[i, j]
            if weight_adj > 0 and i != j:
                # Recuperamos uma distância aproximada (inverso do peso de adjacência)
                distancia_aprox = 1.0 / weight_adj
                
                # Velocidade prevista pela IA para o nó de destino (convertida para km/h)
                # Usamos um valor mínimo de 5km/h para evitar custos infinitos
                v_ia_kmh = max(pred_speeds[j] * 1.609, 5.0)
                
                # Custo = Tempo (Minutos) = (Distância / Velocidade) * 60
                tempo_custo = (distancia_aprox / v_ia_kmh) * 60
                G.add_edge(i, j, weight=tempo_custo)
    return G

def gerar_matriz_custos_granular(model, data_path, stats_path, device, delivery_indices):
    """
    Calcula a matriz de custos entre entregas usando busca de caminho (Dijkstra) 
    no grafo de sensores ponderado pela IA.
    """
    model.eval()
    data = np.load(data_path)
    x_input = torch.from_numpy(data[-12:, :, :]).unsqueeze(0).float().to(device)
    
    stats = pd.read_csv(stats_path)
    mean_v, std_v = stats['mean'].values[0], stats['std'].values[0]

    adj_dir = os.path.dirname(data_path)
    adj_mx_raw = np.load(os.path.join(adj_dir, "adj_mx_chicago.npy"))
    adj_mx_tensor = torch.from_numpy(adj_mx_raw).float().to(device)

    with torch.no_grad():
        prediction = model(x_input, adj_mx_tensor)
        # Desnormalização e média do horizonte (próximos 30 min)
        pred_speed_mph = (prediction[0] * std_v) + mean_v
        avg_future_speed = torch.mean(pred_speed_mph[:3, :], dim=0).cpu().numpy()

    # Criar o grafo inteligente
    G = criar_grafo_ponderado_ia(adj_mx_raw, avg_future_speed)
    
    num_delivery = len(delivery_indices)
    matriz_custo = np.zeros((num_delivery, num_delivery))

    # Preenche a matriz de custos calculando o caminho mais rápido entre cada entrega
    for i in range(num_delivery):
        for j in range(num_delivery):
            if i == j: continue
            try:
                # Dijkstra para encontrar o tempo mínimo entre dois sensores de entrega
                custo_tempo = nx.shortest_path_length(G, source=delivery_indices[i], 
                                                     target=delivery_indices[j], weight='weight')
                matriz_custo[i, j] = int(custo_tempo * 100)
            except nx.NetworkXNoPath:
                matriz_custo[i, j] = 99999 # Penalidade se não houver conexão no grafo

    return matriz_custo, avg_future_speed, G

def resolver_open_tsp(matriz_custo, delivery_indices):
    """Resolve a ordem das entregas (TSP Aberto)."""
    num_nodes = len(matriz_custo)
    matriz_com_dummy = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
    matriz_com_dummy[:num_nodes, :num_nodes] = matriz_custo
    
    manager = pywrapcp.RoutingIndexManager(num_nodes + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return matriz_com_dummy[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        rota_ordenada = []
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx < num_nodes:
                rota_ordenada.append(delivery_indices[node_idx])
            index = solution.Value(routing.NextVar(index))
        return rota_ordenada
    return delivery_indices