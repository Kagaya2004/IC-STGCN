import torch
import numpy as np
import pandas as pd
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import os

def criar_grafo_ponderado_ia(adj_mx, pred_speeds_kmh):
    """ 
    Constrói o Grafo Direcionado.
    Reverte a formula Gaussiana para recuperar a distancia e calcula o tempo de viagem.
    """
    G = nx.DiGraph()
    num_nodes = adj_mx.shape[0]
    sigma_squared = 10.0 ** 2 # O mesmo sigma definido em matriz_adjacencia.py
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight_adj = adj_mx[i, j]
            
            # Se weight_adj > 0, existe uma conexao logica e fisica valida
            if weight_adj > 0 and i != j:
                # 1. Reversao Matematica: d = sqrt(-sigma^2 * ln(W))
                # Limita o weight a 0.9999 para evitar log(1) = 0
                w_safe = min(weight_adj, 0.9999) 
                distancia_km = np.sqrt(-sigma_squared * np.log(w_safe))
                
                # 2. Velocidade Predita (Ja desnormalizada)
                # Piso de 5 km/h para evitar engarrafamentos infinitos (divisao por zero)
                v_kmh = max(pred_speeds_kmh[j], 5.0)
                
                # 3. Custo: Tempo em Minutos
                tempo_minutos = (distancia_km / v_kmh) * 60.0
                
                G.add_edge(i, j, weight=tempo_minutos, distance_km=distancia_km)
                
    return G

def gerar_matriz_custos_granular(model, data_path, stats_path, adj_mx_path, device, delivery_indices):
    """ 
    Motor de Inferência: Prepara dados, prevê, desnormaliza e extrai caminhos mínimos.
    delivery_indices: Lista de IDs dos sensores (0 a 177) onde o caminhao deve parar.
    """
    model.eval()
    
    # 1. Carrega Tensores e Topologia
    data = np.load(data_path).astype(np.float32)
    adj_mx = np.load(adj_mx_path).astype(np.float32)
    stats = pd.read_csv(stats_path)
    mean_speed = stats['mean'].values[0]
    std_speed = stats['std'].values[0]
    
    # 2. Preparacao do Input (Fatia os ultimos 12 passos da serie temporal)
    # data shape original: [Total_Passos, Nos, Features]
    x_input = data[-12:, :, :]
    
    # Adiciona a dimensao do Batch para o PyTorch: [1, 12, 178, 2]
    x_tensor = torch.tensor(x_input).unsqueeze(0).to(device)
    
    # 3. Previsao (Inferencia)
    with torch.no_grad():
        # Retorna predicoes: [Batch, Horizonte(12), Nos(178)]
        pred_zscore = model(x_tensor)
        
    # Pega a previsao media para as proximas 2 horas (media no eixo do horizonte)
    pred_mean_zscore = pred_zscore[0].mean(dim=0).cpu().numpy()
    
    # 4. A Etapa Critica: DESNORMALIZACAO (Z-Score -> MPH -> KM/H)
    pred_speed_mph = (pred_mean_zscore * std_speed) + mean_speed
    pred_speed_kmh = pred_speed_mph * 1.60934
    
    # 5. Construcao do Grafo Dinamico
    G = criar_grafo_ponderado_ia(adj_mx, pred_speed_kmh)
    
    # 6. Geracao da Sub-Matriz de Custos M x M via Dijkstra
    num_deliveries = len(delivery_indices)
    matriz_custos = np.zeros((num_deliveries, num_deliveries), dtype=int)
    rotas_detalhadas = {} # Guarda os caminhos para desenhar no mapa
    
    for idx_i, start_node in enumerate(delivery_indices):
        for idx_j, end_node in enumerate(delivery_indices):
            if start_node == end_node:
                matriz_custos[idx_i, idx_j] = 0
            else:
                try:
                    # Dijkstra: Acha o menor tempo entre a Parada A e Parada B
                    custo_minutos = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
                    caminho_nos = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
                    
                    # Multiplica por 100 para converter para int (exigencia do OR-Tools)
                    matriz_custos[idx_i, idx_j] = int(custo_minutos * 100)
                    rotas_detalhadas[(start_node, end_node)] = caminho_nos
                except nx.NetworkXNoPath:
                    matriz_custos[idx_i, idx_j] = 999999 # Penalidade Severa para caminhos impossiveis
                    
    return matriz_custos, rotas_detalhadas, G

def resolver_open_tsp(matriz_custo, delivery_indices):
    """
    Resolve o Problema do Caixeiro Viajante Aberto (Não precisa voltar a origem).
    """
    num_nodes = len(matriz_custo)
    
    # Insercao do Dummy Node para o Open TSP (conecta-se a todos com custo 0)
    matriz_com_dummy = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
    matriz_com_dummy[:num_nodes, :num_nodes] = matriz_custo
    
    manager = pywrapcp.RoutingIndexManager(num_nodes + 1, 1, 0)
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
            node_idx = manager.IndexToNode(index)
            # Ignora o Dummy Node na resposta final
            if node_idx < num_nodes: 
                rota_otimizada_indices.append(delivery_indices[node_idx])
            index = solution.Value(routing.NextVar(index))
            
        return rota_otimizada_indices
    return None