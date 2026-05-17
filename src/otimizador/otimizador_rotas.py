import torch
import numpy as np
import pandas as pd
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import requests
import os
from datetime import datetime  # Inclusão da biblioteca de tempo real

def criar_grafo_ponderado_ia(adj_mx, pred_speeds):
    """
    Transforma a matriz de adjacência num grafo NetworkX onde o peso 
    das arestas é o tempo de viagem (Distância / Velocidade prevista).
    """
    G = nx.DiGraph()
    num_nodes = adj_mx.shape[0]
    
    # Recuperação das métricas de custo baseadas na matriz de proximidade topológica
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight_adj = adj_mx[i, j]
            if weight_adj > 0 and i != j:
                # Recupera a distância aproximada (inverso do peso de adjacência)
                distancia_aprox = 1.0 / weight_adj
                
                # Velocidade prevista pela IA para o nó de destino (convertida para km/h)
                # Definição de valor mínimo de 5km/h para blindagem contra divisões por zero
                v_ia_kmh = max(pred_speeds[j] * 1.609, 5.0)
                
                # Custo da aresta = Tempo em minutos = (Distância / Velocidade) * 60
                tempo_custo = (distancia_aprox / v_ia_kmh) * 60
                G.add_edge(i, j, weight=tempo_custo)
    return G

def gerar_matriz_custos_granular(model, data_path, stats_path, device, delivery_indices):
    """
    Calcula a matriz de custos entre entregas usando busca de caminho (Dijkstra) 
    no grafo de sensores ponderado pela IA, sincronizado com o horário atual da máquina.
    """
    model.eval()
    data = np.load(data_path)  # Array de dados históricos (T, N, Features)
    
    # --- LÓGICA DE DETECÇÃO COMPLETA EM TEMPO REAL ---
    agora = datetime.now()
    minutos_totais = agora.hour * 60 + agora.minute
    
    # O dataset do Chicago Loop é padronizado em intervalos amostrais de 10 minutos
    indice_no_dia = minutos_totais // 10
    passos_por_dia = 144  # (24 horas * 60 minutos) / 10 minutos
    
    # Identifica o offset inicial do último dia completo contido no arquivo npy
    total_passos = data.shape[0]
    offset_ultimo_dia = (total_passos // passos_por_dia - 1) * passos_por_dia
    
    # Calcula o índice exato do slot correspondente ao horário da execução
    idx_alvo = offset_ultimo_dia + indice_no_dia
    
    # Garante a janela mínima mandatória de 12 timesteps (2 horas de histórico de entrada)
    if idx_alvo < 12:
        idx_alvo = 12
        
    # Fatia a janela temporal adequada com base na hora em que o botão foi pressionado
    x_raw = data[idx_alvo-12 : idx_alvo, :, :]
    x_input = torch.from_numpy(x_raw).unsqueeze(0).float().to(device)
    
    # Carregamento de dados estatísticos do scaler
    stats = pd.read_csv(stats_path)
    mean_v, std_v = stats['mean'].values[0], stats['std'].values[0]

    adj_dir = os.path.dirname(data_path)
    adj_mx_raw = np.load(os.path.join(adj_dir, "adj_mx_chicago.npy"))
    adj_mx_tensor = torch.from_numpy(adj_mx_raw).float().to(device)

    with torch.no_grad():
        # A STGCN processa a janela temporal baseada na hora atual do usuário
        prediction = model(x_input, adj_mx_tensor)
        # Desnormalização e conversão para MPH das predições de velocidades futuras
        pred_speed_mph = (prediction[0] * std_v) + mean_v
        # Média do horizonte de predição inicial (próximos 30 minutos)
        avg_future_speed = torch.mean(pred_speed_mph[:3, :], dim=0).cpu().numpy()

    # Criação do grafo de pesos dinâmicos ajustado ao tráfego do horário simulado
    G = criar_grafo_ponderado_ia(adj_mx_raw, avg_future_speed)
    
    num_delivery = len(delivery_indices)
    matriz_custo = np.zeros((num_delivery, num_delivery))

    # Preenchimento matricial via caminho mais curto de Dijkstra entre os nós do itinerário
    for i in range(num_delivery):
        for j in range(num_delivery):
            if i == j: continue
            try:
                custo_tempo = nx.shortest_path_length(G, source=delivery_indices[i], 
                                                     target=delivery_indices[j], weight='weight')
                matriz_custo[i, j] = int(custo_tempo * 100)
            except nx.NetworkXNoPath:
                matriz_custo[i, j] = 99999  # Penalização estrita para nós desconectados no grafo

    return matriz_custo, avg_future_speed, G

def resolver_open_tsp(matriz_custo, delivery_indices):
    """Resolve a ordem das entregas otimizando um caminho aberto (TSP Aberto)."""
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