import torch
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import requests
import datetime

def gerar_matriz_custos_granular(model, data_path, meta_path, stats_path, device, delivery_indices):
    """
    Une IA e OSRM para criar uma matriz de custos baseada em predições futuras.
    """
    model.eval()
    
    # 1. Preparação da IA (Entrada Multicanal)
    data = np.load(data_path) # (T, N, 2)
    # Pegamos os últimos 12 passos de tempo para prever os próximos
    x_raw = data[-12:, :, :] 
    x_input = torch.from_numpy(x_raw).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    stats = pd.read_csv(stats_path)
    m_v = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    s_v = stats.loc[stats['feature'] == 'speed', 'std'].values[0]

    # Carregar matriz de adjacência para a inferência
    adj_mx = torch.from_numpy(np.load(data_path.replace("data_chicago.npy", "adj_mx_chicago.npy"))).float().to(device)

    with torch.no_grad():
        # Predição da rede STGCN
        pred_norm = model(x_input, adj_mx).squeeze() # (12, 60)
        # Desnormalização e clipping de segurança (5 a 80 mph)
        v_preditas = (pred_norm.cpu().numpy() * s_v) + m_v
        v_preditas = np.clip(v_preditas, 5.0, 80.0)
        
        # Cálculo do Índice de Impedância (Média do horizonte de 2h)
        # Índice > 1 significa que a rua está mais lenta que o normal
        indices_por_sensor = m_v / np.mean(v_preditas, axis=0)
        v_media_ia_kmh = np.mean(v_preditas) * 1.60934

    # 2. OSRM: Distâncias e Duração base da malha viária (Cenário Livre)
    meta = pd.read_csv(meta_path)
    # Extraímos coordenadas dos sensores selecionados
    coords = []
    for idx in delivery_indices:
        node = meta.loc[meta['NODE_INDEX'] == idx]
        coords.append(f"{node['START_LONGITUDE'].values[0]},{node['START_LATITUDE'].values[0]}")
    
    url = f"http://router.project-osrm.org/table/v1/driving/{';'.join(coords)}?annotations=duration"
    
    try:
        r = requests.get(url).json()
        matriz_base = np.array(r['durations']) # Duração em segundos (asfalto vazio)
        
        # 3. FUSÃO HÍBRIDA: Aplicação do peso da IA no custo de cada trecho
        matriz_final = np.zeros_like(matriz_base)
        for i in range(len(delivery_indices)):
            for j in range(len(delivery_indices)):
                # O custo para chegar ao destino 'j' é penalizado pelo sensor local de 'j'
                idx_sensor_destino = delivery_indices[j]
                peso_local = indices_por_sensor[idx_sensor_destino]
                matriz_final[i, j] = matriz_base[i, j] * peso_local
                
        return matriz_final.astype(int), v_media_ia_kmh
    except Exception as e:
        print(f"Erro na API OSRM: {e}")
        return np.zeros((len(delivery_indices), len(delivery_indices))), v_media_ia_kmh

def resolver_open_tsp(matriz_custo, delivery_indices):
    """
    Resolve o Open TSP (caminho aberto) usando Google OR-Tools.
    """
    num_nos = len(matriz_custo)
    
    # Para tornar o TSP "Aberto", adicionamos um Nó Dummy (nó fantasma)
    # O custo para ir de qualquer nó para o dummy é ZERO.
    matriz_com_dummy = np.zeros((num_nos + 1, num_nos + 1), dtype=int)
    matriz_com_dummy[:num_nos, :num_nos] = matriz_custo
    
    # Manager: (número de nós, número de veículos, nó inicial)
    manager = pywrapcp.RoutingIndexManager(num_nos + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matriz_com_dummy[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Parâmetros de busca
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        index = routing.Start(0)
        rota_ordenada = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index < num_nos: # Ignora o nó dummy no resultado final
                rota_ordenada.append(delivery_indices[node_index])
            index = solution.Value(routing.NextVar(index))
        return rota_ordenada
    return []