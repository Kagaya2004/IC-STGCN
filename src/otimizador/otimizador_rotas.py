import torch
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import os
import requests

def gerar_matriz_custos(model, data_path, meta_path, stats_path, device, delivery_indices):
    """Integra IA e OSRM para criar uma matriz de custos baseada no asfalto real."""
    model.eval()
    
    # 1. IA: Predição de Tráfego (STGCN)
    data = np.load(data_path)
    ultimo_bloco = data[-12:, :, :] 
    x_input = torch.from_numpy(ultimo_bloco).permute(1, 0, 2).unsqueeze(0).float().to(device)
    
    stats = pd.read_csv(stats_path)
    mean_speed = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_speed = stats.loc[stats['feature'] == 'speed', 'std'].values[0]

    with torch.no_grad():
        pred_norm = model(x_input)
        v_preditas = (pred_norm[0, 0, :].cpu().numpy() * std_speed) + mean_speed
        v_preditas = np.clip(v_preditas, a_min=5.0, a_max=80.0)
        
        # ÍNDICE DE TRÁFEGO IA: 
        # Se a velocidade predita for menor que a média histórica, o tempo aumenta.
        indice_trafego = mean_speed / np.mean(v_preditas)
        v_media_kmh = np.mean(v_preditas) * 1.60934

    # 2. OSRM: Busca de métricas reais (Table Service)
    meta = pd.read_csv(meta_path)
    coords = []
    for idx in delivery_indices:
        row = meta.loc[meta['NODE_INDEX'] == idx].iloc[0]
        coords.append(f"{row['START_LONGITUDE']},{row['START_LATITUDE']}")
    
    string_coords = ";".join(coords)
    url = f"http://router.project-osrm.org/table/v1/driving/{string_coords}?annotations=duration,distance"
    
    try:
        r = requests.get(url).json()
        # Matriz base do OSRM (em segundos)
        matriz_duracao_base = np.array(r['durations'])
        
        # 3. FUSÃO HÍBRIDA: Ajusta o tempo do asfalto com a previsão da IA
        # O OSRM dá o tempo "vazio", a IA aplica o atraso do trânsito futuro.
        matriz_ajustada = (matriz_duracao_base * indice_trafego).astype(int)
        
        return matriz_ajustada, v_media_kmh
    except Exception as e:
        # Fallback caso o serviço esteja offline
        return np.zeros((len(delivery_indices), len(delivery_indices))), v_media_kmh

def resolver_tsp(matriz_custo, delivery_indices, v_media):
    """Resolve a sequência ótima (Caminho Aberto) baseada na matriz híbrida."""
    num_nos = len(matriz_custo)
    # Adição de nó fictício para não obrigar o retorno ao ponto de partida
    matriz_com_dummy = np.zeros((num_nos + 1, num_nos + 1), dtype=int)
    matriz_com_dummy[:num_nos, :num_nos] = matriz_custo
    
    manager = pywrapcp.RoutingIndexManager(num_nos + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def transit_callback(from_idx, to_idx):
        return int(matriz_com_dummy[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])
    
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(transit_callback))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        rota_indices = []
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx < num_nos:
                rota_indices.append(delivery_indices[node_idx])
            index = solution.Value(routing.NextVar(index))
        return rota_indices, v_media
    return [], 0