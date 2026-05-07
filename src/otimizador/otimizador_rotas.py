import torch
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import requests

def gerar_matriz_custos(model, data_path, meta_path, stats_path, device, delivery_indices):
    model.eval()
    
    # 1. IA: Predição de Fluxo para TODOS os sensores
    data = np.load(data_path)
    ultimo_bloco = data[-12:, :, :] 
    x_input = torch.from_numpy(ultimo_bloco).permute(1, 0, 2).unsqueeze(0).float().to(device)
    
    stats = pd.read_csv(stats_path)
    mean_speed_hist = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_speed_hist = stats.loc[stats['feature'] == 'speed', 'std'].values[0]

    with torch.no_grad():
        pred_norm = model(x_input)
        v_preditas = (pred_norm[0, 0, :].cpu().numpy() * std_speed_hist) + mean_speed_hist
        v_preditas = np.clip(v_preditas, 5.0, 80.0)
        
        # Agora temos um vetor de índices (um para cada nó/sensor)
        # Índice > 1 significa que aquele sensor específico detectou lentidão
        indices_por_sensor = mean_speed_hist / v_preditas
        v_media_ia_kmh = np.mean(v_preditas) * 1.60934

    # 2. OSRM: Busca de matriz base
    meta = pd.read_csv(meta_path)
    coords = [f"{meta.loc[meta['NODE_INDEX']==idx, 'START_LONGITUDE'].values[0]},{meta.loc[meta['NODE_INDEX']==idx, 'START_LATITUDE'].values[0]}" for idx in delivery_indices]
    url = f"http://router.project-osrm.org/table/v1/driving/{';'.join(coords)}?annotations=duration"
    
    try:
        r = requests.get(url).json()
        matriz_base = np.array(r['durations'])
        
        # 3. FUSÃO GRANULAR:
        # Para simplificar na matriz de distâncias entre pontos de entrega, 
        # usamos o índice do sensor de DESTINO de cada perna.
        matriz_final = np.zeros_like(matriz_base)
        for i in range(len(delivery_indices)):
            for j in range(len(delivery_indices)):
                idx_sensor_destino = delivery_indices[j]
                peso_transito_local = indices_por_sensor[idx_sensor_destino]
                matriz_final[i, j] = matriz_base[i, j] * peso_transito_local
                
        return matriz_final.astype(int), v_media_ia_kmh
    except:
        return np.zeros((len(delivery_indices), len(delivery_indices))), v_media_ia_kmh

def resolver_tsp(matriz_custo, delivery_indices, v_media_ia):
    num_nos = len(matriz_custo)
    matriz_com_dummy = np.zeros((num_nos + 1, num_nos + 1), dtype=int)
    matriz_com_dummy[:num_nos, :num_nos] = matriz_custo
    manager = pywrapcp.RoutingIndexManager(num_nos + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def transit_cb(f_idx, t_idx): return int(matriz_com_dummy[manager.IndexToNode(f_idx)][manager.IndexToNode(t_idx)])
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(transit_cb))
    search_p = pywrapcp.DefaultRoutingSearchParameters()
    search_p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_p)
    if solution:
        idx, rota_ids = routing.Start(0), []
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            if node_idx < num_nos: rota_ids.append(delivery_indices[node_idx])
            idx = solution.Value(routing.NextVar(idx))
        return rota_ids, v_media_ia
    return [], 0