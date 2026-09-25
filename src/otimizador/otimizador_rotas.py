import torch
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path
from scipy.spatial import KDTree
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ox.settings.log_console = False
ox.settings.use_cache = True


def obter_grafo_osmnx(data_dir):
    """
    Carrega o grafo viário de Chicago a partir de um arquivo GraphML local 
    ou baixa a malha oficial do OpenStreetMap delimitada por Bounding Box exata.
    """
    filepath = Path(data_dir) / "chicago_map.graphml"
    
    if not filepath.exists():
        print("⚠️ Malha viária não encontrada no disco.")
        print("-> Baixando malha por coordenadas estritas (The Loop, Chicago)...")
        
        # Limites geográficos matematicamente alinhados com a interface do usuário
        norte, sul = 41.8900, 41.8600
        leste, oeste = -87.6100, -87.6450
        
        try:
            # Sintaxe OSMnx padrão (v1.x)
            G = ox.graph_from_bbox(
                north=norte, south=sul, east=leste, west=oeste, 
                network_type="drive", simplify=True
            )
        except TypeError:
            # Fallback automático para a nova sintaxe do OSMnx (v2.0+)
            G = ox.graph_from_bbox(
                bbox=(oeste, sul, leste, norte), 
                network_type="drive", simplify=True
            )
            
        ox.save_graphml(G, filepath=filepath)
        return G
        
    return ox.load_graphml(filepath)


def construir_grafo_hibrido(G_osm, meta_df, pred_speeds_zscore, mean_speed, std_speed, min_mph=10.0):
    print("-> Sincronizando previsões da IA com ruas reais (KDTree)...")
    
    mapa_abreviacoes = {"dr": "drive", "st": "street", "ave": "avenue", "blvd": "boulevard", "rd": "road", "pkwy": "parkway", "pl": "place"}
    ruas_cta_normalizadas = []
    
    # BLINDAGEM DEFENSIVA: Verifica se a coluna 'STREET' existe no metadado
    if 'STREET' in meta_df.columns:
        for rua_cta in meta_df['STREET'].dropna().unique():
            tokens_cta = str(rua_cta).lower().replace("-", " ").split()
            tokens_expandidos = {mapa_abreviacoes.get(t, t) for t in tokens_cta}
            ruas_cta_normalizadas.append(tokens_expandidos)
    else:
        print("⚠️ Aviso: Coluna 'STREET' ausente em node_metadata.csv. Filtro nominal desativado (usando apenas coordenadas).")
    
    coords_ia = meta_df[['START_LATITUDE', 'START_LONGITUDE']].values
    kdtree = KDTree(coords_ia)
    
    pred_mph = np.maximum(min_mph, (pred_speeds_zscore * std_speed) + mean_speed)
    
    for u, v, k, data in G_osm.edges(keys=True, data=True):
        if 'geometry' in data:
            lat_edge = data['geometry'].centroid.y
            lon_edge = data['geometry'].centroid.x
        else:
            lat_edge = (G_osm.nodes[u]['y'] + G_osm.nodes[v]['y']) / 2.0
            lon_edge = (G_osm.nodes[u]['x'] + G_osm.nodes[v]['x']) / 2.0
            
        _, idx_ia = kdtree.query([lat_edge, lon_edge])
        idx_ia = int(np.atleast_1d(idx_ia)[0]) 
        
        try:
            if idx_ia >= len(pred_mph):
                idx_ia = len(pred_mph) - 1
            vel_prevista_mph = pred_mph[idx_ia]
        except IndexError:
            vel_prevista_mph = mean_speed
            
        vel_prevista_kmh = vel_prevista_mph * 1.60934
        
        length_meters = data.get('length', 100.0)
        if isinstance(length_meters, list):
            length_meters = float(length_meters[0])
        else:
            length_meters = float(length_meters)
            
        length_km = length_meters / 1000.0
        
        tempo_segundos = (length_km / vel_prevista_kmh) * 3600.0
        tempo_segundos *= 2.20
        
        nome_osm = data.get('name', '')
        if isinstance(nome_osm, list):
            nome_osm = " ".join(nome_osm)
            
        tokens_osm = set(str(nome_osm).lower().replace("-", " ").split())
        
        # Heurística Anti Zigue-Zague adaptativa
        via_conhecida_ia = True # Assume true por padrão para não penalizar caso não tenhamos a coluna STREET
        
        if ruas_cta_normalizadas:
            via_conhecida_ia = False
            for tokens_cta in ruas_cta_normalizadas:
                if tokens_cta.issubset(tokens_osm):
                    via_conhecida_ia = True
                    break
                    
        if not via_conhecida_ia:
            tempo_segundos *= 1.30  
            
        velocidade_efetiva_kmh = (length_km / tempo_segundos) * 3600.0 if tempo_segundos > 0 else vel_prevista_kmh
            
        data['travel_time_sec'] = float(tempo_segundos)
        data['travel_time'] = float(tempo_segundos)
        data['weight'] = float(tempo_segundos)
        data['predicted_speed_kmh'] = float(velocidade_efetiva_kmh)
        data['speed_kph'] = float(velocidade_efetiva_kmh) 

    return G_osm


def resolver_open_tsp(matriz_custo):
    num_nodes = len(matriz_custo)
    if num_nodes < 2: return list(range(num_nodes))
        
    matriz_tsp = np.pad(matriz_custo, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    num_tsp_nodes = num_nodes + 1
    
    manager = pywrapcp.RoutingIndexManager(num_tsp_nodes, 1, [0], [num_nodes])
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matriz_tsp[from_node][to_node] * 1000)
        
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    if not solution: return list(range(num_nodes))
        
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node < num_nodes: route.append(node)
        index = solution.Value(routing.NextVar(index))
    return route


def gerar_matriz_custos_granular(model, data_path, meta_path, adj_tensor, device, coordenadas_reais_usuario, modo="Tempo Real"):
    print(f"\n--- MOTOR HÍBRIDO DE ROTEAMENTO ({modo}) ---")
    data_dir = Path(data_path).parent
    G_osm = obter_grafo_osmnx(data_dir)
    meta_df = pd.read_csv(meta_path)
    
    caminho_stats = data_dir / "scaler_stats.csv"
    if caminho_stats.exists():
        stats = pd.read_csv(caminho_stats)
        mean_speed = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
        std_speed = stats.loc[stats['feature'] == 'speed', 'std'].values[0]
    else:
        print("⚠️ scaler_stats.csv não encontrado. Assumindo fallback (Média 21.65, Std 7.01).")
        mean_speed, std_speed = 21.65, 7.01
    
    data_teste = np.load(data_path)
    input_tensor = torch.tensor(data_teste[-1:], dtype=torch.float32).unsqueeze(-1).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_zscore = model(input_tensor, adj_tensor).squeeze(0).cpu().numpy()
        
    pred_zscore = np.nan_to_num(pred_zscore, nan=0.0, posinf=0.0, neginf=0.0)
    pred_zscore_avg = np.mean(pred_zscore, axis=0)
    
    limite_velocidade_mph = 10.0
    if modo == "Pico Histórico (Stress Test)":
        pred_zscore_avg = pred_zscore_avg - 1.0
        limite_velocidade_mph = 10.0
        
    if pred_zscore_avg.ndim > 1:
        pred_zscore_avg = np.mean(pred_zscore_avg, axis=-1)
        
    G_hibrido = construir_grafo_hibrido(
        G_osm, meta_df, pred_zscore_avg, 
        mean_speed=mean_speed, std_speed=std_speed, 
        min_mph=limite_velocidade_mph
    )
    
    nos_usuario = []
    for lat, lon in coordenadas_reais_usuario:
        no_proximo = ox.distance.nearest_nodes(G_hibrido, X=lon, Y=lat)
        nos_usuario.append(no_proximo)
        
    num_pontos = len(nos_usuario)
    matriz_custos = np.zeros((num_pontos, num_pontos))
    
    for i in range(num_pontos):
        for j in range(num_pontos):
            if i == j: continue
            origem, destino = nos_usuario[i], nos_usuario[j]
            try:
                length_sec = nx.shortest_path_length(G_hibrido, origem, destino, weight='travel_time_sec')
                matriz_custo_val = length_sec
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                matriz_custo_val = 999999.0
            matriz_custos[i][j] = matriz_custo_val

    return matriz_custos, nos_usuario, G_hibrido