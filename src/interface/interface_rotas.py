import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
import torch
import numpy as np
import os, sys, requests, time
from streamlit_folium import st_folium

# Ajuste de Path para integração de módulos internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos_granular, resolver_open_tsp

# Configurações de Hardware e Diretórios
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = r"C:\IC\Aplication"
data_dir = os.path.join(base_dir, "data", "tratados")
modelo_dir = os.path.join(base_dir, "src", "modelo")

PALETA_CORES = [
    'blue', 'red', 'green', 'orange', 'purple', 
    'darkred', 'cadetblue', 'darkgreen', 'darkpurple', 'pink'
]

@st.cache_resource
def carregar_recursos():
    """ Carrega metadados, topologia e os pesos treinados da IA apenas uma vez. """
    meta = pd.read_csv(os.path.join(data_dir, "node_metadata.csv"))
    
    # Matriz de adjacencia para a rede neural
    adj_mx = np.load(os.path.join(data_dir, "adj_mx.npy"))
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)
    
    # Inicializacao sincronizada com a ultima arquitetura
    model = STGCN_TrafficModel(num_nodes=178, num_features=2, num_timesteps_input=12, num_timesteps_output=12, adj_mx=adj_tensor).to(device)
    
    caminho_pesos = os.path.join(modelo_dir, "best_stgcn_chicago.pt")
    if os.path.exists(caminho_pesos):
        model.load_state_dict(torch.load(caminho_pesos, map_location=device))
    else:
        st.warning("Pesos do modelo não encontrados. Rodando com inicialização aleatória.")
        
    model.eval()
    return model, meta

def geocodificar_endereco(endereco):
    """ Busca coordenadas protegendo a requisicao com User-Agent exigido pelo Nominatim. """
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': endereco + ", Chicago, IL", 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'RouteStation_IC_Academic_Project/1.0 (contact@academico.com)'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200 and len(response.json()) > 0:
            res = response.json()[0]
            return float(res['lat']), float(res['lon'])
    except Exception as e:
        st.error(f"Erro na API de Geocodificação: {e}")
    return None, None

def encontrar_no_mais_proximo(lat_usuario, lon_usuario, meta_df):
    """ Ancora a coordenada do usuario ao sensor STGCN mais proximo. """
    lat1 = np.radians(lat_usuario)
    lon1 = np.radians(lon_usuario)
    lat2 = np.radians(meta_df['START_LATITUDE'].values)
    lon2 = np.radians(meta_df['START_LONGITUDE'].values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distancias_km = 6371.0 * c 
    
    return int(np.argmin(distancias_km))

st.set_page_config(page_title="RouteStation - Otimizador IA", layout="wide")
st.title("🚦 RouteStation - STGCN Otimizador")

model, meta = carregar_recursos()

if 'pontos' not in st.session_state:
    st.session_state.pontos = []
if 'itinerario' not in st.session_state:
    st.session_state.itinerario = None
if 'rotas_geo' not in st.session_state:
    st.session_state.rotas_geo = None

with st.sidebar:
    st.header("Gestão de Destinos")
    endereco = st.text_input("Digite um endereço ou ponto de referência:")
    
    if st.button("Adicionar Parada", use_container_width=True):
        if endereco:
            lat, lon = geocodificar_endereco(endereco)
            if lat and lon:
                no_ancora = encontrar_no_mais_proximo(lat, lon, meta)
                rua_ancora = meta.iloc[no_ancora]['STREET']
                st.session_state.pontos.append({
                    'endereco_raw': endereco,
                    'lat': lat, 'lon': lon,
                    'idx': no_ancora,
                    'rua': rua_ancora
                })
                st.success(f"Ancorado no sensor: {rua_ancora}")
            else:
                st.error("Endereço não encontrado.")
    
    if st.session_state.pontos:
        st.write("---")
        st.write("**Paradas Registradas:**")
        for i, p in enumerate(st.session_state.pontos):
            st.markdown(f"{i+1}. {p['endereco_raw']} *(Sensor: {p['rua']})*")
            
        if st.button("Gerar Rota Otimizada com IA", type="primary", use_container_width=True):
            with st.spinner("IA calculando propagação de tráfego..."):
                indices_entrega = [p['idx'] for p in st.session_state.pontos]
                
                # Chamada do Otimizador Refatorado
                matriz_custos, rotas_detalhadas, G = gerar_matriz_custos_granular(
                    model,
                    os.path.join(data_dir, "data_chicago.npy"),
                    os.path.join(data_dir, "scaler_stats.csv"),
                    os.path.join(data_dir, "adj_mx.npy"),
                    device,
                    indices_entrega
                )
                
                rota_ordem = resolver_open_tsp(matriz_custos, indices_entrega)
                
                if rota_ordem:
                    itinerario = []
                    rotas_geo_detalhadas = []
                    
                    for i in range(len(rota_ordem) - 1):
                        origem_idx = rota_ordem[i]
                        destino_idx = rota_ordem[i+1]
                        
                        # Resgata o caminho de nos intermediarios do Dijkstra
                        caminho_nos = rotas_detalhadas.get((origem_idx, destino_idx), [])
                        
                        # Converte nos para Coordenadas Geograficas Reais para o Mapa
                        caminho_coords = [
                            [meta.iloc[n]['START_LATITUDE'], meta.iloc[n]['START_LONGITUDE']] 
                            for n in caminho_nos
                        ]
                        rotas_geo_detalhadas.append(caminho_coords)
                        
                        nome_origem = meta.iloc[origem_idx]['STREET']
                        nome_destino = meta.iloc[destino_idx]['STREET']
                        tempo_trecho = nx.shortest_path_length(G, origem_idx, destino_idx, weight='weight')
                        dist_trecho = nx.shortest_path_length(G, origem_idx, destino_idx, weight='distance_km')
                        
                        itinerario.append({
                            'origem': nome_origem, 'destino': nome_destino,
                            'tempo': tempo_trecho, 'distancia': dist_trecho
                        })
                        
                    st.session_state.itinerario = itinerario
                    st.session_state.rotas_geo = rotas_geo_detalhadas
                else:
                    st.error("Não foi possível encontrar uma rota viável.")
                    
        if st.button("Limpar Rota", use_container_width=True):
            st.session_state.pontos = []
            st.session_state.itinerario = None
            st.session_state.rotas_geo = None
            st.rerun()

# --- CONTEÚDO PRINCIPAL ---
c_map, c_met = st.columns([2, 1])

with c_map:
    # Foco inicial no centro de Chicago
    m = folium.Map(location=[41.8818, -87.6231], zoom_start=14, tiles="cartodbpositron")
    
    # Plota a topologia real da rota desviando do transito
    if st.session_state.rotas_geo:
        for i, caminho_coords in enumerate(st.session_state.rotas_geo):
            cor = PALETA_CORES[i % len(PALETA_CORES)]
            AntPath(
                locations=caminho_coords,
                color=cor, weight=5, opacity=0.8,
                tooltip=f"Trecho {i+1}"
            ).add_to(m)

    # Plota as paradas ancoradas
    for i, p in enumerate(st.session_state.pontos):
        folium.Marker(
            location=[p['lat'], p['lon']],
            popup=f"Parada {i+1}: {p['endereco_raw']}",
            icon=folium.Icon(color="black", icon="info-sign")
        ).add_to(m)

    st_folium(m, width=800, height=500, returned_objects=[])

with c_met:
    if st.session_state.itinerario:
        df_m = pd.DataFrame(st.session_state.itinerario)
        st.subheader("Painel da Rota Otimizada")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Tempo Total Previsto", f"{df_m['tempo'].sum():.1f} min")
        col_m2.metric("Distância Total", f"{df_m['distancia'].sum():.2f} km")
        st.divider()
        
        st.write("**Navegação Passo a Passo**")
        for i, r in df_m.iterrows():
            st.info(f"**Trecho {i+1}:** {r['origem']} ➔ {r['destino']}\n\nTempo: {r['tempo']:.1f} min | Dist: {r['distancia']:.2f} km")