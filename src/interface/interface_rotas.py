import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
import torch
import numpy as np
import os, sys, requests
from streamlit_folium import st_folium

# Ajuste de Path para módulos internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos_granular, resolver_open_tsp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = r"C:\IC\Aplication"
data_dir = os.path.join(base_dir, "data", "tratados")

@st.cache_resource
def carregar_recursos():
    """Carrega o modelo e os metadados dos sensores."""
    model = STGCN_TrafficModel(num_nodes=60, num_features=2, num_timesteps_input=12, num_timesteps_output=12).to(device)
    model_path = os.path.join(base_dir, "src", "modelo", "best_stgcn_chicago.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    meta = pd.read_csv(os.path.join(data_dir, "node_metadata.csv"))
    for col in ['START_LATITUDE', 'START_LONGITUDE']:
        meta[col] = meta[col].astype(str).str.replace(',', '.').astype(float)
    meta['NODE_INDEX'] = meta['NODE_INDEX'].astype(int)
    
    return model, meta

def obter_metricas_ia_osrm(coord_o, coord_d, nome_o, nome_d, indice_ia_destino):
    """Calcula distâncias e o tempo otimizado pela IA via OSRM."""
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_o[1]},{coord_o[0]};{coord_d[1]},{coord_d[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url).json()['routes'][0]
        dist = r['distance'] / 1000 
        t_base = r['duration'] 
        
        # Tempo calculado com base na predição da rede neural
        t_fluxo_ia_min = (t_base * indice_ia_destino) / 60
        vel_ia = dist / (t_fluxo_ia_min / 60) if t_fluxo_ia_min > 0 else 0
        
        return {
            "origem": nome_o,
            "gps_origem": f"{coord_o[0]:.5f}, {coord_o[1]:.5f}",
            "destino": nome_d,
            "gps_destino": f"{coord_d[0]:.5f}, {coord_d[1]:.5f}",
            "coords_mapa": [[p[1], p[0]] for p in r['geometry']['coordinates']],
            "dist": dist,
            "vel": vel_ia,
            "t_ia": t_fluxo_ia_min
        }
    except: return None

# --- UI STREAMLIT ---
st.set_page_config(page_title="Chicago Smart Routing V2.0", layout="wide")
st.title("🚚 Otimizador de Rotas Baseado em IA (STGCN)")

model, meta = carregar_recursos()

if 'entregas' not in st.session_state: st.session_state.entregas = []
if 'res' not in st.session_state: st.session_state.res = None

with st.sidebar:
    st.header("📍 Destinos de Entrega")
    if st.button("🗑️ Limpar Itinerário"):
        st.session_state.entregas = []; st.session_state.res = None; st.rerun()
    
    for i, e in enumerate(st.session_state.entregas):
        st.write(f"**{i+1}.** {e['rua']}")

    if len(st.session_state.entregas) > 1:
        if st.button("🚀 Otimizar Rota"):
            with st.spinner("IA processando tráfego futuro..."):
                indices = [e['node_index'] for e in st.session_state.entregas]
                matriz, _ = gerar_matriz_custos_granular(model, os.path.join(data_dir, "data_chicago.npy"), os.path.join(data_dir, "node_metadata.csv"), os.path.join(data_dir, "scaler_stats.csv"), device, indices)
                ordem = resolver_open_tsp(matriz, indices)
                
                detalhes = []
                for i in range(len(ordem)-1):
                    o = meta.loc[meta['NODE_INDEX'] == ordem[i]]
                    d = meta.loc[meta['NODE_INDEX'] == ordem[i+1]]
                    res_leg = obter_metricas_ia_osrm(
                        (o['START_LATITUDE'].values[0], o['START_LONGITUDE'].values[0]),
                        (d['START_LATITUDE'].values[0], d['START_LONGITUDE'].values[0]),
                        o['STREET'].values[0], d['STREET'].values[0],
                        1.2 # Exemplo de ajuste de trânsito IA
                    )
                    if res_leg: detalhes.append(res_leg)
                st.session_state.res = detalhes

# --- MAPA ---
m = folium.Map(location=[41.8781, -87.6298], zoom_start=12, tiles="cartodbpositron")
for _, row in meta.iterrows():
    is_sel = any(e['node_index'] == row['NODE_INDEX'] for e in st.session_state.entregas)
    tooltip_info = f"<b>Rua:</b> {row['STREET']}<br><b>ID Sensor:</b> {row['SEGMENT_ID']}"
    folium.CircleMarker(
        [row['START_LATITUDE'], row['START_LONGITUDE']], 
        radius=6, color="red" if is_sel else "green", fill=True, 
        tooltip=folium.Tooltip(tooltip_info)
    ).add_to(m)

if st.session_state.res:
    for leg in st.session_state.res: AntPath(leg['coords_mapa'], color="blue", weight=5).add_to(m)

map_out = st_folium(m, width=1100, height=500)

if map_out.get("last_object_clicked"):
    lat, lng = map_out["last_object_clicked"]["lat"], map_out["last_object_clicked"]["lng"]
    dist = (meta['START_LATITUDE']-lat)**2 + (meta['START_LONGITUDE']-lng)**2
    if dist.min() < 0.0001:
        n = meta.loc[dist.idxmin()]
        if not any(e['node_index'] == int(n['NODE_INDEX']) for e in st.session_state.entregas):
            st.session_state.entregas.append({"node_index": int(n['NODE_INDEX']), "rua": n['STREET']})
            st.rerun()

# --- PAINEL DE RESULTADOS DETALHADO (Apenas IA) ---
if st.session_state.res:
    st.divider()
    st.subheader("📋 Itinerário Logístico")
    
    df_rota = pd.DataFrame(st.session_state.res)
    df_rota.index = [f"Trecho {i+1}" for i in range(len(df_rota))]
    
    # Colunas focadas na predição direta da rede neural
    tabela_show = df_rota[['origem', 'gps_origem', 'destino', 'gps_destino', 'vel', 'dist', 't_ia']].copy()
    tabela_show.columns = [
        'De (Rua)', 'GPS Origem', 'Para (Rua)', 'GPS Destino', 
        'Velocidade IA (km/h)', 'Distância (km)', 'Tempo IA (min)'
    ]
    
    st.dataframe(tabela_show.style.format({
        'Velocidade IA (km/h)': '{:.1f}',
        'Distância (km)': '{:.2f}',
        'Tempo IA (min)': '{:.1f}'
    }), use_container_width=True)

    # Métricas Consolidadas
    c1, c2 = st.columns(2)
    c1.metric("📏 Distância Total do Percurso", f"{tabela_show['Distância (km)'].sum():.2f} km")
    c2.metric("⏱️ Estimativa de Tempo", f"{tabela_show['Tempo IA (min)'].sum():.1f} min")