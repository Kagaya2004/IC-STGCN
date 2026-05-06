import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
import torch
import numpy as np
import os, sys, requests
from streamlit_folium import st_folium

# Configuração de Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos, resolver_tsp

# --- FERRAMENTA DE MEDIÇÃO OSRM ---
def obter_metricas_reais_osrm(coord_origem, coord_destino):
    """Consulta o OSRM para obter distância real e tempo real no asfalto."""
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_origem[1]},{coord_origem[0]};{coord_destino[1]},{coord_destino[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url).json()
        route = r['routes'][0]
        return {
            "coords": [[p[1], p[0]] for p in route['geometry']['coordinates']],
            "dist_km": route['distance'] / 1000,
            "tempo_seg": route['duration'] # Tempo direto do servidor OSRM
        }
    except:
        return {"coords": [coord_origem, coord_destino], "dist_km": 0, "tempo_seg": 0}

# --- UI E RECURSOS ---
st.set_page_config(page_title="Chicago Hybrid Router V15", layout="wide", page_icon="🚚")

@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN_TrafficModel(num_nodes=60, num_features=2).to(device)
    pesos = os.path.join(r"C:\IC\Aplication", "src/modelo/best_stgcn_chicago.pt")
    model.load_state_dict(torch.load(pesos, map_location=device, weights_only=True))
    return model, device

@st.cache_data
def load_meta():
    df = pd.read_csv(r"C:\IC\Aplication\data\tratados\node_metadata.csv")
    for col in ['START_LATITUDE', 'START_LONGITUDE']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    return df

# Estado da Sessão
if 'entregas' not in st.session_state: st.session_state.entregas = []
if 'trechos_coloridos' not in st.session_state: st.session_state.trechos_coloridos = []
if 'resultados' not in st.session_state: st.session_state.resultados = {}

meta = load_meta()
modelo, device = load_resources()

st.title("🏙️ Arquitetura Híbrida: IA Estrategista + Tempo OSRM")
st.markdown("Passe o cursor sobre os nós para ver metadados. O tempo é puxado da rede viária real.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📋 Paradas Selecionadas")
    if st.button("🗑️ LIMPAR TUDO", use_container_width=True, type="primary"):
        st.session_state.entregas = []; st.session_state.trechos_coloridos = []; st.session_state.resultados = {}; st.rerun()
    
    st.divider()
    for i, ent in enumerate(st.session_state.entregas):
        st.write(f"**{i+1}.** {ent['rua']} (ID: {ent['node_index']})")
    
    if st.session_state.entregas and st.button("🚀 OTIMIZAR E MEDIR ROTA", use_container_width=True):
        with st.spinner("Puxando dados temporais do OSRM e IA..."):
            ids = [e['node_index'] for e in st.session_state.entregas]
            base = r"C:\IC\Aplication"
            
            # 1. IA gera a estratégia
            matriz, v_kmh = gerar_matriz_custos(modelo, 
                os.path.join(base, "data/tratados/data_chicago.npy"),
                os.path.join(base, "data/tratados/node_metadata.csv"),
                os.path.join(base, "data/tratados/scaler_stats.csv"), device, ids)
            
            ordem_ids, _ = resolver_tsp(matriz, ids, v_kmh)
            
            if ordem_ids:
                legs, d_total, t_total = [], 0, 0
                CORES = ['#3388ff', '#ff3333', '#22aa22', '#ffaa00', '#aa33ff']
                
                for i in range(len(ordem_ids) - 1):
                    o = meta.loc[meta['NODE_INDEX'] == ordem_ids[i]].iloc[0]
                    d = meta.loc[meta['NODE_INDEX'] == ordem_ids[i+1]].iloc[0]
                    
                    # 2. OSRM fornece o tempo real do percurso
                    m_osrm = obter_metricas_reais_osrm([o['START_LATITUDE'], o['START_LONGITUDE']], 
                                                     [d['START_LATITUDE'], d['START_LONGITUDE']])
                    
                    d_total += m_osrm['dist_km']
                    t_total += m_osrm['tempo_seg']
                    
                    legs.append({
                        "coords": m_osrm['coords'], "cor": CORES[i % len(CORES)],
                        "orig_id": ordem_ids[i], "orig_nome": o['STREET'],
                        "orig_coord": f"{o['START_LATITUDE']:.5f}, {o['START_LONGITUDE']:.5f}",
                        "dest_id": ordem_ids[i+1], "dest_nome": d['STREET'],
                        "dest_coord": f"{d['START_LATITUDE']:.5f}, {d['START_LONGITUDE']:.5f}",
                        "dist": m_osrm['dist_km'], "tempo": m_osrm['tempo_seg'] / 60
                    })
                
                st.session_state.trechos_coloridos = legs
                st.session_state.resultados = {
                    "tempo_fmt": f"{int(t_total//60)}min {int(t_total%60)}seg",
                    "v_kmh": v_kmh, "dist_km": d_total, "ordem_indices": ordem_ids
                }

# --- MAPA INTERATIVO ---
m = folium.Map(location=[41.8818, -87.6231], zoom_start=14, tiles='cartodbpositron')

for trecho in st.session_state.trechos_coloridos:
    AntPath(locations=trecho['coords'], color=trecho['cor'], pulse_color='white', weight=6).add_to(m)

for _, row in meta.iterrows():
    idx_no = int(row['NODE_INDEX'])
    ordem = None
    if st.session_state.resultados and idx_no in st.session_state.resultados['ordem_indices']:
        ordem = st.session_state.resultados['ordem_indices'].index(idx_no) + 1
    elif any(e['node_index'] == idx_no for e in st.session_state.entregas):
        for i, e in enumerate(st.session_state.entregas):
            if e['node_index'] == idx_no: ordem = i + 1; break

    info_h = f"<b>ID:</b> {idx_no}<br><b>Rua:</b> {row['STREET']}<br><b>Coord:</b> {row['START_LATITUDE']:.5f}, {row['START_LONGITUDE']:.5f}"

    if ordem:
        folium.Marker(
            location=[row['START_LATITUDE'], row['START_LONGITUDE']],
            icon=folium.DivIcon(html=f'<div style="color:white; background:blue; border-radius:50%; width:25px; height:28px; display:flex; align-items:center; justify-content:center; font-weight:bold; border:2px solid white;">{ordem}</div>'),
            tooltip=folium.Tooltip(info_h, sticky=True)
        ).add_to(m)
    else:
        folium.CircleMarker(
            location=[row['START_LATITUDE'], row['START_LONGITUDE']],
            radius=6, color='green', fill=True, fill_opacity=0.3,
            tooltip=folium.Tooltip(info_h, sticky=True)
        ).add_to(m)

map_out = st_folium(m, width=1200, height=500, key="mapa_v15_osrm_time")

# Lógica de Clique (Vizinho Mais Próximo)
if map_out.get("last_object_clicked"):
    lat_c, lng_c = map_out["last_object_clicked"]["lat"], map_out["last_object_clicked"]["lng"]
    dist = (meta['START_LATITUDE'] - lat_c)**2 + (meta['START_LONGITUDE'] - lng_c)**2
    idx_closest = dist.idxmin()
    if dist.min() < 1e-7:
        node_d = meta.loc[idx_closest]
        if not any(e['node_index'] == int(node_d['NODE_INDEX']) for e in st.session_state.entregas):
            st.session_state.entregas.append({"node_index": int(node_d['NODE_INDEX']), "rua": node_d['STREET']})
            st.session_state.trechos_coloridos = []; st.session_state.resultados = {}; st.rerun()

# --- RESULTADOS MÉTRICOS ---
if st.session_state.resultados:
    res = st.session_state.resultados
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Tempo de Percurso (OSRM)", res['tempo_fmt'])
    c2.metric("🚀 Velocidade Média (STGCN)", f"{res['v_kmh']:.1f} km/h")
    c3.metric("📏 Distância de Rede (OSRM)", f"{res['dist_km']:.2f} km")

    st.subheader("🚩 Itinerário Detalhado da Rede Viária")
    df_iti = pd.DataFrame([{
        "Ordem": f"{i+1}º",
        "Origem (ID)": f"{t['orig_nome']} ({t['orig_id']})",
        "Lat/Lon Origem": t['orig_coord'],
        "Destino (ID)": f"{t['dest_nome']} ({t['dest_id']})",
        "Lat/Lon Destino": t['dest_coord'],
        "Dist. (km)": f"{t['dist']:.2f}",
        "Tempo (min)": f"{t['tempo']:.1f}",
        "Cor Mapa": t['cor']
    } for i, t in enumerate(st.session_state.trechos_coloridos)])
    
    st.table(df_iti.style.map(lambda v: f'background-color: {v}; color: white; font-weight: bold;', subset=['Cor Mapa']))