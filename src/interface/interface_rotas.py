import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
import torch
import numpy as np
import os, sys, requests
from streamlit_folium import st_folium

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos, resolver_tsp

# Adicionamos a lógica granular na busca de cada trecho
def obter_metricas_osrm_granular(coord_o, coord_d, indice_ia_destino):
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_o[1]},{coord_o[0]};{coord_d[1]},{coord_d[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url).json()['routes'][0]
        t_base = r['duration']
        # Tempo de Fluxo considerando o trânsito LOCAL daquele sensor (IA)
        t_fluxo_ia = t_base * indice_ia_destino
        # Tempo Realista (Fluxo IA + Atraso de Sinais 2.2x)
        t_realista = t_fluxo_ia * 2.2
        return {
            "coords": [[p[1], p[0]] for p in r['geometry']['coordinates']],
            "dist_km": r['distance'] / 1000,
            "tf": t_fluxo_ia / 60,
            "tr": t_realista / 60
        }
    except: return None

st.set_page_config(page_title="Chicago Hybrid Router V20", layout="wide", page_icon="🚦")

@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN_TrafficModel(num_nodes=60, num_features=2).to(device)
    pesos = r"C:\IC\Aplication\src\modelo\best_stgcn_chicago.pt"
    model.load_state_dict(torch.load(pesos, map_location=device, weights_only=True))
    return model, device

@st.cache_data
def load_meta():
    df = pd.read_csv(r"C:\IC\Aplication\data\tratados\node_metadata.csv")
    for c in ['START_LATITUDE', 'START_LONGITUDE']: df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
    return df

if 'entregas' not in st.session_state: st.session_state.entregas = []
if 'legs' not in st.session_state: st.session_state.legs = []
if 'res' not in st.session_state: st.session_state.res = {}

meta, (modelo, device) = load_meta(), load_resources()

# Lógica extra para pegar os índices individuais da IA para a interface
def obter_indices_individuais(modelo, device):
    data = np.load(r"C:\IC\Aplication\data\tratados\data_chicago.npy")
    ultimo_bloco = data[-12:, :, :] 
    x_input = torch.from_numpy(ultimo_bloco).permute(1, 0, 2).unsqueeze(0).float().to(device)
    stats = pd.read_csv(r"C:\IC\Aplication\data\tratados\scaler_stats.csv")
    mean_v = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_v = stats.loc[stats['feature'] == 'speed', 'std'].values[0]
    with torch.no_grad():
        pred = modelo(x_input)
        v_p = (pred[0, 0, :].cpu().numpy() * std_v) + mean_v
        return mean_v / np.clip(v_p, 5, 80)

st.title("🏙️ Otimizador Híbrido V2.0: Granularidade Espacial")

with st.sidebar:
    st.header("📋 Destinos")
    if st.button("🗑️ LIMPAR TUDO"):
        st.session_state.entregas = []; st.session_state.legs = []; st.session_state.res = {}; st.rerun()
    
    for i, e in enumerate(st.session_state.entregas):
        st.write(f"**{i+1}.** {e['rua']}")
    
    if st.session_state.entregas and st.button("🚀 OTIMIZAR GRANULAR"):
        ids = [e['node_index'] for e in st.session_state.entregas]
        indices_ia = obter_indices_individuais(modelo, device)
        matriz, v_media = gerar_matriz_custos(modelo, r"C:\IC\Aplication\data\tratados\data_chicago.npy", r"C:\IC\Aplication\data\tratados\node_metadata.csv", r"C:\IC\Aplication\data\tratados\scaler_stats.csv", device, ids)
        ordem, _ = resolver_tsp(matriz, ids, v_media)
        
        if ordem:
            legs, d_tot, tf_tot, tr_tot = [], 0, 0, 0
            for i in range(len(ordem)-1):
                o, d = meta.loc[meta['NODE_INDEX']==ordem[i]].iloc[0], meta.loc[meta['NODE_INDEX']==ordem[i+1]].iloc[0]
                # Passamos o índice específico do sensor de destino deste trecho
                m = obter_metricas_osrm_granular([o['START_LATITUDE'], o['START_LONGITUDE']], 
                                               [d['START_LATITUDE'], d['START_LONGITUDE']], 
                                               indices_ia[ordem[i+1]])
                if m:
                    d_tot += m['dist_km']; tf_tot += m['tf']; tr_tot += m['tr']
                    legs.append({
                        "orig": o['STREET'], "dest": d['STREET'], "coords": m['coords'],
                        "o_c": f"{o['START_LATITUDE']:.4f}, {o['START_LONGITUDE']:.4f}",
                        "d_c": f"{d['START_LATITUDE']:.4f}, {d['START_LONGITUDE']:.4f}",
                        "km": m['dist_km'], "tf": m['tf'], "tr": m['tr']
                    })
            st.session_state.legs = legs
            st.session_state.res = {"tf": tf_tot, "tr": tr_tot, "v": v_media, "d": d_tot}

# --- MAPA ---
m = folium.Map(location=[41.8818, -87.6231], zoom_start=14, tiles='cartodbpositron')
CORES = ['#3388ff', '#ff3333', '#22aa22', '#ffaa00']
for i, l in enumerate(st.session_state.legs):
    AntPath(l['coords'], color=CORES[i % len(CORES)], weight=6).add_to(m)

for _, row in meta.iterrows():
    idx = int(row['NODE_INDEX'])
    info = f"<b>ID:</b> {idx}<br><b>Rua:</b> {row['STREET']}<br><b>Coord:</b> {row['START_LATITUDE']:.5f}, {row['START_LONGITUDE']:.5f}"
    if any(e['node_index'] == idx for e in st.session_state.entregas):
        folium.Marker([row['START_LATITUDE'], row['START_LONGITUDE']], icon=folium.DivIcon(html=f'<div style="color:white; background:blue; border-radius:50%; width:25px; height:25px; display:flex; align-items:center; justify-content:center; font-weight:bold; border:2px solid white;">{idx}</div>'), tooltip=folium.Tooltip(info, sticky=True)).add_to(m)
    else:
        folium.CircleMarker([row['START_LATITUDE'], row['START_LONGITUDE']], radius=5, color='green', fill=True, fill_opacity=0.3, tooltip=folium.Tooltip(info, sticky=True)).add_to(m)

map_out = st_folium(m, width=1200, height=500)

if map_out.get("last_object_clicked"):
    lat, lng = map_out["last_object_clicked"]["lat"], map_out["last_object_clicked"]["lng"]
    dist = (meta['START_LATITUDE']-lat)**2 + (meta['START_LONGITUDE']-lng)**2
    if dist.min() < 1e-7:
        n = meta.loc[dist.idxmin()]
        if not any(e['node_index'] == int(n['NODE_INDEX']) for e in st.session_state.entregas):
            st.session_state.entregas.append({"node_index": int(n['NODE_INDEX']), "rua": n['STREET']})
            st.rerun()

if st.session_state.res:
    r = st.session_state.res
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Distância Total", f"{r['d']:.2f} km")
    c2.metric("🚀 Velocidade Média IA", f"{r['v']:.1f} km/h")
    c3.metric("⏱️ Tempo Fluxo (Local)", f"{int(r['tf'])}min {int((r['tf']%1)*60)}seg")
    c4.metric("⚖️ Tempo Realista (Calibrado)", f"{int(r['tr'])}min {int((r['tr']%1)*60)}seg")
    st.table(pd.DataFrame([{
        "Trecho": f"{i+1}º", "De": l['orig'], "Coord. Origem": l['o_c'], "Para": l['dest'], "Coord. Destino": l['d_c'],
        "KM": f"{l['km']:.2f}", "Fluxo (min)": f"{l['tf']:.1f}", "Realista (min)": f"{l['tr']:.1f}"
    } for i, l in enumerate(st.session_state.legs)]))