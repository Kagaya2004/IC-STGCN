import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
import torch
import numpy as np
import os
import sys
import requests
from streamlit_folium import st_folium

# 1. Configuração de Ambiente e Caminhos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos, resolver_tsp

# --- PALETA DE CORES ---
CORES_TRECHOS = ['#3388ff', '#ff3333', '#22aa22', '#ffaa00', '#aa33ff', '#33cccc', '#cc3388']

# --- FUNÇÕES DE APOIO ---
def formatar_tempo_logistico(minutos_decimais):
    """Converte minutos para formato amigável (ex: 2min 15seg)."""
    total_seg = int(minutos_decimais * 60)
    minutos, segundos = divmod(total_seg, 60)
    return f"{minutos}min {segundos}seg" if minutos > 0 else f"{segundos}seg"

def obter_geometria_trecho(coord_origem, coord_destino):
    """Consulta o OSRM para obter o traçado real das ruas entre dois pontos."""
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_origem[1]},{coord_origem[0]};{coord_destino[1]},{coord_destino[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url)
        geometria = r.json()['routes'][0]['geometry']['coordinates']
        return [[p[1], p[0]] for p in geometria]
    except:
        return [coord_origem, coord_destino]

# --- CARREGAMENTO DE RECURSOS ---
st.set_page_config(page_title="Chicago AI Route Optimizer", layout="wide", page_icon="🚦")

@st.cache_resource
def load_resources():
    """Carrega o modelo STGCN e os pesos treinados."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = r"C:\IC\Aplication"
    model = STGCN_TrafficModel(num_nodes=60, num_features=2).to(device)
    pesos = os.path.join(base, "src", "modelo", "best_stgcn_chicago.pt")
    model.load_state_dict(torch.load(pesos, map_location=device, weights_only=True))
    return model, device

@st.cache_data
def load_meta():
    """Carrega os metadados das vias de Chicago."""
    df = pd.read_csv(r"C:\IC\Aplication\data\tratados\node_metadata.csv")
    for col in ['START_LATITUDE', 'START_LONGITUDE', 'LENGTH']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

# --- ESTADO DA SESSÃO ---
if 'entregas' not in st.session_state: st.session_state.entregas = []
if 'trechos_coloridos' not in st.session_state: st.session_state.trechos_coloridos = []
if 'resultados' not in st.session_state: st.session_state.resultados = {}

meta = load_meta()
modelo, device = load_resources()

st.title("🏙️ Otimizador Preditivo de Rotas - Chicago")
st.markdown("Otimização baseada em **Deep Learning (STGCN)** e **Pesquisa Operacional (OR-Tools)**.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📋 Gerenciamento de Paradas")
    if st.button("🗑️ LIMPAR TUDO", use_container_width=True, type="primary"):
        st.session_state.entregas = []; st.session_state.trechos_coloridos = []; st.session_state.resultados = {}; st.rerun()
    
    st.divider()
    if st.session_state.entregas:
        for i, ent in enumerate(st.session_state.entregas):
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{i+1}.** {ent['rua']}")
            if c2.button("❌", key=f"del_{i}"):
                st.session_state.entregas.pop(i); st.session_state.trechos_coloridos = []; st.rerun()
        
        st.divider()
        if st.button("🚀 OTIMIZAR E GERAR COORDENADAS", use_container_width=True):
            with st.spinner("IA processando tráfego e calculando vias..."):
                ids = [e['node_index'] for e in st.session_state.entregas]
                base = r"C:\IC\Aplication"
                
                # 1. IA e Otimizador
                matriz, v_med_mph, _ = gerar_matriz_custos(modelo, 
                    os.path.join(base, "data/tratados/data_chicago.npy"),
                    os.path.join(base, "data/tratados/node_metadata.csv"),
                    os.path.join(base, "data/tratados/scaler_stats.csv"),
                    device, ids)
                
                ordem_ids, tempo_min, _ = resolver_tsp(matriz, ids, v_med_mph)
                
                if ordem_ids:
                    legs = []
                    for i in range(len(ordem_ids) - 1):
                        row_origem = meta[meta['NODE_INDEX']==ordem_ids[i]].iloc[0]
                        row_destino = meta[meta['NODE_INDEX']==ordem_ids[i+1]].iloc[0]
                        
                        orig_coord = [row_origem['START_LATITUDE'], row_origem['START_LONGITUDE']]
                        dest_coord = [row_destino['START_LATITUDE'], row_destino['START_LONGITUDE']]
                        
                        geometria = obter_geometria_trecho(orig_coord, dest_coord)
                        
                        legs.append({
                            "coords": geometria,
                            "cor": CORES_TRECHOS[i % len(CORES_TRECHOS)],
                            "orig_nome": row_origem['STREET'],
                            "orig_coord": f"{orig_coord[0]}, {orig_coord[1]}",
                            "dest_nome": row_destino['STREET'],
                            "dest_coord": f"{dest_coord[0]}, {dest_coord[1]}"
                        })
                    
                    st.session_state.trechos_coloridos = legs
                    v_kmh = v_med_mph * 1.60934
                    st.session_state.resultados = {
                        "tempo_fmt": formatar_tempo_logistico(tempo_min),
                        "v_kmh": v_kmh,
                        "dist_km": v_kmh * (tempo_min / 60),
                        "ordem_indices": ordem_ids
                    }
    else:
        st.info("Adicione paradas clicando nos círculos verdes do mapa.")

# --- MAPA ---
st.subheader("🗺️ Visualização Geográfica do Percurso")
m = folium.Map(location=[41.8818, -87.6231], zoom_start=14, tiles='cartodbpositron')

for trecho in st.session_state.trechos_coloridos:
    AntPath(locations=trecho['coords'], color=trecho['cor'], pulse_color='white', weight=6).add_to(m)

for _, row in meta.iterrows():
    idx_no = int(row['NODE_INDEX'])
    ordem = None
    if st.session_state.resultados:
        ids_seq = st.session_state.resultados['ordem_indices']
        if idx_no in ids_seq: ordem = ids_seq.index(idx_no) + 1
    else:
        for i, e in enumerate(st.session_state.entregas):
            if e['node_index'] == idx_no: ordem = i + 1; break

    if ordem:
        folium.Marker(
            location=[row['START_LATITUDE'], row['START_LONGITUDE']],
            icon=folium.DivIcon(html=f"""<div style="color:white; background:blue; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-weight:bold; border:2px solid white;">{ordem}</div>"""),
            tooltip=f"Ponto {ordem}: {row['STREET']}"
        ).add_to(m)
    else:
        folium.CircleMarker(location=[row['START_LATITUDE'], row['START_LONGITUDE']], radius=6, color='green', fill=True, fill_opacity=0.3).add_to(m)

map_out = st_folium(m, width=1200, height=500, key="mapa_multicolor_coords")

if map_out.get("last_object_clicked"):
    lat, lng = map_out["last_object_clicked"]["lat"], map_out["last_object_clicked"]["lng"]
    match = meta[np.isclose(meta['START_LATITUDE'], lat, atol=1e-4) & np.isclose(meta['START_LONGITUDE'], lng, atol=1e-4)]
    if not match.empty:
        idx = int(match.iloc[0]['NODE_INDEX'])
        if not any(e['node_index'] == idx for e in st.session_state.entregas):
            st.session_state.entregas.append({"node_index": idx, "rua": match.iloc[0]['STREET']})
            st.session_state.trechos_coloridos = []; st.session_state.resultados = {}; st.rerun()

# --- RESULTADOS E TABELA DETALHADA COM COORDENADAS ---
if st.session_state.resultados:
    st.divider()
    res = st.session_state.resultados
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Tempo Total Estimado", res['tempo_fmt'])
    c2.metric("🚀 Velocidade Média (Rede)", f"{res['v_kmh']:.1f} km/h")
    c3.metric("📏 Distância Total", f"{res['dist_km']:.2f} km")

    st.subheader("🚩 Itinerário Detalhado (Coordenadas de Nós)")
    itinerario = []
    for i, t in enumerate(st.session_state.trechos_coloridos):
        itinerario.append({
            "Trecho": f"{i+1}º",
            "Origem": t['orig_nome'],
            "Lat/Lon (Origem)": t['orig_coord'],
            "Destino": t['dest_nome'],
            "Lat/Lon (Destino)": t['dest_coord'],
            "Cor Mapa": t['cor']
        })
    
    df_iti = pd.DataFrame(itinerario)
    
    def color_trecho(val):
        return f'background-color: {val}; color: white; font-weight: bold;'
    
    # Exibição com o estilo corrigido para Pandas 2.1+
    st.table(df_iti.style.map(color_trecho, subset=['Cor Mapa']))