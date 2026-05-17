import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
import torch
import numpy as np
import os, sys, requests, time
import networkx as nx
from datetime import datetime  # Biblioteca incluída para exibir o log de hora atual no painel
from streamlit_folium import st_folium

# Ajuste de Path para integração de módulos internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos_granular, resolver_open_tsp

# Configurações de Hardware e Diretórios
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = r"C:\IC\Aplication"
data_dir = os.path.join(base_dir, "data", "tratados")

PALETA_CORES = [
    'blue', 'red', 'green', 'orange', 'purple', 
    'darkred', 'cadetblue', 'darkgreen', 'darkpurple', 'pink'
]

@st.cache_resource
def carregar_recursos():
    meta = pd.read_csv(os.path.join(data_dir, "node_metadata.csv"))
    for col in ['START_LATITUDE', 'START_LONGITUDE']:
        meta[col] = pd.to_numeric(meta[col].astype(str).str.replace(',', '.'), errors='coerce')
    meta = meta.dropna(subset=['START_LATITUDE', 'START_LONGITUDE'])
    
    num_nos = len(meta)
    model = STGCN_TrafficModel(num_nodes=num_nos, num_features=2, num_timesteps_input=12, num_timesteps_output=12).to(device)
    
    model_path = os.path.join(base_dir, "src", "modelo", "best_stgcn_chicago.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    return model, meta

def geocodificar_endereco(numero, rua):
    query = f"{numero} {rua}, Chicago, IL"
    url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
    headers = {'User-Agent': 'IC_Traffic_Project_Bot'}
    try:
        res = requests.get(url, headers=headers).json()
        if res: return float(res[0]['lat']), float(res[0]['lon'])
    except: return None
    return None

def calcular_leg(coord_o, coord_d, info_o, info_d, vel_pred_kmh):
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_o[1]},{coord_o[0]};{coord_d[1]},{coord_d[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url).json()['routes'][0]
        dist_km = r['distance'] / 1000
        t_ia = (dist_km / vel_pred_kmh) * 60 if vel_pred_kmh > 5 else r['duration'] / 60
        return {
            "origem": info_o['STREET'], "lat_o": info_o['START_LATITUDE'], "lon_o": info_o['START_LONGITUDE'],
            "destino": info_d['STREET'], "lat_d": info_d['START_LATITUDE'], "lon_d": info_d['START_LONGITUDE'],
            "coords_geo": [[p[1], p[0]] for p in r['geometry']['coordinates']],
            "distancia": dist_km, "velocidade": vel_pred_kmh, "tempo": t_ia
        }
    except: return None

# --- UI CONFIG ---
st.set_page_config(page_title="IA Chicago V2.0", layout="wide")
st.title("Otimização de Percursos com Predição STGCN")

model, meta = carregar_recursos()

if 'pontos' not in st.session_state: st.session_state.pontos = []
if 'itinerario' not in st.session_state: st.session_state.itinerario = None
if 'trecho_focado' not in st.session_state: st.session_state.trecho_focado = None

with st.sidebar:
    st.header("Gestão de Destinos")
    
    # Exibição informativa da sincronização temporal do modelo na interface
    hora_atual_execucao = datetime.now().strftime("%H:%M")
    st.info(f"🕒 Horário da Simulação: **{hora_atual_execucao}**")
    st.caption("A STGCN extrairá as predições de fluxo correspondentes a este período.")
    
    with st.expander("Importar Lista (CSV)"):
        up_file = st.file_uploader("Ficheiro numero,rua", type="csv")
        if up_file and st.button("Carregar CSV"):
            df_up = pd.read_csv(up_file)
            p_bar = st.progress(0)
            for i, row in df_up.iterrows():
                coords = geocodificar_endereco(row['numero'], row['rua'])
                if coords:
                    dists = (meta['START_LATITUDE'] - coords[0])**2 + (meta['START_LONGITUDE'] - coords[1])**2
                    alvo = meta.loc[dists.idxmin()]
                    st.session_state.pontos.append({"idx": int(alvo['NODE_INDEX']), "rua": f"{row['numero']} {row['rua']}"})
                p_bar.progress((i+1)/len(df_up)); time.sleep(0.4)
            st.rerun()

    with st.expander("Entrada Manual", expanded=True):
        n_casa = st.text_input("Número:")
        r_nome = st.text_input("Rua:")
        if st.button("Adicionar"):
            coords = geocodificar_endereco(n_casa, r_nome)
            if coords:
                dists = (meta['START_LATITUDE'] - coords[0])**2 + (meta['START_LONGITUDE'] - coords[1])**2
                alvo = meta.loc[dists.idxmin()]
                st.session_state.pontos.append({"idx": int(alvo['NODE_INDEX']), "rua": f"{n_casa} {r_nome}"})
                st.rerun()

    st.divider()
    st.subheader("Fila de Entregas")
    for i, p in enumerate(st.session_state.pontos): st.text(f"{i+1}. {p['rua']}")
    
    if st.button("Limpar Tudo"):
        st.session_state.pontos = []; st.session_state.itinerario = None; st.session_state.trecho_focado = None; st.rerun()

    if len(st.session_state.pontos) > 1:
        if st.button("Calcular Melhor Rota IA", type="primary"):
            indices_brutos = [p['idx'] for p in st.session_state.pontos]
            indices_unicos = []
            for idx in indices_brutos:
                if not indices_unicos or idx != indices_unicos[-1]: indices_unicos.append(idx)
            
            with st.spinner("Processando grafo dinâmico..."):
                matriz, speeds, G_ia = gerar_matriz_custos_granular(model, os.path.join(data_dir, "data_chicago.npy"), 
                                                               os.path.join(data_dir, "scaler_stats.csv"), device, indices_unicos)
                ordem = resolver_open_tsp(matriz, indices_unicos)
                res_it = []
                for i in range(len(ordem)-1):
                    try:
                        caminho_ia = nx.shortest_path(G_ia, source=ordem[i], target=ordem[i+1], weight='weight')
                        for s in range(len(caminho_ia)-1):
                            o_idx, d_idx = caminho_ia[s], caminho_ia[s+1]
                            o_i, d_i = meta.loc[meta['NODE_INDEX'] == o_idx].iloc[0], meta.loc[meta['NODE_INDEX'] == d_idx].iloc[0]
                            v_kmh = speeds[d_idx] * 1.609
                            leg = calcular_leg((o_i['START_LATITUDE'], o_i['START_LONGITUDE']), (d_i['START_LATITUDE'], d_i['START_LONGITUDE']), o_i, d_i, v_kmh)
                            if leg: res_it.append(leg)
                    except: continue
                st.session_state.itinerario = res_it
                st.session_state.trecho_focado = None

# --- CONTEÚDO PRINCIPAL ---
c_map, c_met = st.columns([2, 1])

with c_map:
    m = folium.Map(location=[41.8818, -87.6231], zoom_start=14, tiles="cartodbpositron")
    for _, row in meta.iterrows():
        sel = any(p['idx'] == row['NODE_INDEX'] for p in st.session_state.pontos)
        folium.CircleMarker(
            location=[row['START_LATITUDE'], row['START_LONGITUDE']],
            radius=4, color="green" if sel else "blue", fill=True, fill_opacity=0.5,
            tooltip=f"Sensor: {row['STREET']}"
        ).add_to(m)

    if st.session_state.itinerario:
        for i, l in enumerate(st.session_state.itinerario):
            is_focado = (st.session_state.trecho_focado == i)
            opacidade = 1.0 if (is_focado or st.session_state.trecho_focado is None) else 0.15
            peso = 8 if is_focado else 5
            
            AntPath(l['coords_geo'], 
                    color=PALETA_CORES[i % len(PALETA_CORES)], 
                    weight=peso, 
                    opacity=opacidade).add_to(m)
    
    st_folium(m, width=800, height=550)

with c_met:
    if st.session_state.itinerario:
        df_m = pd.DataFrame(st.session_state.itinerario)
        st.subheader("Resumo da Rota")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Tempo Total", f"{df_m['tempo'].sum():.1f} min")
        col_m2.metric("Distância Total", f"{df_m['distancia'].sum():.2f} km")
        
        if st.session_state.trecho_focado is not None:
            if st.button("Ver Rota Completa"):
                st.session_state.trecho_focado = None; st.rerun()

        st.divider()
        st.subheader("Navegação por Trecho")
        for i, r in df_m.iterrows():
            if st.button(f"● Trecho {i+1}: {r['origem'][:12]} → {r['destino'][:12]}", key=f"b_{i}"):
                st.session_state.trecho_focado = i; st.rerun()
            if st.session_state.trecho_focado == i:
                st.info(f"**Distância:** {r['distancia']:.2f} km | **Velocidade:** {r['velocidade']:.1f} km/h")

# --- TABELA FINAL ---
if st.session_state.itinerario:
    st.divider()
    st.subheader("Detalhamento Técnico Geográfico")
    df_f = pd.DataFrame(st.session_state.itinerario)
    tab = df_f[['origem', 'lat_o', 'lon_o', 'destino', 'lat_d', 'lon_d', 'distancia', 'velocidade', 'tempo']].copy()
    tab.columns = ['Origem', 'Lat. O', 'Lon. O', 'Destino', 'Lat. D', 'Lon. D', 'Dist. (km)', 'IA (km/h)', 'Tempo (min)']
    st.table(tab.style.format({'Lat. O': '{:.4f}', 'Lon. O': '{:.4f}', 'Lat. D': '{:.4f}', 'Lon. D': '{:.4f}', 'Dist. (km)': '{:.2f}', 'IA (km/h)': '{:.1f}', 'Tempo (min)': '{:.1f}'}))