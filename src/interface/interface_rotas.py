import os
import sys
import math
import requests
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import torch
import folium
from folium.plugins import AntPath
import streamlit as st
from streamlit_folium import st_folium
from pathlib import Path

# 1. Configuração Dinâmica de Módulos
base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from src.modelo.modelo_STGCN import STGCN_TrafficModel
from src.otimizador.otimizador_rotas import gerar_matriz_custos_granular, resolver_open_tsp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = base_dir / "data" / "tratados"
modelo_dir = base_dir / "src" / "modelo"

PALETA_CORES_HEX = ['#9b5de5', '#f15bb5', '#00bbf9', '#00f5d4', '#fee440', '#ff9f1c', '#e63946', '#457b9d', '#1d3557', '#2a9d8f']

@st.cache_resource
def carregar_recursos_globais():
    meta_path = data_dir / "node_metadata.csv"
    adj_path = data_dir / "adj_mx.npy"
    data_teste_path = data_dir / "data_teste.npy"
    pesos_path = modelo_dir / "stgcn_pesos_chicago.pth"

    for p in [meta_path, adj_path, data_teste_path, pesos_path]:
        if not p.exists():
            st.error(f"⚠️ Artefato ausente: {p.name}. Verifique a esteira de dados.")
            st.stop()

    meta_df = pd.read_csv(meta_path)
    num_nodes = len(meta_df)

    adj_mx = np.load(adj_path)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)

    model = STGCN_TrafficModel(in_channels=1, hidden_channels=64, out_channels=6, num_nodes=num_nodes).to(device)
    model.load_state_dict(torch.load(pesos_path, map_location=device, weights_only=True))
    model.eval()

    return model, meta_df, adj_tensor, meta_path, data_teste_path

def geocodificar_endereco(endereco):
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'RouteStation_IC_Project/1.0 (StudentResearch)'}
    params = {'q': f"{endereco}, Chicago, IL", 'format': 'json', 'limit': 1}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        dados = response.json()
        if dados:
            return float(dados[0]['lat']), float(dados[0]['lon']), dados[0]['display_name']
    except Exception as e:
        st.toast(f"Erro de conexão com satélite: {e}")
    return None, None, None

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def obter_caminho_detalhado_sem_cortes(G_osm, path_nodes):
    """ Extrai a curva das ruas sobre o asfalto. """
    coords_rua = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        edge_data = G_osm.get_edge_data(u, v)
        
        if edge_data is not None:
            key_min = min(edge_data.keys(), key=lambda k: edge_data[k].get('length', 1e9))
            dados_aresta = edge_data[key_min]
            
            if 'geometry' in dados_aresta:
                curvas = list(dados_aresta['geometry'].coords)
                coords_rua.extend([[lat, lon] for lon, lat in curvas[:-1]])
            else:
                coords_rua.append([G_osm.nodes[u]['y'], G_osm.nodes[u]['x']])
        else:
            coords_rua.append([G_osm.nodes[u]['y'], G_osm.nodes[u]['x']])
            
    last_node = path_nodes[-1]
    coords_rua.append([G_osm.nodes[last_node]['y'], G_osm.nodes[last_node]['x']])
    
    return coords_rua

# --- CONFIGURAÇÃO STREAMLIT ---
st.set_page_config(page_title="RouteStation - Painel IA", layout="wide", page_icon="🚦")

st.title("🚦 RouteStation - Roteamento Híbrido Ciberfísico")
st.markdown("Integração **STGCN (Inteligência Artificial Espaço-Temporal)** com **OpenStreetMap (Grafo Topológico)**.")

if 'pontos' not in st.session_state:
    st.session_state.pontos = []
if 'metricas' not in st.session_state:
    st.session_state.metricas = None

model, meta_df, adj_tensor, meta_path, data_teste_path = carregar_recursos_globais()

with st.sidebar:
    st.header("⚙️ Configurações de Cenário")
    modo_simulacao = st.radio(
        "Selecione o Modo de Tráfego:",
        ("Tempo Real", "Pico Histórico (Stress Test)"),
        help="O 'Pico Histórico' extrai a janela de maior assimetria histórica e aplica Temperature Scaling."
    )
    st.session_state.modo_trafego = modo_simulacao

    st.divider()
    st.header("📍 Itinerário de Entregas")

    endereco_input = st.text_input("Adicionar Endereço (The Loop, Chicago):", placeholder="Ex: Willis Tower")
    if st.button("➕ Adicionar Ponto"):
        if endereco_input:
            lat, lon, nome_oficial = geocodificar_endereco(endereco_input)
            if lat and lon:
                if 41.8600 <= lat <= 41.8900 and -87.6450 <= lon <= -87.6100:
                    st.session_state.pontos.append({'lat': lat, 'lon': lon, 'nome': nome_oficial.split(',')[0]})
                    st.success("Adicionado com sucesso!")
                    st.rerun()
                else:
                    st.error("Ponto fora do escopo geográfico do The Loop (Centro).")
            else:
                st.error("Endereço não localizado pela API global.")

    st.divider()
    if st.session_state.pontos:
        st.write(f"**Paradas mapeadas:** {len(st.session_state.pontos)}")
        for idx, p in enumerate(st.session_state.pontos):
            st.caption(f"{idx+1}º Parada: {p['nome']}")

        if st.button("🗑️ Limpar Lista", use_container_width=True):
            st.session_state.pontos = []
            st.session_state.metricas = None
            st.rerun()

        st.divider()
        if len(st.session_state.pontos) >= 2:
            if st.button("🚀 Otimizar Rota (IA)", type="primary", use_container_width=True):
                with st.spinner(f"Invocando STGCN sob condições de {st.session_state.modo_trafego}..."):
                    coordenadas = [(p['lat'], p['lon']) for p in st.session_state.pontos]

                    matriz_custo, nos_entrega, G_hibrido = gerar_matriz_custos_granular(
                        model, data_teste_path, meta_path, adj_tensor, device, coordenadas, st.session_state.modo_trafego
                    )

                    rota_indices = resolver_open_tsp(matriz_custo)

                    if rota_indices:
                        tempo_total_segundos = 0.0
                        distancia_total_acumulada_km = 0.0
                        trechos_detalhes = []
                        caminhos_completos_ia = []

                        for i in range(len(rota_indices) - 1):
                            idx_atual = rota_indices[i]
                            idx_prox = rota_indices[i+1]
                            orig = nos_entrega[idx_atual]
                            dest = nos_entrega[idx_prox]

                            p_orig_alfinete = st.session_state.pontos[idx_atual]
                            p_dest_alfinete = st.session_state.pontos[idx_prox]

                            path_nodes = nx.shortest_path(G_hibrido, orig, dest, weight='travel_time_sec')
                            caminhos_completos_ia.append(path_nodes)
                            
                            tempo_trecho_segundos = nx.shortest_path_length(G_hibrido, orig, dest, weight='travel_time_sec')
                            
                            # BLINDAGEM DE INTERFACE: Protege contra valores nulos não previstos antes da conversão int
                            if math.isnan(tempo_trecho_segundos) or math.isinf(tempo_trecho_segundos):
                                tempo_trecho_segundos = 999.0
                                
                            tempo_total_segundos += tempo_trecho_segundos
                            minutos_trecho = max(1, int(round(tempo_trecho_segundos / 60.0)))

                            try:
                                lat_no_orig, lon_no_orig = G_hibrido.nodes[orig]['y'], G_hibrido.nodes[orig]['x']
                                lat_no_dest, lon_no_dest = G_hibrido.nodes[dest]['y'], G_hibrido.nodes[dest]['x']

                                dist_acesso_origem = calcular_distancia_haversine(p_orig_alfinete['lat'], p_orig_alfinete['lon'], lat_no_orig, lon_no_orig)
                                dist_acesso_destino = calcular_distancia_haversine(lat_no_dest, lon_no_dest, p_dest_alfinete['lat'], p_dest_alfinete['lon'])
                                
                                distancia_malha_metros = sum(
                                    G_hibrido.get_edge_data(path_nodes[n], path_nodes[n+1])[0].get('length', 0)
                                    for n in range(len(path_nodes) - 1)
                                )

                                distancia_total_metros = distancia_malha_metros + dist_acesso_origem + dist_acesso_destino
                                distancia_km = distancia_total_metros / 1000.0
                            except Exception:
                                distancia_km = 0.0

                            distancia_total_acumulada_km += distancia_km

                            trechos_detalhes.append({
                                "Seq": f"{i+1}",
                                "Trecho (Origem ➔ Destino)": f"{st.session_state.pontos[idx_atual]['nome']} ➔ {st.session_state.pontos[idx_prox]['nome']}",
                                "Distância": f"{distancia_km:.2f} km",
                                "Tempo": f"{minutos_trecho} min"
                            })

                        tempo_total_minutos = tempo_total_segundos / 60.0
                        minutos_totais_global = max(1, int(round(tempo_total_minutos)))
                        tempo_texto_final = f"{minutos_totais_global} min" if minutos_totais_global < 60 else f"{minutos_totais_global // 60} h {minutos_totais_global % 60} min"

                        st.session_state.metricas = {
                            "sequencia_ideal": [st.session_state.pontos[i]['nome'] for i in rota_indices],
                            "rota_indices": rota_indices,
                            "caminhos_ia": caminhos_completos_ia,
                            "tempo_exibicao": tempo_texto_final,
                            "trechos_tabela": trechos_detalhes,
                            "distancia_total_km": distancia_total_acumulada_km,
                            "minutos_totais": minutos_totais_global
                        }
                        st.rerun()
                    else:
                        st.error("O Solver não localizou uma rota matematicamente viável.")

# --- MAPA INTERATIVO E MÉTRICAS ---
st.divider()
coluna_esquerda_mapa, coluna_direita_metricas = st.columns([2, 1])

with coluna_esquerda_mapa:
    st.subheader("Visualização Espacial da Rota Otimizada")
    mapa_folium = folium.Map(location=[41.8781, -87.6298], zoom_start=14, tiles="cartodbpositron")

    if st.session_state.metricas and 'rota_indices' in st.session_state.metricas:
        with st.spinner("Projetando geometria no asfalto..."):
            try:
                G_osm = ox.load_graphml(data_dir / "chicago_map.graphml")
            except Exception:
                st.error("Erro Crítico: Malha viária GraphML ausente.")
                st.stop()

            indices_otimizados = st.session_state.metricas['rota_indices']
            caminhos_ia = st.session_state.metricas.get('caminhos_ia', [])
            num_segmentos = len(indices_otimizados) - 1

            for i in range(num_segmentos):
                orig_idx, dest_idx = indices_otimizados[i], indices_otimizados[i+1]
                p_orig_exato, p_dest_exato = st.session_state.pontos[orig_idx], st.session_state.pontos[dest_idx]

                try:
                    path_nodes = caminhos_ia[i]
                    coords_rua = obter_caminho_detalhado_sem_cortes(G_osm, path_nodes)

                    espessura_linha = max(9 - (i * 2), 3)
                    cor_atual = PALETA_CORES_HEX[i % len(PALETA_CORES_HEX)]

                    AntPath(
                        locations=coords_rua, color=cor_atual, weight=espessura_linha,
                        opacity=0.9, dash_array=[12, 24], pulse_color="#ffffff", delay=550
                    ).add_to(mapa_folium)

                    folium.PolyLine([[p_orig_exato['lat'], p_orig_exato['lon']], coords_rua[0]],
                                    color="#6c757d", weight=2, dash_array="4, 6", opacity=0.8).add_to(mapa_folium)
                                    
                    folium.PolyLine([coords_rua[-1], [p_dest_exato['lat'], p_dest_exato['lon']]],
                                    color="#6c757d", weight=2, dash_array="4, 6", opacity=0.8).add_to(mapa_folium)

                except (IndexError, nx.NetworkXNoPath):
                    pass

            for ordem_visita, original_idx in enumerate(indices_otimizados):
                p = st.session_state.pontos[original_idx]
                cor_fundo = "#28a745" if ordem_visita == 0 else ("#dc3545" if ordem_visita == num_segmentos else "#007bff")
                rotulo = "🟢 BASE" if ordem_visita == 0 else ("🏁 CHEGADA" if ordem_visita == num_segmentos else f"📍 PARADA {ordem_visita}")

                html_marcador = f"""
                <div style="background-color: {cor_fundo}; color: white; border-radius: 50%; width: 28px; height: 28px;
                            display: flex; align-items: center; justify-content: center; font-weight: bold;
                            font-family: sans-serif; font-size: 13px; border: 2px solid white; box-shadow: 0px 2px 5px rgba(0,0,0,0.4);">
                    {ordem_visita if ordem_visita != num_segmentos else 'F'}
                </div>
                """
                folium.Marker(location=[p['lat'], p['lon']], popup=f"**{rotulo}**<br>{p['nome']}", tooltip=p['nome'],
                              icon=folium.DivIcon(html=html_marcador, icon_size=(28, 28), icon_anchor=(14, 14))).add_to(mapa_folium)

    else:
        for parada_idx, p in enumerate(st.session_state.pontos):
            folium.Marker(location=[p['lat'], p['lon']], popup=p['nome'],
                          icon=folium.Icon(color="black" if parada_idx == 0 else "gray", icon="play" if parada_idx == 0 else "map-marker")).add_to(mapa_folium)

    st_folium(mapa_folium, use_container_width=True, height=550, returned_objects=[])

with coluna_direita_metricas:
    st.subheader("Métricas da Rota (IA)")
    if st.session_state.metricas:
        st.success(f"**Tempo Total Estimado:** {st.session_state.metricas['tempo_exibicao']}")

        st.write("### 🔀 Sequência Global")
        for i, nome in enumerate(st.session_state.metricas['sequencia_ideal']):
            prefixo = "**🟢 Partida:**" if i == 0 else ("**🏁 Destino Final:**" if i == len(st.session_state.metricas['sequencia_ideal']) - 1 else f"**📍 Entrega {i}:**")
            st.markdown(f"{prefixo} {nome}")

        st.divider()
        st.write("### 🗺️ Interoperabilidade Google Maps")

        indices_otimizados = st.session_state.metricas['rota_indices']
        p_inicio = st.session_state.pontos[indices_otimizados[0]]
        routing_url = f"https://www.google.com/maps/dir/{p_inicio['lat']:.6f},{p_inicio['lon']:.6f}/"
        for i in range(1, len(indices_otimizados)):
            p_atual = st.session_state.pontos[indices_otimizados[i]]
            routing_url += f"{p_atual['lat']:.6f},{p_atual['lon']:.6f}/"

        st.link_button("🌐 Abrir Rota Completa no Google Maps", url=routing_url, type="primary", use_container_width=True)

    else:
        st.info("Insira os endereços na barra lateral e clique em Otimizar Rota.")

# --- PAINEL DE DESEMPENHO E AUDITORIA ---
st.divider()
coluna_tabela_esq, coluna_tabela_dir = st.columns([2, 1])

with coluna_tabela_esq:
    st.subheader("📊 Painel de Desempenho por Segmento")
    if st.session_state.metricas and 'trechos_tabela' in st.session_state.metricas:
        df_trechos = pd.DataFrame(st.session_state.metricas['trechos_tabela'])

        distancia_total = f"{st.session_state.metricas['distancia_total_km']:.2f} km"
        tempo_total = f"{st.session_state.metricas['minutos_totais']} min"

        linha_total = pd.DataFrame([{
            "Seq": "∑",
            "Trecho (Origem ➔ Destino)": "✨ TOTAL DO PERCURSO OTIMIZADO",
            "Distância": distancia_total,
            "Tempo": tempo_total
        }])

        df_final_com_totais = pd.concat([df_trechos, linha_total], ignore_index=True)
        st.dataframe(df_final_com_totais, hide_index=True, width='stretch')
    else:
        st.caption("A tabela de desempenho será gerada após a otimização da rota.")

with coluna_tabela_dir:
    st.subheader("📋 Auditoria de Lat/Lon")
    if st.session_state.pontos:
        df_pontos = pd.DataFrame(st.session_state.pontos)

        df_pontos['Coordenadas (Ctrl+C)'] = df_pontos.apply(
            lambda row: f"{float(row['lat']):.6f},{float(row['lon']):.6f}", axis=1
        )

        df_auditoria = df_pontos[['nome', 'Coordenadas (Ctrl+C)']]
        df_auditoria.columns = ['Localidade', 'Ctrl+C ➔ Google Maps']

        st.dataframe(df_auditoria, hide_index=True, width='stretch')
    else:
        st.caption("Insira localidades para auditar as coordenadas do GPS.")