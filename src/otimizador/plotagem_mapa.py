import folium
import pandas as pd
import os
import numpy as np

def gerar_mapa_entrega(indices_rota, caminho_metadata, pasta_saida):
    """
    Desenha a rota otimizada em um mapa real de Chicago.
    """
    # 1. Carregar e Corrigir Metadados
    df_meta = pd.read_csv(caminho_metadata)
    for col in ['START_LATITUDE', 'START_LONGITUDE']:
        df_meta[col] = df_meta[col].astype(str).str.replace(',', '.').astype(float)

    # 2. Organizar Coordenadas e Nomes
    coords_rota = []
    nomes_vias = []
    
    for idx in indices_rota:
        selecao = df_meta[df_meta['NODE_INDEX'] == idx]
        if not selecao.empty:
            row = selecao.iloc[0]
            coords_rota.append((row['START_LATITUDE'], row['START_LONGITUDE']))
            nomes_vias.append(row['STREET'])

    if not coords_rota:
        print("Erro: Nenhuma coordenada encontrada.")
        return

    # 3. Criar o Mapa
    mapa_chicago = folium.Map(location=coords_rota[0], zoom_start=14, tiles='cartodbpositron')

    # 4. ADICIONAR MARCADORES (VERSÃO CORRIGIDA)
    # Usamos zip para pegar o par (lat, lon) inteiro e o nome da rua separadamente
    for i, (coord, nome_rua) in enumerate(zip(coords_rota, nomes_vias)):
        cor = 'red' if i == 0 else 'blue'
        legenda = f"Ponto {i}: {nome_rua}" if i > 0 else f"Início/Depósito: {nome_rua}"
        
        folium.Marker(
            location=coord, # Agora 'coord' é o par (lat, lon), exatamente como o Folium quer
            popup=legenda,
            tooltip=f"{i} - {nome_rua}",
            icon=folium.Icon(color=cor, icon='info-sign')
        ).add_to(mapa_chicago)

    # 5. Desenhar a Linha da Rota
    folium.PolyLine(coords_rota, weight=5, color='darkblue', opacity=0.7).add_to(mapa_chicago)

    # 6. Salvar
    os.makedirs(pasta_saida, exist_ok=True)
    nome_arquivo = os.path.join(pasta_saida, "mapa_rota_otimizada.html")
    mapa_chicago.save(nome_arquivo)
    
    print(f"\nSUCESSO: Mapa corrigido gerado em: {nome_arquivo}")

if __name__ == "__main__":
    # Teste com os seus índices reais obtidos no otimizador
    rota_real = [5, 12, 33, 45, 58, 5] 
    
    base_dir = r"C:\IC\Aplication"
    meta_path = os.path.join(base_dir, "data", "tratados", "node_metadata.csv")
    out_dir = os.path.join(base_dir, "data", "logs")
    
    gerar_mapa_entrega(rota_real, meta_path, out_dir)