import pandas as pd
from pathlib import Path
import os
import shutil

def gerar_metadados_e_filtrar():
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    LAT_MIN, LAT_MAX = 41.8600, 41.8900
    LON_MIN, LON_MAX = -87.6400, -87.6100
    
    print("1. Extraindo Metadados Canônicos do Treinamento...")
    arquivo_treino = data_dir / "treinamento.csv"
    
    if not arquivo_treino.exists():
        print("Erro: treinamento.csv não encontrado. Execute o uniao_dataset.py primeiro.")
        return
        
    # Leitura em streaming apenas das colunas espaciais para não sobrecarregar a RAM
    coords_lista = []
    chunk_meta = 500000
    for chunk in pd.read_csv(arquivo_treino, usecols=['SEGMENT_ID', 'START_LATITUDE', 'START_LONGITUDE'], 
                             chunksize=chunk_meta, engine='c', low_memory=False):
        coords_lista.append(chunk)
        
    df_coords = pd.concat(coords_lista, ignore_index=True)
    
    # Coordenada representativa mediana por segmento viário
    nos_unicos = df_coords.groupby('SEGMENT_ID').median().reset_index()
    
    # Recorte espacial estrito no The Loop
    mask = (
        (nos_unicos['START_LATITUDE'] >= LAT_MIN) & 
        (nos_unicos['START_LATITUDE'] <= LAT_MAX) & 
        (nos_unicos['START_LONGITUDE'] >= LON_MIN) & 
        (nos_unicos['START_LONGITUDE'] <= LON_MAX)
    )
    nos_filtrados = nos_unicos[mask].copy()
    
    caminho_meta = data_dir / "node_metadata.csv"
    nos_filtrados.sort_values(by='SEGMENT_ID').to_csv(caminho_meta, index=False)
    print(f"-> node_metadata.csv salvo com sucesso! {len(nos_filtrados)} segmentos retidos no The Loop.\n")
    
    # 2. Filtragem Out-of-Core dos CSVs com gravação em streaming
    segmentos_validos = set(nos_filtrados['SEGMENT_ID'])
    splits = ["treinamento", "validacao", "teste"]
    chunk_tamanho = 250000
    
    for split in splits:
        arquivo_csv = data_dir / f"{split}.csv"
        arquivo_temp = data_dir / f"{split}_filtrado_temp.csv"
        
        if not arquivo_csv.exists():
            continue
            
        if arquivo_temp.exists():
            arquivo_temp.unlink()
            
        print(f"[{split.upper()}] Recortando dados em streaming (blocos de {chunk_tamanho:,} linhas)...")
        
        linhas_antes = 0
        linhas_depois = 0
        primeiro_chunk = True
        
        for chunk in pd.read_csv(arquivo_csv, chunksize=chunk_tamanho, engine='c', low_memory=False):
            linhas_antes += len(chunk)
            
            # Filtra apenas os segmentos confirmados do Loop
            chunk_filtrado = chunk[chunk['SEGMENT_ID'].isin(segmentos_validos)]
            linhas_depois += len(chunk_filtrado)
            
            if not chunk_filtrado.empty:
                chunk_filtrado.to_csv(arquivo_temp, mode='a', index=False, header=primeiro_chunk)
                primeiro_chunk = False
                
        retencao = (linhas_depois / linhas_antes) * 100 if linhas_antes > 0 else 0
        print(f"[{split.upper()}] Retenção: {linhas_antes:,} -> {linhas_depois:,} linhas ({retencao:.2f}%).")
        
        # Substituição atómica do ficheiro bruto pelo ficheiro recortado
        if arquivo_temp.exists():
            arquivo_csv.unlink()
            shutil.move(str(arquivo_temp), str(arquivo_csv))
            print(f"[{split.upper()}] Ficheiro original compactado e atualizado com sucesso.\n")

if __name__ == "__main__":
    gerar_metadados_e_filtrar()