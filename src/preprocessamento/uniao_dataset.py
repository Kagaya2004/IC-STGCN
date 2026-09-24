import pandas as pd
import numpy as np
from pathlib import Path
import os

def unir_datasets_modulares():
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "brutos"
    tratados_dir = base_dir / "data" / "tratados"
    
    tratados_dir.mkdir(parents=True, exist_ok=True)

    splits_fisicos = {
        "treinamento": raw_dir / "data_treinamento",
        "validacao": raw_dir / "data_validacao",
        "teste": raw_dir / "data_teste"
    }

    mapeamento_colunas = {
        'TIME': ['TIME', '_LAST_UPDATED', 'TIMESTAMP', '_TIME', 'LAST UPDATED'],
        'SPEED': ['SPEED', '_SPEED', 'CURRENTSPEED', 'CURRENT SPEED'],
        'SEGMENT_ID': ['SEGMENT_ID', 'SEGMENTID', 'SEGMENT ID'],
        'STREET': ['STREET', 'STREET_NAME', 'STREET NAME', 'STREETNAME'],
        'DIRECTION': ['DIRECTION', 'DIR'],
        'FROM_STREET': ['FROM_STREET', 'FROM STREET'],
        'TO_STREET': ['TO_STREET', 'TO STREET'],
        'START_LATITUDE': ['START_LATITUDE', 'START_LAT', 'LATITUDE', 'START LATITUDE'],
        'START_LONGITUDE': ['START_LONGITUDE', 'START_LON', 'LONGITUDE', 'START LONGITUDE']
    }

    for nome_split, pasta_split in splits_fisicos.items():
        print(f"\n[{nome_split.upper()}] A iniciar processamento em: {pasta_split}")
        
        if not pasta_split.exists():
            continue

        arquivos = list(pasta_split.rglob("*.csv"))
        if not arquivos:
            continue

        arquivo_saida = tratados_dir / f"{nome_split}.csv"
        
        # Remove o ficheiro antigo se existir para não duplicar dados
        if arquivo_saida.exists():
            arquivo_saida.unlink()

        total_linhas = 0

        for f in arquivos:
            nome_f = f.name
            try:
                with open(f, 'r', encoding='utf-8-sig') as file:
                    header_line = file.readline()
                    cols_limpas = [c.replace('"', '').strip().upper() for c in header_line.split(',')]
                
                mapeamento_final = {}
                for alvo, sinonimos in mapeamento_colunas.items():
                    for s in sinonimos:
                        if s.upper() in cols_limpas:
                            mapeamento_final[s.upper()] = alvo
                            break

                if len(mapeamento_final) != 9:
                    continue

                chunk_tamanho = 250000 
                chunk_iter = pd.read_csv(f, skiprows=1, names=cols_limpas, sep=',', 
                                         engine='c', low_memory=False, dtype=str, 
                                         chunksize=chunk_tamanho)
                
                for df_chunk in chunk_iter:
                    df_chunk = df_chunk[list(mapeamento_final.keys())]
                    df_chunk = df_chunk.rename(columns=mapeamento_final)
                    
                    coords = ['START_LATITUDE', 'START_LONGITUDE']
                    for col in coords:
                        df_chunk[col] = df_chunk[col].astype(str).str.replace(',', '.')
                        df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce').astype('float32')

                    df_chunk['SPEED'] = pd.to_numeric(df_chunk['SPEED'], errors='coerce')
                    df_chunk['SPEED'] = df_chunk['SPEED'].replace(-1, np.nan).astype('float32')
                    df_chunk['SEGMENT_ID'] = pd.to_numeric(df_chunk['SEGMENT_ID'], errors='coerce').astype('Int32')
                    
                    # Padronização e limpeza directas do tempo
                    df_chunk['TIME'] = pd.to_datetime(df_chunk['TIME'], format='mixed', errors='coerce')
                    df_chunk = df_chunk.dropna(subset=['TIME'])
                    
                    cols_texto = ['STREET', 'DIRECTION', 'FROM_STREET', 'TO_STREET']
                    for col in cols_texto:
                        df_chunk[col] = df_chunk[col].astype('category')
                        
                    # Grava directamente no disco em modo append ('a')
                    df_chunk.to_csv(arquivo_saida, mode='a', index=False, header=not arquivo_saida.exists())
                    total_linhas += len(df_chunk)
                
                print(f"  -> Ficheiro processado: {nome_f}")

            except Exception as e:
                print(f"  -> Erro ao processar {nome_f}: {e}")

        print(f"[{nome_split.upper()}] Concluído. Salvo em: {arquivo_saida} ({total_linhas} linhas)\n")

if __name__ == "__main__":
    unir_datasets_modulares()