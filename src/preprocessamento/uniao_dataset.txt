import pandas as pd
import numpy as np
from pathlib import Path

def unir_datasets_modulares():
    # 1. Configurações Dinâmicas de Caminho com Pathlib
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "brutos"
    tratados_dir = base_dir / "data" / "tratados"
    
    # Garante a criação da pasta de saída
    tratados_dir.mkdir(parents=True, exist_ok=True)

    # Dicionário mapeando as pastas de entrada para os arquivos de saída desejados
    splits_fisicos = {
        "treinamento": raw_dir / "data_treinamento",
        "validacao": raw_dir / "data_validacao",
        "teste": raw_dir / "data_teste"
    }

    # 2. Dicionário de Sinônimos Expandido (Garantia Topológica)
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

    # 3. Processamento Modular por Split
    for nome_split, pasta_split in splits_fisicos.items():
        print(f"\n[{nome_split.upper()}] Iniciando processamento em: {pasta_split}")
        
        if not pasta_split.exists():
            print(f"Alerta: A pasta {pasta_split} não existe. Pulando...")
            continue

        arquivos = list(pasta_split.rglob("*.csv"))
        if not arquivos:
            print(f"Aviso: Nenhum arquivo CSV encontrado em {pasta_split}. Pulando...")
            continue

        print(f"[{nome_split.upper()}] Total de arquivos encontrados: {len(arquivos)}")
        lista_dfs = []

        for f in arquivos:
            nome_f = f.name
            try:
                # Leitura do cabeçalho
                with open(f, 'r', encoding='utf-8-sig') as file:
                    header_line = file.readline()
                    cols_limpas = [c.replace('"', '').strip().upper() for c in header_line.split(',')]
                
                # Mapeamento dinâmico prévio
                mapeamento_final = {}
                for alvo, sinonimos in mapeamento_colunas.items():
                    for s in sinonimos:
                        if s.upper() in cols_limpas:
                            mapeamento_final[s.upper()] = alvo
                            break

                if len(mapeamento_final) != 9:
                    faltantes = set(mapeamento_colunas.keys()) - set(mapeamento_final.values())
                    print(f"  -> Alerta: Arquivo ignorado {nome_f}. Faltam colunas: {faltantes}")
                    continue

                # ==============================================================
                # IMPLEMENTAÇÃO DE CHUNKING (Prevenção de Out-of-Memory)
                # ==============================================================
                chunk_tamanho = 250000  # Lê 250 mil linhas por vez
                chunk_iter = pd.read_csv(f, skiprows=1, names=cols_limpas, sep=',', 
                                         engine='c', low_memory=False, dtype=str, 
                                         chunksize=chunk_tamanho)
                
                chunks_processados = 0
                for df_chunk in chunk_iter:
                    # 1. Filtra colunas
                    df_chunk = df_chunk[list(mapeamento_final.keys())]
                    df_chunk = df_chunk.rename(columns=mapeamento_final)
                    
                    # 2. Sanitização de Coordenadas
                    coords = ['START_LATITUDE', 'START_LONGITUDE']
                    for col in coords:
                        df_chunk[col] = df_chunk[col].astype(str).str.replace(',', '.')
                        df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce').astype('float32')

                    # 3. Tratamento Matemático Crítico (Velocidade e Ausências)
                    df_chunk['SPEED'] = pd.to_numeric(df_chunk['SPEED'], errors='coerce')
                    df_chunk['SPEED'] = df_chunk['SPEED'].replace(-1, np.nan).astype('float32')
                    
                    df_chunk['SEGMENT_ID'] = pd.to_numeric(df_chunk['SEGMENT_ID'], errors='coerce').astype('Int32')
                    
                    # 4. Otimização de Texto
                    cols_texto = ['STREET', 'DIRECTION', 'FROM_STREET', 'TO_STREET']
                    for col in cols_texto:
                        df_chunk[col] = df_chunk[col].astype('category')
                        
                    lista_dfs.append(df_chunk)
                    chunks_processados += 1
                
                print(f"  -> Arquivo processado: {nome_f} ({chunks_processados} chunks lidos)")

            except Exception as e:
                print(f"  -> Erro crítico ao processar {nome_f}: {e}")

        if not lista_dfs:
            print(f"[{nome_split.upper()}] Erro: Nenhum dado válido extraído.")
            continue

        # 4. Concatenação e Ordenação do Split Específico
        print(f"[{nome_split.upper()}] Concatenando bloco...")
        df_split = pd.concat(lista_dfs, ignore_index=True)

        print(f"[{nome_split.upper()}] Padronizando tempo e ordenando...")
        df_split['TIME'] = pd.to_datetime(df_split['TIME'], format='mixed', errors='coerce')
        df_split = df_split.dropna(subset=['TIME'])
        df_split = df_split.sort_values(['SEGMENT_ID', 'TIME']).reset_index(drop=True)
        
        # 5. Salvamento Direto
        arquivo_saida = tratados_dir / f"{nome_split}.csv"
        df_split.to_csv(arquivo_saida, index=False)
        print(f"[{nome_split.upper()}] Concluído. Salvo em: {arquivo_saida} ({len(df_split)} linhas)\n")

if __name__ == "__main__":
    unir_datasets_modulares()