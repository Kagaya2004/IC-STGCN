import pandas as pd
import glob
import os
import numpy as np

def unir_datasets():
    # 1. Configuracoes de Caminho
    base_dir = r"C:\IC\Aplication"
    raw_dir = os.path.join(base_dir, "data", "brutos")
    arquivo_saida = os.path.join(base_dir, "data", "tratados", "dataset_unificado.csv")
    
    if not os.path.exists(os.path.dirname(arquivo_saida)):
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    # 2. Dicionario de Sinonimos Expandido (Garantia Topologica)
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

    print("Iniciando localizacao de arquivos CSV em:", raw_dir)
    arquivos = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    
    if not arquivos:
        print("Erro: Nenhum arquivo encontrado.")
        return

    print(f"Total de arquivos encontrados: {len(arquivos)}")
    lista_dfs = []
    
    for f in arquivos:
        nome_f = os.path.basename(f)
        try:
            # Resgate de arquivos com cabecalhos corrompidos
            with open(f, 'r', encoding='utf-8-sig') as file:
                header_line = file.readline()
                cols_limpas = [c.replace('"', '').strip().upper() for c in header_line.split(',')]
            
            # Carrega arquivo como string temporariamente para evitar falhas de tipagem mista com as virgulas
            df_temp = pd.read_csv(f, skiprows=1, names=cols_limpas, sep=',', engine='c', low_memory=False, dtype=str)
            
            # Mapeamento dinamico
            mapeamento_final = {}
            for alvo, sinonimos in mapeamento_colunas.items():
                for s in sinonimos:
                    if s.upper() in cols_limpas:
                        mapeamento_final[s.upper()] = alvo
                        break
            
            # Valida se todas as 9 colunas essenciais foram encontradas
            if len(mapeamento_final) == 9:
                # Mantem apenas colunas necessarias e descarta ruido (HOUR, BUS_COUNT, etc.)
                df_temp = df_temp[list(mapeamento_final.keys())]
                df_temp = df_temp.rename(columns=mapeamento_final)
                lista_dfs.append(df_temp)
                print(f"Arquivo mapeado: {nome_f}")
            else:
                faltantes = set(mapeamento_colunas.keys()) - set(mapeamento_final.values())
                print(f"Alerta: Arquivo ignorado {nome_f}. Faltam colunas: {faltantes}")

        except Exception as e:
            print(f"Erro critico ao processar {nome_f}: {e}")

    if not lista_dfs:
        print("Erro fatal: Nenhum dado valido foi extraido para uniao.")
        return

    # 3. Concatenacao Massiva
    print("\nConcatenando todos os dados. Isso pode consumir muita RAM...")
    df_bruto = pd.concat(lista_dfs, ignore_index=True)

    # 4. Tratamento de Formatação e Tipagem (Crucial para nao estourar memoria)
    print("Sanitizando formato de decimais e otimizando tipos de memoria...")
    
    # Tratamento de Virgulas em coordenadas que vieram como string no dataset antigo
    coords = ['START_LATITUDE', 'START_LONGITUDE']
    for col in coords:
        df_bruto[col] = df_bruto[col].astype(str).str.replace(',', '.')
        df_bruto[col] = pd.to_numeric(df_bruto[col], errors='coerce').astype('float32')

    df_bruto['SEGMENT_ID'] = pd.to_numeric(df_bruto['SEGMENT_ID'], errors='coerce').astype('Int32')
    df_bruto['SPEED'] = pd.to_numeric(df_bruto['SPEED'], errors='coerce').astype('float32')
    
    # Textos categorizados para poupar RAM
    cols_texto = ['STREET', 'DIRECTION', 'FROM_STREET', 'TO_STREET']
    for col in cols_texto:
        df_bruto[col] = df_bruto[col].astype('category')

    # 5. Tratamento Temporal Basico e Filtragem (2023-2025)
    print("Padronizando timestamps...")
    # Tenta inferir o formato, lidando com inconsistencias
    df_bruto['TIME'] = pd.to_datetime(df_bruto['TIME'], format='mixed', errors='coerce')
    df_bruto = df_bruto.dropna(subset=['TIME'])
    
    print("Filtrando periodo solicitado...")
    mask_periodo = (df_bruto['TIME'] >= '2023-01-01') & (df_bruto['TIME'] <= '2025-12-31 23:59:59')
    df_bruto = df_bruto[mask_periodo]

    # 6. Ordenacao e Salvamento
    print("Ordenando indices espaciais e temporais...")
    df_bruto = df_bruto.sort_values(['SEGMENT_ID', 'TIME']).reset_index(drop=True)
    
    print(f"Salvando dataset bruto unificado com {len(df_bruto)} linhas...")
    df_bruto.to_csv(arquivo_saida, index=False)
    print(f"Processo concluido. Arquivo salvo em: {arquivo_saida}")

if __name__ == "__main__":
    unir_datasets()