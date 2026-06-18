import pandas as pd
import glob
import os
import numpy as np
import io

def unir_datasets():
    # 1. Configuracoes de Caminho
    base_dir = r"C:\IC\Aplication"
    raw_dir = os.path.join(base_dir, "data", "brutos")
    arquivo_saida = os.path.join(base_dir, "data", "tratados", "dataset_unificado.csv")
    
    if not os.path.exists(os.path.dirname(arquivo_saida)):
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    # 2. Dicionario de Sinonimos para colunas
    mapeamento_colunas = {
        'TIME': ['TIME', '_LAST_UPDATED', 'TIMESTAMP', '_TIME', 'LAST UPDATED'],
        'SPEED': ['SPEED', '_SPEED', 'CURRENTSPEED', 'CURRENT SPEED'],
        'SEGMENT_ID': ['SEGMENT_ID', 'SEGMENTID', 'SEGMENT ID'],
        'STREET': ['STREET', 'STREET_NAME', 'STREET NAME', 'STREETNAME'],
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
            # Resgate de arquivos com cabecalhos corrompidos (Halsted/Congress)
            # Lemos a primeira linha como texto bruto para limpar aspas mal posicionadas
            with open(f, 'r', encoding='utf-8-sig') as file:
                header_line = file.readline()
                # Remove aspas e espacos, e separa por virgula
                cols_limpas = [c.replace('"', '').strip().upper() for c in header_line.split(',')]
            
            # Carrega o arquivo ignorando a linha original do cabecalho e usando a limpa
            df_temp = pd.read_csv(f, skiprows=1, names=cols_limpas, sep=',', engine='c', low_memory=False)
            
            # Mapeamento dinâmico baseado na lista limpa
            mapeamento_final = {}
            for alvo, sinonimos in mapeamento_colunas.items():
                for s in sinonimos:
                    if s.upper() in cols_limpas:
                        mapeamento_final[s.upper()] = alvo
                        break
            
            if len(mapeamento_final) == 6:
                # Mantem apenas as colunas necessarias e renomeia
                df_temp = df_temp[list(mapeamento_final.keys())]
                df_temp = df_temp.rename(columns=mapeamento_final)
                lista_dfs.append(df_temp)
                print(f"Arquivo carregado com sucesso: {nome_f}")
            else:
                faltantes = set(mapeamento_colunas.keys()) - set(mapeamento_final.values())
                print(f"Erro no arquivo {nome_f}. Faltam colunas: {faltantes}")

        except Exception as e:
            print(f"Erro critico ao processar {nome_f}: {e}")

    if not lista_dfs:
        print("Erro: Nenhum dado valido foi extraido.")
        return

    # 3. Concatenacao
    print("\nConcatenando todos os dados...")
    df_bruto = pd.concat(lista_dfs, ignore_index=True)

    # 4. Tratamento de Datas e Tipos
    print("Padronizando formatos temporais...")
    # Formato padrao do portal de Chicago
    df_bruto['TIME'] = pd.to_datetime(df_bruto['TIME'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Tratamento de falhas na data (caso mude para formato ISO)
    nans = df_bruto['TIME'].isna()
    if nans.any():
        df_bruto.loc[nans, 'TIME'] = pd.to_datetime(df_bruto.loc[nans, 'TIME'], errors='coerce')

    df_bruto = df_bruto.dropna(subset=['TIME'])

    # Otimizacao de tipos para RTX 3050 (6GB VRAM)
    df_bruto['SEGMENT_ID'] = df_bruto['SEGMENT_ID'].astype('int32')
    df_bruto['SPEED'] = df_bruto['SPEED'].astype('float32')

    # 5. Filtragem do Periodo (2023 - 2025)
    print("Filtrando periodo solicitado...")
    mask_periodo = (df_bruto['TIME'] >= '2023-01-01') & (df_bruto['TIME'] <= '2025-12-31 23:59:59')
    df_final = df_bruto[mask_periodo].copy()
    
    # 6. Ordenacao e Sequenciamento
    print("Ordenando indices espaciais e temporais...")
    df_final = df_final.sort_values(['SEGMENT_ID', 'TIME']).reset_index(drop=True)

    print("Gerando metadados de continuidade (Sequence ID)...")
    df_final['TIME_DIFF'] = df_final.groupby('SEGMENT_ID')['TIME'].diff()
    df_final['SEQUENCE_ID'] = (df_final['TIME_DIFF'] > pd.Timedelta(hours=1)).cumsum()
    
    print("Criando mascara de dados validos...")
    df_final['MASK'] = (df_final['SPEED'] > 0).astype('int8')

    df_final.drop(columns=['TIME_DIFF'], inplace=True)
    
    # 7. Salvamento
    print(f"Salvando dataset unificado com {len(df_final)} linhas...")
    df_final.to_csv(arquivo_saida, index=False)
    print(f"Processo concluido. Arquivo salvo em: {arquivo_saida}")

if __name__ == "__main__":
    unir_datasets()