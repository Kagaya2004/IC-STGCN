import pandas as pd

def unir_datasets_ic(arquivo_2023, arquivo_2024_2025):
    # Colunas essenciais para o modelo e grafo
    cols = ['TIME', 'SEGMENT_ID', 'SPEED', 'STREET', 'FROM_STREET', 'TO_STREET', 'LENGTH']
    
    # 1. Carregar e filtrar o Bloco 1 (2023)
    df1 = pd.read_csv(arquivo_2023, usecols=cols)
    df1['TIME'] = pd.to_datetime(df1['TIME'])
    # Filtro conforme image_704157.png: 01/01/2023 até 08/09/2023 às 12:51:14
    mask1 = (df1['TIME'] >= '2023-01-01 00:00:00') & (df1['TIME'] <= '2023-09-08 12:51:14')
    df1 = df1[mask1]
    
    # 2. Carregar e filtrar o Bloco 2 (2024-2025)
    df2 = pd.read_csv(arquivo_2024_2025, usecols=cols)
    df2['TIME'] = pd.to_datetime(df2['TIME'])
    # Filtro conforme image_704157.png: 05/03/2024 às 09:20:34 até 31/12/2025 às 11:50:24
    mask2 = (df2['TIME'] >= '2024-03-05 09:20:34') & (df2['TIME'] <= '2025-12-31 23:59:59')
    df2 = df2[mask2]
    
    # 3. Concatenar os blocos
    df_final = pd.concat([df1, df2]).drop_duplicates()
    df_final = df_final.sort_values(['SEGMENT_ID', 'TIME']).reset_index(drop=True)
    
    # 4. Lógica de 'Sequence ID' para a STGCN
    # Identifica a lacuna para que o gerador de janelas não cruze de 2023 para 2024
    # Se a diferença entre registros de um mesmo ID for > 24 horas, inicia nova sequência
    df_final['gap'] = df_final.groupby('SEGMENT_ID')['TIME'].diff() > pd.Timedelta(days=1)
    df_final['sequence_id'] = df_final.groupby('SEGMENT_ID')['gap'].cumsum()
    
    return df_final