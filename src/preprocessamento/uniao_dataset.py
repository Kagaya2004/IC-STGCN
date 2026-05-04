import pandas as pd
import glob
import os

def consolidar_chicago():
    # 1. Definição de Caminhos
    base_dir = r"C:\IC\Aplication"
    pasta_2023 = os.path.join(base_dir, "data", "brutos", "data_2023")
    pasta_2024 = os.path.join(base_dir, "data", "brutos", "data_2024")
    
    arquivo_saida = os.path.join(base_dir, "data", "tratados", "dataset_unificado_stgcn.csv")
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    # 2. Todas as colunas necessárias (Treino + Adjacência + Metadados)
    colunas_uniao = [
        'TIME', 'SEGMENT_ID', 'SPEED', 'STREET', 'FROM_STREET', 
        'TO_STREET', 'LENGTH', 'DAY_OF_WEEK', 'START_LATITUDE', 'START_LONGITUDE'
    ]
    
    # Coletar arquivos de ambas as pastas
    arquivos_2023 = glob.glob(os.path.join(pasta_2023, "*.csv"))
    arquivos_2024 = glob.glob(os.path.join(pasta_2024, "*.csv"))
    todos_arquivos = arquivos_2023 + arquivos_2024

    if not todos_arquivos:
        print(f"ERRO: Nenhum arquivo .csv encontrado.")
        return

    print(f"Unindo {len(todos_arquivos)} arquivos viários...")

    lista_dfs = []
    for f in todos_arquivos:
        # low_memory=False ajuda com a RTX 3050 a evitar avisos de tipos mistos
        df_temp = pd.read_csv(f, usecols=colunas_uniao, low_memory=False)
        lista_dfs.append(df_temp)

    # 3. União e Tratamento de Tempo 
    df_bruto = pd.concat(lista_dfs).drop_duplicates()
    
    print("Convertendo datas (otimizado)...")
    # Especificar o formato evita a lentidão e o aviso do dateutil
    df_bruto['TIME'] = pd.to_datetime(df_bruto['TIME'], format='%m/%d/%Y %I:%M:%S %p')

    # 4. Filtragem por Intervalos de Tempo (Removendo a lacuna de 6 meses)
    bloco_2023 = df_bruto[
        (df_bruto['TIME'] >= '2023-01-01 00:00:00') & 
        (df_bruto['TIME'] <= '2023-09-08 12:51:14')
    ]

    bloco_2024_2025 = df_bruto[
        (df_bruto['TIME'] >= '2024-03-05 09:20:34') & 
        (df_bruto['TIME'] <= '2025-12-31 11:50:24')
    ]

    # 5. Consolidação e Criação do Sequence ID
    df_final = pd.concat([bloco_2023, bloco_2024_2025])
    df_final = df_final.sort_values(['SEGMENT_ID', 'TIME']).reset_index(drop=True)

    # Proteção contra lacunas: se o tempo pular > 24h, nova sequência
    df_final['gap'] = df_final.groupby('SEGMENT_ID')['TIME'].diff() > pd.Timedelta(days=1)
    df_final['sequence_id'] = df_final.groupby('SEGMENT_ID')['gap'].cumsum()
    df_final = df_final.drop(columns=['gap'])

    # 6. Salvar Dataset Unificado
    df_final.to_csv(arquivo_saida, index=False)
    
    print("\n" + "="*45)
    print(f"SUCESSO NA UNIFICAÇÃO!")
    print(f"Colunas salvas: {df_final.columns.tolist()}")
    print(f"Total de Registros: {len(df_final)}")
    print(f"Nós (Segmentos) Ativos: {df_final['SEGMENT_ID'].nunique()}")
    print(f"Arquivo gerado: {arquivo_saida}")
    print("="*45)

if __name__ == "__main__":
    consolidar_chicago()