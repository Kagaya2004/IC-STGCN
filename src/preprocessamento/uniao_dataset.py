import pandas as pd
import glob
import os
import numpy as np

def consolidar_chicago_robusto():
    # 1. Configurações de Caminho
    base_dir = r"C:\IC\Aplication"
    pasta_2023 = os.path.join(base_dir, "data", "brutos", "data_2023")
    pasta_2024 = os.path.join(base_dir, "data", "brutos", "data_2024")
    arquivo_saida = os.path.join(base_dir, "data", "tratados", "dataset_unificado_stgcn.csv")
    
    # Criar pasta de destino se não existir
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    # 2. Coleta de Ficheiros
    print("🔍 Localizando arquivos CSV...")
    arquivos = glob.glob(os.path.join(pasta_2023, "*.csv")) + glob.glob(os.path.join(pasta_2024, "*.csv"))
    
    if not arquivos:
        print("❌ Nenhum arquivo encontrado. Verifique os caminhos.")
        return

    lista_dfs = []
    # Selecionamos apenas o necessário para economizar memória
    colunas_foco = ['TIME', 'SEGMENT_ID', 'SPEED', 'START_LATITUDE', 'START_LONGITUDE', 'STREET']

    for f in arquivos:
        temp_df = pd.read_csv(f, usecols=colunas_foco)
        lista_dfs.append(temp_df)

    print(f"📖 Lidos {len(lista_dfs)} arquivos. Concatenando...")
    df_bruto = pd.concat(lista_dfs, ignore_index=True)

    # 3. Tratamento de Tipos e Datas (Otimização de Memória)
    print("⚙️ Processando tipos de dados e datas...")
    # O formato '%m/%d/%Y %I:%M:%S %p' é crucial para a velocidade de conversão em Chicago
    df_bruto['TIME'] = pd.to_datetime(df_bruto['TIME'], format='%m/%d/%Y %I:%M:%S %p')
    df_bruto['SEGMENT_ID'] = df_bruto['SEGMENT_ID'].astype('int32')
    df_bruto['SPEED'] = df_bruto['SPEED'].astype('float32')

    # 4. Filtragem dos Intervalos Solicitados (Exatos)
    print("⏳ Filtrando períodos de tempo...")
    mask_2023 = (df_bruto['TIME'] >= '2023-01-01 00:00:00') & (df_bruto['TIME'] <= '2023-09-08 12:51:14')
    mask_2024 = (df_bruto['TIME'] >= '2024-03-05 09:20:34') & (df_bruto['TIME'] <= '2025-12-31 11:50:24')
    
    df_final = df_bruto[mask_2023 | mask_2024].copy()
    
    # Ordenação física: vital para a lógica de sequência temporal
    df_final = df_final.sort_values(['SEGMENT_ID', 'TIME']).reset_index(drop=True)

    # 5. ESTRATÉGIA 1: Gestão de Macro-Gaps (Sequence ID)
    # Identifica onde a continuidade temporal é quebrada (salto > 1 hora)
    print("🔗 Gerando IDs de Sequência (Estratégia Anti-Gap)...")
    df_final['TIME_DIFF'] = df_final.groupby('SEGMENT_ID')['TIME'].diff()
    # Criamos um novo ID sempre que o tempo pula mais de 1 hora ou mudamos de sensor
    df_final['SEQUENCE_ID'] = (df_final['TIME_DIFF'] > pd.Timedelta(hours=1)).cumsum()

    # 6. ESTRATÉGIA 3: Gestão de Micro-Gaps (Loss Masking)
    # Criamos uma máscara binária: 1 para dados reais, 0 para leituras inválidas
    print("🎭 Criando Máscara de Perda (Estratégia Masking)...")
    df_final['MASK'] = (df_final['SPEED'] > 0).astype('int8')

    # 7. Limpeza Final e Exportação
    df_final.drop(columns=['TIME_DIFF'], inplace=True)
    
    print(f"💾 Salvando dataset final ({len(df_final)} linhas)...")
    df_final.to_csv(arquivo_saida, index=False)
    print(f"✅ Sucesso! Arquivo salvo em: {arquivo_saida}")

if __name__ == "__main__":
    consolidar_chicago_robusto()