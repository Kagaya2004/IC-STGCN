import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def processar_tensores_treinamento():
    # 1. Configuracoes de Caminho
    base_dir = r"C:\IC\Aplication"
    data_dir = os.path.join(base_dir, "data", "tratados")
    caminho_csv = os.path.join(data_dir, "dataset_filtrado.csv")
    caminho_meta = os.path.join(data_dir, "node_metadata.csv")
    
    if not os.path.exists(caminho_csv) or not os.path.exists(caminho_meta):
        print("Erro: Arquivos necessarios (CSV unificado ou Metadados) nao encontrados.")
        return

    # 2. Carregamento de Metadados para Alinhamento de Nos
    print("Carregando metadados e alinhando ordem dos sensores...")
    meta = pd.read_csv(caminho_meta)
    ordem_sensores = meta['SEGMENT_ID'].tolist()
    num_nos = len(ordem_sensores)

    # 3. Carregamento e Sincronizacao Temporal
    print("Lendo dataset unificado (processo intensivo de memoria)...")
    # Carregamento seletivo de colunas para preservar memoria RAM
    df = pd.read_csv(caminho_csv, usecols=['TIME', 'SEGMENT_ID', 'SPEED'])
    df['TIME'] = pd.to_datetime(df['TIME'])
    
    # Arredondamento para janelas de 10 minutos conforme arquitetura do modelo
    df['TIME'] = df['TIME'].dt.round('10min')
    
    # Pivotagem: Conversao de registros em matriz (Tempo x Sensores)
    print("Executando pivotagem de matriz temporal...")
    pivot_speed = df.pivot_table(index='TIME', columns='SEGMENT_ID', values='SPEED', aggfunc='mean')
    
    # Reordenacao para garantir que a ordem dos sensores coincida com a Matriz de Adjacencia
    pivot_speed = pivot_speed.reindex(columns=ordem_sensores)

    # 4. Normalizacao da Continuidade Temporal
    print("Normalizando continuidade do indice temporal...")
    inicio, fim = pivot_speed.index.min(), pivot_speed.index.max()
    full_index = pd.date_range(start=inicio, end=fim, freq='10min')
    pivot_speed = pivot_speed.reindex(full_index)

    # 5. Criacao da Mascara de Dados Reais
    print("Gerando tensor de mascara binaria...")
    # A mascara identifica lacunas reais no dataset (1 = dado real, 0 = gap)
    mask_array = (~pivot_speed.isna()).astype('int8').values

    # 6. Interpolacao e Normalizacao Z-Score
    print("Tratando valores ausentes via interpolacao linear...")
    # Interpola lacunas de ate 30 minutos; gaps maiores sao preenchidos com a media
    pivot_speed = pivot_speed.interpolate(method='linear', axis=0, limit=3).fillna(pivot_speed.mean())
    
    print("Aplicando normalizacao Z-Score...")
    scaler = StandardScaler()
    speed_norm = scaler.fit_transform(pivot_speed.values)

    # Armazenamento de estatisticas globais para desnormalizacao posterior
    stats = pd.DataFrame({
        'mean': [scaler.mean_.mean()],
        'std': [np.sqrt(scaler.var_).mean()]
    })
    stats.to_csv(os.path.join(data_dir, "scaler_stats.csv"), index=False)

    # 7. Engenharia de Atributos Temporais (Resolucao do Erro de Indexacao)
    print("Extraindo atributos de periodicidade horaria...")
    # Calculo da hora fracionaria
    hours_float = pivot_speed.index.hour + pivot_speed.index.minute / 60.0
    
    # Conversao explicita para NumPy antes da aplicacao de np.newaxis
    # Esta etapa corrige o erro de Multi-dimensional indexing relatado
    hour_sin = np.sin(2 * np.pi * hours_float.to_numpy() / 24.0)
    
    # Expansao do vetor temporal para a forma (Tempo, Nos)
    hour_feature = np.tile(hour_sin[:, np.newaxis], (1, num_nos))

    # 8. Consolidacao e Armazenamento dos Tensores
    # Tensor final em formato (T, N, C) -> (Passos Temporais, Sensores, Canais de Feature)
    print("Consolidando tensores finais...")
    data_final = np.stack([speed_norm, hour_feature], axis=-1).astype('float32')
    
    np.save(os.path.join(data_dir, "data_chicago.npy"), data_final)
    np.save(os.path.join(data_dir, "mask_chicago.npy"), mask_array)

    print(f"Sucesso: Tensores gerados com formato {data_final.shape}")
    print(f"Arquivos salvos em: {data_dir}")

if __name__ == "__main__":
    processar_tensores_treinamento()