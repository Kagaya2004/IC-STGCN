import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def processar_tensores_stgcn(caminho_csv, pasta_saida):
    print("🧪 Iniciando processamento de tensores para STGCN...")
    
    # 1. Carregamento e Alinhamento de Sensores
    df = pd.read_csv(caminho_csv)
    df['TIME'] = pd.to_datetime(df['TIME'])
    
    # Carregamos apenas os sensores validados na etapa da Matriz de Adjacência
    meta = pd.read_csv(os.path.join(pasta_saida, "node_metadata.csv"))
    sensores_validos = meta['SEGMENT_ID'].tolist()
    df = df[df['SEGMENT_ID'].isin(sensores_validos)].copy()
    
    id_para_idx = dict(zip(meta['SEGMENT_ID'], meta['NODE_INDEX']))
    num_nós = len(meta)

    # 2. Sincronização Temporal (Bin de 10 min) por Sequência
    print("⏱️ Sincronizando frequências de 10 min...")
    df['TIME'] = df['TIME'].dt.round('10min')
    
    # Pivotagem: Criamos uma estrutura [Tempo x Sensor] para Velocidade e Máscara
    # Preenchemos com NaN para identificar buracos reais
    pivot_speed = df.pivot_table(index='TIME', columns='SEGMENT_ID', values='SPEED')
    
    # Reordenar colunas conforme o NODE_INDEX da matriz de adjacência
    pivot_speed = pivot_speed.reindex(columns=sensores_validos)

    # 3. Tratamento de Gaps e Interpolação Limitada
    # Só interpolamos falhas menores que 30 minutos (3 registros)
    # Gaps maiores que isso (como o de 9 meses) permanecem como NaN
    print("🩹 Aplicando interpolação seletiva...")
    pivot_speed = pivot_speed.interpolate(method='linear', limit=3, axis=0)

    # 4. Geração da Máscara de Perda (Masking)
    # 1.0 onde temos dado, 0.0 onde é NaN ou Zero
    mask_tensor = (~pivot_speed.isna() & (pivot_speed > 0)).astype(float).values

    # 5. Engenharia de Atributos Temporais (Cíclicos)
    print("📅 Gerando atributos temporais cíclicos (Seno/Cosseno)...")
    times = pivot_speed.index
    # Hora do dia (0-23)
    hour_sin = np.sin(2 * np.pi * times.hour / 24)
    hour_cos = np.cos(2 * np.pi * times.hour / 24)
    
    # 6. Normalização (Z-Score)
    # Preenchemos NaNs temporariamente com a média para o scaler não quebrar
    scaler = StandardScaler()
    speed_array = pivot_speed.fillna(pivot_speed.mean()).values
    speed_norm = scaler.fit_transform(speed_array)

    # Salvar estatísticas para desnormalização no Frontend
    stats = pd.DataFrame({
        'feature': ['speed'],
        'mean': [scaler.mean_.mean()],
        'std': [np.sqrt(scaler.var_).mean()]
    })
    stats.to_csv(os.path.join(pasta_saida, "scaler_stats.csv"), index=False)

    # 7. Construção do Tensor Final (T, N, C)
    # Canal 0: Velocidade Normalizada
    # Canal 1: Hora do Dia (Seno) - Ajuda a IA a entender picos de manhã/tarde
    
    # Broadcast da hora para todos os nós
    hour_feature = np.tile(hour_sin[:, np.newaxis], (1, num_nós))
    
    data_final = np.stack([speed_norm, hour_feature], axis=-1).astype('float32')

    # 8. Salvamento Binário
    np.save(os.path.join(pasta_saida, "data_chicago.npy"), data_final)
    np.save(os.path.join(pasta_saida, "mask_chicago.npy"), mask_tensor)
    
    print(f"✅ Tensores gerados: {data_final.shape}")
    print(f"💾 Arquivos salvos em: {pasta_saida}")

if __name__ == "__main__":
    pasta = r"C:\IC\Aplication\data\tratados"
    processar_tensores_stgcn(
        os.path.join(pasta, "dataset_unificado_stgcn.csv"),
        pasta
    )