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
    meta_path = os.path.join(pasta_saida, "node_metadata.csv")
    if not os.path.exists(meta_path):
        print(f"❌ Erro: {meta_path} não encontrado. Execute matriz_adjacencia.py primeiro.")
        return

    meta = pd.read_csv(meta_path)
    sensores_validos = meta['SEGMENT_ID'].tolist()
    df = df[df['SEGMENT_ID'].isin(sensores_validos)].copy()
    
    num_nós = len(meta)

    # 2. Sincronização Temporal (Bin de 10 min)
    print("⏱️ Sincronizando frequências de 10 min...")
    df['TIME'] = df['TIME'].dt.round('10min')
    
    # Pivotagem
    pivot_speed = df.pivot_table(index='TIME', columns='SEGMENT_ID', values='SPEED')
    pivot_speed = pivot_speed.reindex(columns=sensores_validos)

    # 3. Tratamento de Gaps e Interpolação Limitada
    print("🩹 Aplicando interpolação seletiva...")
    pivot_speed = pivot_speed.interpolate(method='linear', limit=3, axis=0)

    # 4. Geração da Máscara de Perda (Masking)
    mask_tensor = (~pivot_speed.isna() & (pivot_speed > 0)).astype(float).values

    # 5. Engenharia de Atributos Temporais (Cíclicos) - CORREÇÃO PANDAS 3.0
    print("📅 Gerando atributos temporais cíclicos (Seno/Cosseno)...")
    times = pivot_speed.index
    
    # IMPORTANTE: Convertemos para numpy array (.to_numpy()) antes do cálculo
    # Isso evita o erro de Multi-dimensional indexing no Pandas 3.0+
    hours_array = times.hour.to_numpy()
    hour_sin = np.sin(2 * np.pi * hours_array / 24)
    # hour_cos = np.cos(2 * np.pi * hours_array / 24) # Opcional se quiser adicionar mais canais

    # 6. Normalização (Z-Score)
    print("📊 Normalizando dados...")
    scaler = StandardScaler()
    # Preenchemos NaNs com a média para o scaler, mas a MASK cuidará disso no treino
    speed_array = pivot_speed.fillna(pivot_speed.mean()).values
    speed_norm = scaler.fit_transform(speed_array)

    # Salvar estatísticas para desnormalização
    stats = pd.DataFrame({
        'feature': ['speed'],
        'mean': [scaler.mean_.mean()],
        'std': [np.sqrt(scaler.var_).mean()]
    })
    stats.to_csv(os.path.join(pasta_saida, "scaler_stats.csv"), index=False)

    # 7. Construção do Tensor Final (T, N, C) - CORREÇÃOtile
    # hour_sin já é um array numpy, então [:, np.newaxis] agora funciona perfeitamente
    hour_feature = np.tile(hour_sin[:, np.newaxis], (1, num_nós))
    
    # Stack final: Canal 0 = Velocidade, Canal 1 = Hora (Seno)
    data_final = np.stack([speed_norm, hour_feature], axis=-1).astype('float32')

    # 8. Salvamento Binário
    np.save(os.path.join(pasta_saida, "data_chicago.npy"), data_final)
    np.save(os.path.join(pasta_saida, "mask_chicago.npy"), mask_tensor)
    
    print(f"✅ Tensores gerados com sucesso: {data_final.shape}")
    print(f"💾 Arquivos salvos em: {pasta_saida}")

if __name__ == "__main__":
    pasta_tratados = r"C:\IC\Aplication\data\tratados"
    csv_unificado = os.path.join(pasta_tratados, "dataset_unificado_stgcn.csv")
    
    processar_tensores_stgcn(csv_unificado, pasta_tratados)