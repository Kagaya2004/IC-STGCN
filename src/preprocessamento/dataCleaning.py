import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def executar_limpeza_stgcn(caminho_csv, pasta_saida):
    print("Iniciando processamento avançado (multi-feature)...")
    
    # 1. Carregar Dados e Metadados para manter a ordem dos 60 nós
    df = pd.read_csv(caminho_csv)
    df['TIME'] = pd.to_datetime(df['TIME'])
    
    # Carregamos os metadados para garantir que o índice 0-59 seja sempre o mesmo
    meta = pd.read_csv(os.path.join(pasta_saida, "node_metadata.csv"))
    id_para_idx = dict(zip(meta['SEGMENT_ID'], meta['NODE_INDEX']))
    num_nós = len(meta)
    
    # 2. Limpeza Técnica de Velocidade
    # Filtro de outliers: 0mph costuma ser erro de sensor em Chicago
    df.loc[(df['SPEED'] <= 0) | (df['SPEED'] > 80), 'SPEED'] = np.nan
    
    # 3. Pivotagem Multi-Feature
    # Criamos o mapeamento de nós
    df['node_idx'] = df['SEGMENT_ID'].map(id_para_idx)
    
    print("Pivotando canais de Velocidade e Dia da Semana...")
    # Canal 1: Velocidade
    pivot_speed = df.pivot_table(index='TIME', columns='node_idx', values='SPEED')
    
    # Canal 2: Dia da Semana (1=Dom, 7=Sáb)
    pivot_day = df.pivot_table(index='TIME', columns='node_idx', values='DAY_OF_WEEK')

    # 4. Resample Temporal (15 minutos)
    # Padroniza a frequência para as camadas de TCN da STGCN
    res_speed = pivot_speed.resample('15Min').mean()
    res_day = pivot_day.resample('15Min').first() # Dia da semana é constante na janela

    # 5. Interpolação e Tratamento da Lacuna
    # Preenchemos falhas pequenas (<1h), mas a lacuna de 6 meses continua como NaN
    res_speed = res_speed.ffill(limit=2).interpolate(method='linear', limit=4)
    res_day = res_day.ffill() # Dia da semana é fácil de preencher

    # 6. Normalização Z-Score (Crucial para convergência)
    print("Normalizando dados...")
    scaler_speed = StandardScaler()
    # Para o dia da semana, normalizamos para escala 0-1 ou Z-score
    scaler_day = StandardScaler()

    # Preenchemos o que sobrou com a média apenas para o cálculo do scaler
    norm_speed = scaler_speed.fit_transform(res_speed.fillna(res_speed.mean()))
    norm_day = scaler_day.fit_transform(res_day.fillna(res_day.mean()))

    # Salvar estatísticas para desnormalização na fase de predição real
    stats = pd.DataFrame({
        'feature': ['speed', 'day_of_week'],
        'mean': [scaler_speed.mean_.mean(), scaler_day.mean_.mean()],
        'std': [np.sqrt(scaler_speed.var_).mean(), np.sqrt(scaler_day.var_).mean()]
    })
    stats.to_csv(os.path.join(pasta_saida, "scaler_stats.csv"), index=False)

    # 7. Construção do Tensor Final (Timesteps, Canais, Nós)
    # Formato esperado: (T, 2, 60)
    data_final = np.stack([norm_speed, norm_day], axis=1).astype('float32')

    # 8. Salvamento Binário
    npy_path = os.path.join(pasta_saida, "data_chicago.npy")
    np.save(npy_path, data_final)
    
    print("-" * 40)
    print(f"PROCESSAMENTO CONCLUÍDO!")
    print(f"Shape do Tensor: {data_final.shape} (Timesteps, Features, Nós)")
    print(f"Arquivo salvo: {npy_path}")
    print(f"Metadados preservados em: node_metadata.csv")
    print("-" * 40)

if __name__ == "__main__":
    caminho_merged = r"C:\IC\Aplication\data\tratados\dataset_unificado_stgcn.csv"
    diretorio_data = r"C:\IC\Aplication\data\tratados"
    executar_limpeza_stgcn(caminho_merged, diretorio_data)