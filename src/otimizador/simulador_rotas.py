import numpy as np
import pandas as pd

def simular_google_sintetico(rota_indices, df_meta, data_npy, stats_path):
    """
    Simula o cálculo de tempo de uma aplicação comercial reativa.
    Usa a média de velocidade do sensor em vez da predição espaço-temporal.
    """
    # 1. Carregar estatísticas para desnormalização
    stats = pd.read_csv(stats_path)
    mean_speed = stats.loc[stats['feature'] == 'speed', 'mean'].values[0]
    std_speed = stats.loc[stats['feature'] == 'speed', 'std'].values[0]

    # 2. Obter a velocidade atual (último registro do dataset) 
    # Isso simula o trânsito "agora", que é o que o GPS comum costuma usar.
    ultimo_snapshot = data_npy[-1, 0, :] # Canal 0 (Velocidade), último timestep
    velocidades_reais = (ultimo_snapshot * std_speed) + mean_speed
    
    tempo_total_segundos = 0
    
    # 3. Percorrer a rota e somar os tempos baseados no trânsito reativo
    # Note: Usamos os nomes das colunas que limpamos no passo anterior
    for i in range(len(rota_indices) - 1):
        idx_destino = rota_indices[i+1]
        
        # Obter distância do segmento
        distancia = float(df_meta.loc[df_meta['NODE_INDEX'] == idx_destino, 'LENGTH'].values[0])
        
        # Obter velocidade "reativa" (sem predição de futuro)
        # Adicionamos um pequeno ruído de 5% para simular a imprecisão de apps comerciais
        v_comercial = velocidades_reais[idx_destino] * np.random.uniform(0.95, 1.05)
        v_comercial = max(v_comercial, 5.0) # Segurança contra divisão por zero
        
        tempo_segmento = (distancia / v_comercial) * 3600
        tempo_total_segundos += tempo_segmento
        
    return tempo_total_segundos / 60 # Retorna em minutos