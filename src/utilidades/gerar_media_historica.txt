import numpy as np
import pandas as pd
from pathlib import Path

def construir_perfil_historico():
    print("Iniciando construção do Perfil Histórico Espaço-Temporal (Time-of-Day HA)...")
    
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    caminho_dados = data_dir / "data_treinamento.npy"
    caminho_mask = data_dir / "mask_treinamento.npy"
    caminho_tempo = data_dir / "time_treinamento.npy"
    
    if not caminho_tempo.exists():
        raise FileNotFoundError("time_treinamento.npy não encontrado. Certifique-se de ter rodado o dataCleaning.py atualizado.")
        
    dados = np.load(caminho_dados).squeeze(-1) # (T, N)
    mask = np.load(caminho_mask).squeeze(-1)   # (T, N)
    tempos = np.load(caminho_tempo, allow_pickle=True)
    
    print(f"-> Analisando {len(tempos)} janelas temporais de treinamento...")
    
    # Converte os tempos para datetime e mapeia para 144 intervalos diários de 10 min
    df_tempos = pd.to_datetime(tempos)
    idx_intervalos = (df_tempos.hour * 6) + (df_tempos.minute // 10)
    
    num_nos = dados.shape[1]
    perfil_ha = np.zeros((144, num_nos), dtype=np.float32)
    
    for i in range(144):
        mask_horario = (idx_intervalos == i)
        dados_horario = dados[mask_horario]
        mask_dados_horario = mask[mask_horario]
        
        soma = np.sum(dados_horario * mask_dados_horario, axis=0)
        qtd = np.sum(mask_dados_horario, axis=0)
        
        # Média isolada para cada via naquele horário específico (ignora zeros onde não houve leitura)
        perfil_ha[i] = np.divide(soma, qtd, out=np.zeros_like(soma), where=qtd!=0)
        
    caminho_saida = data_dir / "ha_profile.npy"
    np.save(caminho_saida, perfil_ha)
    print(f"✅ Perfil HA gerado com sucesso! Dimensões: {perfil_ha.shape} (144 Horários x {num_nos} Vias). Salvo em {caminho_saida.name}")

if __name__ == "__main__":
    construir_perfil_historico()