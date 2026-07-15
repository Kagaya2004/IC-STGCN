import pandas as pd
from pathlib import Path

def filtrar_por_bounding_box():
    # 1. Configurações Dinâmicas de Caminho
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "tratados"
    
    # 2. Bounding Box do Centro de Chicago (The Loop)
    LAT_MIN, LAT_MAX = 41.8600, 41.8900
    LON_MIN, LON_MAX = -87.6400, -87.6100
    
    splits = ["treinamento", "validacao", "teste"]
    
    print("Iniciando Recorte Espacial (Bounding Box: The Loop - Chicago)...\n")
    
    for split in splits:
        arquivo_csv = data_dir / f"{split}.csv"
        
        if not arquivo_csv.exists():
            print(f"Erro: Arquivo base não encontrado: {arquivo_csv}")
            continue
            
        print(f"[{split.upper()}] Carregando dados...")
        df = pd.read_csv(arquivo_csv, engine='c', low_memory=False)
        linhas_antes = len(df)
        
        if linhas_antes == 0:
            print(f"[{split.upper()}] ALERTA: Arquivo vazio. Rode o uniao_dataset.py novamente.\n")
            continue
            
        # 3. Aplicação do Filtro Espacial (Bounding Box)
        mask = (
            (df['START_LATITUDE'] >= LAT_MIN) & 
            (df['START_LATITUDE'] <= LAT_MAX) & 
            (df['START_LONGITUDE'] >= LON_MIN) & 
            (df['START_LONGITUDE'] <= LON_MAX)
        )
        
        df_filtrado = df[mask]
        
        linhas_depois = len(df_filtrado)
        retencao = (linhas_depois / linhas_antes) * 100 if linhas_antes > 0 else 0
        
        print(f"[{split.upper()}] Retenção: {linhas_antes} -> {linhas_depois} leituras ({retencao:.2f}%).")
        
        # 4. Salvamento
        df_filtrado.to_csv(arquivo_csv, index=False)
        print(f"[{split.upper()}] Arquivo atualizado com sucesso.\n")

if __name__ == "__main__":
    filtrar_por_bounding_box()