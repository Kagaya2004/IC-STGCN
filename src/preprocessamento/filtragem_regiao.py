import pandas as pd
from pathlib import Path

def filtrar_regiao():
    # 1. Configuracoes de Caminho
    base_dir = Path(__file__).resolve().parents[2]
    input_file = base_dir / "data" / "tratados" / "dataset_unificado.csv"
    output_file = base_dir / "data" / "tratados" / "dataset_filtrado.csv"
    
    if not input_file.exists():
        print(f"Erro: Arquivo {input_file} não encontrado.")
        return

    print("Carregando dataset para filtragem espacial...")
    df = pd.read_csv(input_file)

    # 2. Definicao da Bounding Box (Loop de Chicago)
    LAT_MIN, LAT_MAX = 41.8670, 41.8960
    LON_MIN, LON_MAX = -87.6470, -87.6150

    # 3. Filtragem Espacial Vetorizada
    # Aplicamos a condição estrita: ambos os pontos (início e fim) devem estar no polígono
    print("Aplicando filtro de Bounding Box...")
    mask = (
        (df['START_LATITUDE'] >= LAT_MIN) & (df['START_LATITUDE'] <= LAT_MAX) &
        (df['END_LATITUDE'] >= LAT_MIN) & (df['END_LATITUDE'] <= LAT_MAX) &
        (df['START_LONGITUDE'] >= LON_MIN) & (df['START_LONGITUDE'] <= LON_MAX) &
        (df['END_LONGITUDE'] >= LON_MIN) & (df['END_LONGITUDE'] <= LON_MAX)
    )
    
    df_filtrado = df.loc[mask].copy()

    # 4. Limpeza de Integridade
    # Removemos linhas que possuam nulos nas colunas essenciais para o grafo/STGCN
    cols_criticas = ['SPEED', 'TIME', 'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE', 'LENGTH']
    df_filtrado = df_filtrado.dropna(subset=cols_criticas)
    
    # Remove duplicacoes completas (sensores que reportaram no mesmo instante)
    df_filtrado = df_filtrado.drop_duplicates()

    # 5. Exportacao
    df_filtrado.to_csv(output_file, index=False)
    print(f"Filtragem concluída. Arquivo salvo em: {output_file}")
    print(f"Linhas preservadas: {len(df_filtrado)} de {len(df)}")

if __name__ == "__main__":
    filtrar_regiao()