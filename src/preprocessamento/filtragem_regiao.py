import pandas as pd
import os

def filtrar_area_central():
    base_dir = r"C:\IC\Aplication"
    dir_tratados = os.path.join(base_dir, "data", "tratados")
    input_csv = os.path.join(dir_tratados, "dataset_unificado.csv")
    output_csv = os.path.join(dir_tratados, "dataset_filtrado.csv")

    print("Carregando dataset para filtragem espacial...")
    df = pd.read_csv(input_csv)

    # 1. Conversao de seguranca (caso existam virgulas)
    for col in ['START_LATITUDE', 'START_LONGITUDE']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # 2. Definicao do Bounding Box (Retangulo da imagem 2)
    lat_min, lat_max = 41.870, 41.898
    lon_min, lon_max = -87.650, -87.610

    # 3. Aplicacao do Filtro
    df_central = df[
        (df['START_LATITUDE'] >= lat_min) & (df['START_LATITUDE'] <= lat_max) &
        (df['START_LONGITUDE'] >= lon_min) & (df['START_LONGITUDE'] <= lon_max)
    ].copy()

    print(f"Dataset original: {len(df)} linhas.")
    print(f"Dataset centralizado: {len(df_central)} linhas.")
    print(f"Sensores (Segmentos) na area: {df_central['SEGMENT_ID'].nunique()}")

    # 4. Salvamento
    df_central.to_csv(output_csv, index=False)
    print(f"Sucesso: Novo dataset salvo em {output_csv}")

if __name__ == "__main__":
    filtrar_area_central()