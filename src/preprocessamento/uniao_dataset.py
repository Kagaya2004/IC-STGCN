import pandas as pd
import numpy as np
from pathlib import Path

def unir_datasets():
    # 1. Configuração de Caminhos Relativos (Portabilidade)
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "brutos"
    arquivo_saida = base_dir / "data" / "tratados" / "dataset_unificado.csv"
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

    arquivos = list(raw_dir.glob('**/*.csv'))
    if not arquivos:
        print("Erro: Nenhum ficheiro CSV encontrado.")
        return

    # 2. NOVO: Dicionário atualizado para 9 colunas essenciais
    mapeamento_colunas = {
        'TIME': ['TIME', '_LAST_UPDATED', 'TIMESTAMP', '_TIME', 'LAST UPDATED'],
        'SPEED': ['SPEED', '_SPEED', 'CURRENTSPEED', 'CURRENT SPEED'],
        'SEGMENT_ID': ['SEGMENT_ID', 'SEGMENTID', 'SEGMENT ID'],
        'STREET': ['STREET', 'STREET_NAME', 'STREET NAME', 'STREETNAME'],
        'START_LATITUDE': ['START_LATITUDE', 'START_LAT', 'LATITUDE', 'START LATITUDE'],
        'START_LONGITUDE': ['START_LONGITUDE', 'START_LON', 'LONGITUDE', 'START LONGITUDE'],
        # Colunas espaciais obrigatórias adicionadas para a Bounding Box e Grafo
        'END_LATITUDE': ['END_LATITUDE', 'END_LAT'],
        'END_LONGITUDE': ['END_LONGITUDE', 'END_LON'],
        'LENGTH': ['LENGTH', 'STREET_LENGTH', 'SEGMENT_LENGTH']
    }

    # Função interna para Downcasting (Otimização de Memória)
    def otimizar_memoria(df):
        cols_float = df.select_dtypes(include=['float64']).columns
        cols_int = df.select_dtypes(include=['int64']).columns
        df[cols_float] = df[cols_float].astype('float32')
        df[cols_int] = df[cols_int].astype('int32')
        return df

    lista_dfs = []
    
    # 3. Leitura Resiliente e Mapeamento
    for f in arquivos:
        try:
            # Leitura do cabeçalho bruto para tratamento de corrupção
            with open(f, 'r', encoding='utf-8-sig') as file:
                header_line = file.readline()
                cols_limpas = [c.replace('"', '').replace(' ', '_').strip().upper() for c in header_line.split(',')]
            
            # Lê os dados ignorando o cabeçalho problemático
            df_temp = pd.read_csv(f, skiprows=1, names=cols_limpas, sep=',', engine='c', low_memory=False)
            
            mapeamento_final = {}
            for alvo, sinonimos in mapeamento_colunas.items():
                for s in sinonimos:
                    if s.upper() in cols_limpas:
                        mapeamento_final[s.upper()] = alvo
                        break
            
            # NOVO: A validação agora exige rigorosamente as 9 colunas mapeadas
            if len(mapeamento_final) == 9:
                df_temp = df_temp[list(mapeamento_final.keys())]
                df_temp = df_temp.rename(columns=mapeamento_final)
                
                # Otimização de memória precoce
                df_temp = otimizar_memoria(df_temp)
                lista_dfs.append(df_temp)
            else:
                faltantes = set(mapeamento_colunas.keys()) - set(mapeamento_final.values())
                print(f"Aviso: Ficheiro {f.name} ignorado. Faltam colunas: {faltantes}")

        except Exception as e:
            print(f"Alerta: Ficheiro {f.name} ignorado devido a erro de estrutura. Detalhe: {e}")

    if not lista_dfs:
        print("Erro: Nenhum dado extraído.")
        return

    # 4. Concatenação Pré-Tratamento Temporal
    print("A concatenar dataframes e a redefinir índices...")
    df_final = pd.concat(lista_dfs, ignore_index=True)

    # 5. Tratamento Temporal Nível Macro
    print("A processar datas e lotes temporais...")
    coluna_tempo = 'TIME'
    df_final[coluna_tempo] = pd.to_datetime(df_final[coluna_tempo], errors='coerce')
    df_final = df_final.dropna(subset=[coluna_tempo])

    # Lote 1 (2023) e Lote 2 (2024+)
    df_final['lote_temporal'] = np.where(df_final[coluna_tempo].dt.year == 2023, 1, 2)

    # 6. Tratamento Temporal Nível Micro (Obrigatório para STGCN)
    print("A gerar SEQUENCE_ID vetorizado...")
    coluna_segmento = 'SEGMENT_ID'
    
    # Ordenação estrita para garantir a cronologia correta por sensor/rua
    df_final = df_final.sort_values([coluna_segmento, coluna_tempo]).reset_index(drop=True)
    
    # Condição A: Mudança de segmento de rua
    mudanca_segmento = df_final[coluna_segmento] != df_final[coluna_segmento].shift(1)
    
    # Condição B: Quebra temporal maior que 10 minutos no mesmo segmento
    quebra_temporal = df_final[coluna_tempo].diff() > pd.Timedelta(minutes=10)
    
    # Cria o ID único para janelas temporais contínuas
    df_final['SEQUENCE_ID'] = (mudanca_segmento | quebra_temporal).cumsum()

    # 7. Exportação Segura
    print(f"A exportar {len(df_final)} registos tratados...")
    df_final.to_csv(arquivo_saida, index=False)
    print(f"Pré-processamento concluído com sucesso em: {arquivo_saida}")

if __name__ == "__main__":
    unir_datasets()