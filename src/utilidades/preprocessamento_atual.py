# dataCleaning.py
import pandas as pd
import numpy as np
import os

# Função para realizar limpeza do dataset METR-LA
def limpeza_dataset_metrla(df):
    """
    Verificar a existência de colunas do tipo 'TimeStamp' ou 'Data_Hora'.
    """
    col_tempo = 'Timestamp' if 'Timestamp' in df.columns else ('Data_Hora' if 'Data_Hora' in df.columns else None)
    
    """
    Caso existe uma coluna temporal, ela é convertida para o time 'datetime'
    Além disso, ela é definida como índice da base, facilitando em operações posteriores,
    como manipulação temporal e interpolação.
    """
    if col_tempo:
        df[col_tempo] = pd.to_datetime(df[col_tempo])
        df = df.set_index(col_tempo)
    
    """
    No dataset METR-LA, a velocidade é 0km/h qunando ocorre alguma falha no sensor,
    Assim, esses valores são substituídos por NaN para não ocorrer erros durante a interpolação,
    considerando como um buraco a ser preenchido.
    """
    df = df.replace(0, np.nan)
    
    """
    Aqui é realizada a interpolação linear para preenchimento dos buracos no dataset, 
    indicados pelo valor NaN. Esse método utilizada os valores vizinhos (um antes e outro depois)
    para estimar o valor faltante.
    """
    df = df.interpolate(method='linear', axis=0)
    
    """
    Aqui é realizado o tratamento quando os valores faltantes são o primeiro (backward fill)  
    ou o último (forward fill) da série temporal. Assim o valor válido mais próximo é 
    utilizado para preencher essas lacunas.
    """
    df = df.ffill().bfill()
    
    """
    Como o Python usa float64 por padrão e visando reduzir consumo de VRAM pela metade da RTX3050
    Os dados float são convertidos para float32, permitindo uma melhor utilização da capacidade 
    de processamento da GPU.
    """
    df = df.astype(np.float32)
    
    """
    Verificação de segurança para a ocorrência de alguma falha na interpolação, excluindo tais linhas e notificando
    a existência dessa falha 
    """
    if df.isnull().values.any():
        print("Falha na Interpolação: Existem valores nulos na base após o processo de limpeza.")
        df = df.dropna()
        
    return df

def executar_limpeza_completa():
    # Raiz do Projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Caminhos do arquivo de entrada de dados, os dados brutos do dataset METR-LA
    input_path = os.path.join(base_dir, 'data', 'brutos', 'METR-LA.h5')
    # Caminho do arquivo de saída, onde a base após a limpeza e interpolação, será salvo e seu nome e formato
    output_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_cleaned.csv')

    # Criar a pasta 'tratados' caso ela não exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Carregando base de dados: {input_path}")
    
    try:
        # Verifica se o formato do dataset é .h5 ou .csv antes de carregá-lo
        if input_path.endswith('.h5'):
            df_bruto = pd.read_hdf(input_path)
        else:
            df_bruto = pd.read_csv(input_path)

        print("Iniciando processo de limpeza e interpolação...")
        # Chamada da função de limpeza e interpolação
        df_limpo = limpeza_dataset_metrla(df_bruto)

        """
        Salve o resultado do processo de DataCleaning em um arquivo .csv
        O valor True do index garante a coluna temporal seja preservada no arquivo de saída,
        mantendo a integridade da série temporal.
        """
        df_limpo.to_csv(output_path, index=True)
        print(f"Dataset limpa e salva em: {output_path}")
        print(f"Dimensões finais do Dataset: {df_limpo.shape}")

    except Exception as e:
        print(f"Erro ao processar os dados: {e}")

if __name__ == "__main__":
    executar_limpeza_completa()

# featureEngineering.py
import pandas as pd
import numpy as np
import os

def engineer_features_metr_la(df):
    # Verificar se o valor do índice não é do tipo DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        """ 
        Aqui é transformado o índice no tipo DateTimeIndex, facilitando 
        na extração de propriedades como hora e dia da semana.
        """
        df.index = pd.to_datetime(df.index)
    
    """
    Aplicação de Propriedades Temporais Cíclicas, através das funções
    Seno e Cosseno, na informações Hora, permitindo que os modelos 
    compreendam que horários como 23:59 e 00:00 estão próximos.
    """
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    
    """
    Aplicação de Propriedades Temporais Cíclicas, através das funções
    Seno e Cosseno, na informações Dua da Semana, permitindo que os modelos 
    compreendam que os dias como Domingo e Segunda estão próximos.
    """
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    
    return df

def executar_feature_engineering():
    # Caminho da raiz do projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Caminho do arquivo no qual é o dataset após a limpeza realizado pelo dataCleaning.py
    input_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_cleaned.csv')
    # Caminho do arquivo final após a aplicação da Engenharia de Atriibutos/Features Engineering
    output_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_featured.csv')

    # Verificação da existência do arquivo de entrada
    if not os.path.exists(input_path):
        print(f"Erro: O arquivo limpo não foi encontrado em {input_path}")
        return
    
    print(f"Carregando base limpa: {input_path}")
    
    """
    Ao carregar o dataset limpo, é indicado que a coluna 0 (a primeira),
    na qual contém o valor Data_Hora, é o índice do dataFrame.
    """
    df = pd.read_csv(input_path, index_col=0) 

    print("Convertendo os dadso de Hora e Dia da Semana em atributos ciclícos...")
    # Aplicação das transformações de Engenharia de Atributos no dataset limpo
    df_final = engineer_features_metr_la(df)

    # Salvando o arquivo final com as novas colunas, no formato .csv
    df_final.to_csv(output_path)
    print(f"Engenharia de Atributos aplicada com sucesso! Base salva em: {output_path}")
    print(f"Dimensões com novas colunas: {df_final.shape}")

if __name__ == "__main__":
    executar_feature_engineering()

# preparar_grafo.py
import pandas as pd
import pickle
import torch
import numpy as np
import os
import h5py
import scipy.sparse as sp

def calculate_random_walk_matrix(adj_mx):
    """
    Calcula a matriz de transição para a DCRNN (Random Walk).
    P = D_out^-1 * W
    """
    # Conversão da matriz para formato Coordinate
    adj_mx = sp.coo_matrix(adj_mx)
    """
    Cálculo do grau de saída (D_out) de cada sensor, ou seja, 
    # a força de todos as estradas que saem daquela ponto
    """
    d = np.array(adj_mx.sum(1)).flatten()
    # Cálculo do inverso da soma
    d_inv = np.power(d, -1).flatten()
    # Substituição de infinitos por zero (caso haja nós isolados), evitando a ocorrência de erros matemáticos
    d_inv[np.isinf(d_inv)] = 0.
    # Criação da matriz diagonal D^-1
    d_mat_inv = sp.diags(d_inv)
    # Produto escalar entre matriz diagonal invertida (D^-1) e a matriz de adjacência (W)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def preparar_tudo():
    # Raiz do projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Caminho do diretório de dados brutos
    raw_dir = os.path.join(base_dir, 'data', 'brutos')
    # Caminho do direótrio de dados processados
    processed_dir = os.path.join(base_dir, 'data', 'tratados')
    # Cria o diretório de dados processsados, caso não exista
    os.makedirs(processed_dir, exist_ok=True)

    print("Passo 1: Extraindo velocidades...")
    # Leitura do arquivo .h5
    h5_path = os.path.join(raw_dir, 'METR-LA.h5')
    with h5py.File(h5_path, 'r') as f:
        # Extrai a matriz bruta de velocidades
        data = f['df']['block0_values'][:]
        # Coleta os nomes dos sensores a cada registro de velocidade
        sensor_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['df']['axis0'][:]]
        # Coleta os timestamps correspondentes a cada registro de velocidade
        timestamps = f['df']['axis1'][:]
        """
        Conversão dos dados extraídos em estrutura de DataFrame.
        1. Verifica se matriz original foi carregada com os sensores na linha
        Caso tenha, aplica a transposição para inverter a matriz.
        2. Criar cabeçalhos de colunas com os nomes dos sensores, facilitando a identificação e manipulação dos dados.
        """
        df_vel = pd.DataFrame(data.T if data.shape[0] == len(sensor_ids) else data, 
                              columns=[f"Sensor_{sid}" for sid in sensor_ids])
        # Insere a coluna de Timestamp como a primeira coluna do DataFrame, convertendo os valores para o formato datetime
        df_vel.insert(0, 'Timestamp', pd.to_datetime(timestamps))
        # Salva o DataFrame de velocidades em formato CSV, sem o índice numérico, facilitando leitura pelos modelos
        df_vel.to_csv(os.path.join(processed_dir, 'METR-LA_series.csv'), index=False)

    print("Passo 2: Processando matrizes para STGCN e DCRNN...")
    # Leitura em binário do arquivo de adjacência, extraindo a matriz de adjacência original (não normalizada)
    with open(os.path.join(raw_dir, 'adj_METR-LA.pkl'), 'rb') as f:
        # O arquivo contém uma tupla com 3 elementos: (sensor_ids, sensor_id_to_index, adj_mx)
        # OBS: latin1 é usada para garantir a compatibilidade com arquivos gerados em Python 2.x
        adj_data = pickle.load(f, encoding='latin1')
        # Como adj_data é uma tupla com 3 elementos, a matriz de adjacência original (não normalizada) é o terceiro elemento (índice 2)
        adj_mx = adj_data[2]

    # FORMATO STGCN
    """
    Conversão da matriz de pesos para precisão simples, 32 bits.
    Substituição de valores NaN por zero
    """
    adj_stgcn = np.nan_to_num(adj_mx.astype(np.float32))
    """
    Procura todas as posições da matriz onde o valor é maior que zero, ou seja, 
    onde existe uma conexão entre os sensores, retornando as coordenadas dessas posições.
    (linhas e colunas)
    """
    rows, cols = np.where(adj_stgcn > 0)
    """
    Cria um tensor de arestas (edge_index) no formato [2, num_arestas], onde a primeira linha 
    contém os índices dos nós de origem e a segunda linha contém os índices dos nós de destino.
    """
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    """
    Extrai apenas os valores de peso das conexões identificadas anteriormente no tensor
    """
    edge_attr = torch.tensor(adj_stgcn[rows, cols], dtype=torch.float)

    """
    Salva os tensores de arestas e pesos no formato .pt, que é o formato nativo do PyTorch.
    """
    torch.save(edge_index, os.path.join(processed_dir, 'edge_index_stgcn.pt'))
    torch.save(edge_attr, os.path.join(processed_dir, 'edge_attr_stgcn.pt'))

    # FORMATO DCRNN (Dual Random Walk)
    """
    A DCRNN não utiliza apenas a vizinhança simples, mas sim um processo de difusão.
    Ela precisa de dois sentidos: Direto (Forward) e Reverso (Backward).
    """
    adj_dcrnn_list = []

    """
    Matriz Forward (Fluxo Direto):
    Representa a direção natural do tráfego. O comando abaixo calcula a matriz de 
    probabilidade de transição (Random Walk) e a converte em um array denso do NumPy.
    """
    adj_dcrnn_list.append(calculate_random_walk_matrix(adj_stgcn).toarray())

    """
    Matriz Backward (Fluxo Reverso):
    Fundamental para capturar o efeito de "represamento" ou influência de montante (upstream).
    Ao transpor a matriz (adj_stgcn.T), invertemos o sentido das arestas para que o modelo
    aprenda como um congestionamento à frente afeta os sensores situados atrás.
    """
    adj_dcrnn_list.append(calculate_random_walk_matrix(adj_stgcn.T).toarray())
    
    """
    Consolidação dos Dados:
    O comando np.stack empilha as duas matrizes (Forward e Backward).
    O resultado é transformado em um tensor do PyTorch com formato [2, Nós, Nós].
    O tipo 'torch.float' (32 bits) é o padrão exigido para o processamento em GPU.
    """
    adj_dcrnn_tensor = torch.tensor(np.stack(adj_dcrnn_list), dtype=torch.float)

    """
    Persistência em Disco:
    Salva o tensor final no diretório de dados tratados com a extensão .pt.
    Isso permite que o modelo carregue a estrutura do grafo instantaneamente durante o treino.
    """
    torch.save(adj_dcrnn_tensor, os.path.join(processed_dir, 'adj_mx_dcrnn.pt'))

    print(f"Sucesso!")
    print(f"STGCN: {edge_index.shape[1]} arestas salvas.")
    print(f"DCRNN: Matriz de difusão {adj_dcrnn_tensor.shape} salva.")

if __name__ == "__main__":
    preparar_tudo()

# dataLoader_Traffic.py
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, window_size=12, horizon=12):
        # Raiz de projeto
        base_path = r'C:\IC\Projeto_V2\data\tratados'
        # Caminho do arquivo final após a aplicação da Engenharia de Atriibutos/Features Engineering
        file_path = os.path.join(base_path, 'METR-LA_featured.csv')
        
        print(f"A carregar base universal para comparação: {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Erro: {file_path} não encontrado. Execute o pré-processamento primeiro!")

        # Carrega o dataset com a coluna de timestamp como índice
        df = pd.read_csv(file_path, index_col=0)
        
        """
        Separação de Velocidades e Atributos Temporais:
        Identifica as colunas de velocidades e as colunas de atributos temmporais (hora e dia da semana)
        Converte tudo em matrizes NumPy do tipo float32, otimizando processamento
        """
        self.time_features_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.speed_cols = [c for c in df.columns if c not in self.time_features_cols]
        v_data = df[self.speed_cols].values.astype(np.float32)
        t_data = df[self.time_features_cols].values.astype(np.float32)

        """
        Normalização Z-Score (Salva para desnormalizar no avaliar_modelo.py):
        Centraliza as velocidades na média 0 com desvio padrão 1
        """
        self.mean = v_data.mean()
        self.std = v_data.std()
        v_norm = (v_data - self.mean) / (self.std + 1e-8)

        # Coleta a quantidade de sensores (nós) da matriz normalizada
        num_nodes = v_norm.shape[1]
        # Expande a dimensão de features para incluir a velocidade normalizada como o primeiro atributo de cada nó
        v_norm = np.expand_dims(v_norm, axis=-1) # [T, N, 1]
        
        # Expandir atributos temporais para cada nó, permitindo que cada sensor tenha acesso às informações de hora e dia da semana, essenciais para a previsão
        t_feats_expanded = np.tile(np.expand_dims(t_data, axis=1), (1, num_nodes, 1)) # [T, N, 4]
        
        # Concatena a Velocidade + 4 Features Temporais, formando um novo tensor com 5 atributos temporais
        self.data = np.concatenate([v_norm, t_feats_expanded], axis=-1).astype(np.float32)
        
        # Define os parâmetros de janela e horizonte para a construção das sequências de entrada e saída do modelo
        self.window_size = window_size
        self.horizon = horizon

        print(f"Processamento concluído: {self.data.shape[0]} timesteps x {num_nodes} sensores.")

    def __len__(self):
        # Garantimos que haja espaço para a janela de entrada E a de saída (horizonte)
        return len(self.data) - self.window_size - self.horizon

    def __getitem__(self, idx):
        # X: [Window_Size, Nodes, Features] -> Entrada do modelo
        x = self.data[idx : idx + self.window_size]
        
        # y: [Horizon, Nodes] -> Alvo (apenas a velocidade, índice 0)
        # Importante: DCRNN precisa da sequência, STGCN pode pegar apenas y[0]
        y = self.data[idx + self.window_size : idx + self.window_size + self.horizon, :, 0]
        
        return torch.tensor(x), torch.tensor(y)

def preparar_loaders(batch_size=32, window_size=12, horizon=12):
    print(f"\nPreparando DataLoaders (Janela: {window_size}, Horizonte: {horizon})...")
    
    dataset = TrafficDataset(window_size=window_size, horizon=horizon)
    
    # Divisão fixa para garantir comparação justa (80% treino, 20% teste)
    train_size = int(0.8 * len(dataset))
    test_ds_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_ds_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"Concluído! Treino: {len(train_loader)} batches | Teste: {len(test_loader)} batches.\n")
    
    return train_loader, test_loader