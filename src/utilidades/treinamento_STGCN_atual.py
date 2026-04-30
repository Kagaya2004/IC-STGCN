# Treinamento STGCN
import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def treinar_stgcn():
    # Definição do dispositivo para processamento (GPU ou CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Raiz do projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Diretório dos dados tratados
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    # Diretório dos logs
    log_dir = os.path.join(base_dir, 'data', 'logs')
    # Diretório dos modelos
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    # Cria as pastas de logs e modelos, caso não existam
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Identificação única do experimento para histórico dos logs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_stgcn_{timestamp}.txt")
    
    # Função auxiliar para imprimir mensagens no console e salvar no arquivo de log
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # Definição dos hiperparâmetros de treinamento
    """
    Batch size é o número de amostras que o modelo irá processar antes de atualizar 
    os pesos. Ao invés de processar todos os registros, o modelo processa uma pequana
    amostra, no caso, 32 amostras por vez.
    """
    batch_size = 32
    """
    Época se refere a passagem completa de todo conjunto de dados do treinamento.
    """
    epochs = 50
    """
    Taxa de aprendizado define o tamanho dos passos que o modelo dá ao atualizar os pesos.
    """
    lr = 0.001
    
    """
    Preparação dos dados:
    O modelo STGCN irá processar os dados em uma janela de 60 minutos (12 passos de 5 minutos)
    para prever os próximos 60 minutos (12 passos).
    """
    train_loader, test_loader = preparar_loaders(batch_size=batch_size, window_size=12, horizon=12)
    
    # Instanciação da ST-GCN
    model = STGCN_TrafficModel(num_features=5).to(device)
    
    # Carregamento do Grafo (Agora usando os pesos NORMALIZADOS SIMETRICAMENTE)
    try:
        edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
        edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)
        log_print(f"Grafo carregado com {edge_index.shape[1]} arestas normalizadas.")
    except FileNotFoundError:
        log_print("Erro: Arquivos de grafo não encontrados. Rode preparar_grafo.py primeiro!")
        return

    # Otimização e Critério de Perda
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # MAE mascarado para ignorar sensores falhos (velocidade = 0)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"[STGCN] Iniciando Treinamento")
    log_print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    best_val_mae = float('inf')
    history = []
    start_train = time.time()

    # Loop de Treinamento
    for epoch in range(epochs):
        # Inicia o modo de treinamento
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            # Move os dados para o dispositivo (GPU ou CPU)
            x, y = x.to(device), y.to(device) # y shape: [Batch, 12, Nodes]
            
            # Limpa os gradientes acumulados na memória
            optimizer.zero_grad()
            """
            Realize Forward Pass: Recebe histórico e projeta simultaneamente os 12 
            passos (60 minutos) para todos os sensores do grafo.
            """
            output = model(x, edge_index, edge_weight=edge_attr) 
            
            # Máscara para identificar dados válidos
            mask = (y > 0).float()
            # Aplica o erro apenas onde o valor da máscara é igual a 1 (sensores válidos)
            loss = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            
            """
            Realiza o Backpropagation: Retornando o erro para calcular quanto cada
            neurônio contribuiu no erro final.
            """
            loss.backward()
            # Atualiza os pesos do modelo com base no erro calculado
            optimizer.step()
            # Acumula a perda total para o cálculo da média no final da época
            train_loss += loss.item()

        # Validação (Conjunto de Teste)
        # Inicia o modo de avaliação
        model.eval()
        # Valor do MAE inicial
        val_mae = 0
        # Desativa o cálculo de gradientes para economizar memória e acelerar a inferência
        with torch.no_grad():
            for xv, yv in test_loader:
                # Move os dados de validação para o dispositivo
                xv, yv = xv.to(device), yv.to(device)
                # Realiza a predição usando o modelo treinado, passando o grafo como entrada
                pred = model(xv, edge_index, edge_weight=edge_attr)
                # Máscara para identificar os dados válidos
                mask_v = (yv > 0).float()
                # Cálculo do MAE usando a máscara para considerar apenas os sensores válidos
                v_mae = (mae_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                # Acumula o MAE de validação para calcular a média no final da época
                val_mae += v_mae.item()

        # Cálculo da média do MAE para treinamento e validação
        avg_train_mae = train_loss / len(train_loader)
        # Cálculo da média do MAE de validação
        avg_val_mae = val_mae / len(test_loader)
        
        # Ajuste dinâmico da taxa de aprendizado
        scheduler.step(avg_val_mae)
        
        # Salva o modelo com melhor MAE de validação
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(model_dir, 'stgcn_model.pth'))
            status = "Melhor Modelo!"
        else:
            status = ""

        log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} {status}")
        history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

    # Finalização e Salvamento do Histórico para Gráficos
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_train)))
    log_print(f"\nTreino ST-GCN Finalizado em {total_time}")
    
    # Nome do CSV compatível com os scripts de avaliação automatizada
    history_path = os.path.join(log_dir, f"history_stgcn_{timestamp}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    log_print(f"Histórico salvo em: {history_path}")

if __name__ == "__main__":
    treinar_stgcn()