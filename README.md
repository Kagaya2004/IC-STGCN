🚦 RouteStation

Arquitetura Híbrida Ciberfísica para Otimização de Rotas Logísticas Urbanas

O RouteStation é um sistema de roteamento avançado desenvolvido como projeto de Iniciação Científica. Ele integra Inteligência Artificial Preditiva (Deep Learning) e Otimização Combinatória para gerar trajetos dinâmicos de entregas de última milha (last-mile delivery).

O sistema substitui o uso de matrizes de custos estáticas tradicionais por previsões de tráfego espaço-temporais geradas por uma Rede Neural Convolucional em Grafos (STGCN), validada com dados reais do distrito financeiro de The Loop em Chicago.

🧠 Arquitetura do Sistema

O projeto é estruturado em uma esteira de quatro pilares metodológicos:

Ingestão e Processamento de Dados (Data Pipeline): Processamento de dados reais de GPS da frota de ônibus da Chicago Transit Authority (CTA), lidando com Schema Drift e estruturando tensores tridimensionais históricos.

Modelagem Preditiva (STGCN): Uma Rede Convolucional em Grafos Espaço-Temporais que prevê a velocidade do trânsito nos próximos 60 minutos, alcançando um Erro Absoluto Médio (MAE) de 4,54 km/h, superando métodos estatísticos clássicos (Média Histórica e Média Móvel).

Motor Neuro-Operacional (Roteamento Híbrido): Integração das predições da IA na malha viária extraída do OpenStreetMap (OSMnx) usando mapeamento espacial KDTree. O sequenciamento ótimo das entregas é resolvido como um Problema do Caixeiro Viajante com Rota Aberta (Open TSP) através do Google OR-Tools e algoritmo de Dijkstra.

Interface Gráfica (Frontend): Um painel de controle interativo construído com Streamlit e Folium, permitindo simulações visuais da rota com projeção ortogonal na via (Snap-to-Road).

✨ Principais Funcionalidades

Previsão Espaço-Temporal: A STGCN entende a topologia das ruas, propagando a informação de congestionamento de uma via para suas vizinhas.

Testes de Estresse Logístico: O sistema conta com um modo "Pico Histórico", simulando condições extremas de engarrafamento heterogêneo para validar rotas alternativas.

Roteamento Geométrico Fiel: A interface renderiza rotas acompanhando a curvatura real das ruas, extraindo a propriedade geometry dos nós do OSMnx.

Integração Google Maps: Geração dinâmica de URLs para exportação e espelhamento da rota otimizada diretamente no aplicativo comercial do entregador.

📁 Estrutura do Repositório

📦 RouteStation
 ┣ 📂 data/
 ┃ ┗ 📂 tratados/       # Tensores normalizados, matriz de adjacência e metadados
 ┣ 📂 src/
 ┃ ┣ 📂 preprocessamento/ # Scripts de limpeza, filtragem regional e tensores
 ┃ ┣ 📂 modelo/           # Arquitetura modelo_STGCN.py e pesos treinados (.pth)
 ┃ ┣ 📂 otimizador/       # otimizador_rotas.py (Integração IA + OR-Tools + Dijkstra)
 ┃ ┗ 📂 interface/        # interface_rotas.py (Frontend Streamlit)
 ┗ 📜 README.md


🚀 Como Executar o Projeto

Pré-requisitos

Python 3.9+

Recomenda-se o uso de ambiente virtual (venv ou conda) com suporte a GPU (PyTorch via CUDA), embora o sistema rode em CPU.

1. Clonar o repositório e instalar dependências

git clone https://github.com/seu-usuario/RouteStation.git
cd RouteStation
pip install -r requirements.txt


2. Executar a Interface Web

Certifique-se de que os dados tratados (.npy, .csv) e os pesos do modelo (.pth) estejam nas pastas corretas. Em seguida, inicie o painel do Streamlit:

streamlit run src/interface/interface_rotas.py


3. Operação Básica

Acesse o servidor local (geralmente http://localhost:8501).

Utilize o menu lateral para alternar entre Visão Simplificada (Operacional) e Visão Avançada (Painel Técnico).

Insira endereços pertencentes ao distrito The Loop de Chicago (ex: Willis Tower, Art Institute of Chicago).

Clique em Otimizar Rota (IA) para visualizar o sequenciamento dinâmico.

📊 Resultados e Validação

Nas baterias de testes comparativos, a STGCN mitigou as limitações inerciais da Média Móvel na detecção de gargalos, entregando uma representação topológica muito próxima ao trânsito real. Em testes de rota (ex: cenário de estresse na S State Street), o RouteStation obteve estimativas temporais com variações menores que 5 minutos quando comparadas aos roteadores proprietários de crowdsourcing (como o Google Maps), consolidando sua eficácia sem depender de APIs pagas de trânsito em tempo real.

Projeto desenvolvido como pesquisa de Iniciação Científica - Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP).
