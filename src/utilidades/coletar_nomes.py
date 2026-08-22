import osmnx as ox
from pathlib import Path

def auditar_nomes_ruas():
    # 1. Navegação segura: "Sobe" dois níveis saindo de src/utilidades para a raiz C:\IC\Aplication
    base_dir = Path(__file__).resolve().parents[2]
    caminho_mapa = base_dir / "data" / "tratados" / "chicago_map.graphml"
    
    if not caminho_mapa.exists():
        print(f"Erro Crítico: Mapa não encontrado no caminho resolvido:\n{caminho_mapa}")
        return

    print("-> Carregando o grafo de Chicago do disco (aguarde)...")
    G_osm = ox.load_graphml(caminho_mapa)
    
    print("\n--- AMOSTRA DE 15 RUAS DO OSMnx ---")
    nomes_vistos = set()
    contador = 0
    
    # 2. Varredura das arestas do grafo
    for u, v, k, data in G_osm.edges(keys=True, data=True):
        nome_rua = data.get('name')
        
        # Ignora arestas sem nome (vias de serviço/becos)
        if nome_rua and nome_rua not in nomes_vistos:
            nomes_vistos.add(str(nome_rua))
            # Mostra o tipo de dado e o conteúdo exato
            print(f"Tipo: {type(nome_rua).__name__} | Conteúdo: {nome_rua}")
            contador += 1
            
        if contador >= 15:
            break
            
    print("-" * 35)

if __name__ == "__main__":
    auditar_nomes_ruas()