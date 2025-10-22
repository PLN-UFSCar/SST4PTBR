import sys
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import pickle
import time

# Geração dos embeddings médios
def vetor_medio(tokens, modelo, tamanho_vetor=100):
    vetores = [modelo.wv[t] for t in tokens if t in modelo.wv]
    if not vetores:
        return np.zeros(tamanho_vetor)
    return np.mean(vetores, axis=0)

def aplica_word2vec(df, nome_coluna, estrategia='skip-gram'):
    frases = []
    indices_validos = []

    # Filtra frases não nulas e converte arrays para listas
    for idx, tokens in enumerate(df[nome_coluna]):
        if tokens is not None and isinstance(tokens, (list, np.ndarray)) and len(tokens) > 0:
            if isinstance(tokens, np.ndarray):
                frases.append(tokens.tolist())
            else:
                frases.append(tokens)
            indices_validos.append(idx)

    if not frases:
        raise ValueError(f"Nenhuma frase válida encontrada na coluna '{nome_coluna}'.")

    # Parâmetros do modelo Word2Vec
    tamanho_vetor = 100
    janela       = 5
    min_count    = 1
    epocas       = 10
    num_threads  = 4
    sg           = 1 if estrategia == 'skip-gram' else 0

    modelo = Word2Vec(
        sentences=frases,
        vector_size=tamanho_vetor,
        window=janela,
        min_count=min_count,
        sg=sg,
        epochs=epocas,
        workers=num_threads
    )


    embeddings = [vetor_medio(tokens, modelo, tamanho_vetor) for tokens in frases]

    return modelo, embeddings, indices_validos

if __name__ == '__main__':
    action = int(sys.argv[1]) # 0 para treinar, 1 para prever
    if action == 0:
        caminho = sys.argv[2]        # Ex: "corpus.parquet"
        coluna = sys.argv[3]          # Ex: "tokens"
        estrategia = sys.argv[4]      # Ex: "skip-gram" ou "cbow"

        df = pd.read_parquet(caminho, engine="pyarrow")

        print(f"[INFO] Iniciando treino Word2Vec com {len(df)} registros...")
        inicio = time.perf_counter()

        modelo, embeddings, indices_validos = aplica_word2vec(df, coluna, estrategia)

        modelo.save("../modelos/modelo_word2vec.model")

        fim = time.perf_counter()
        print(f"[INFO] Treinamento concluído em {fim - inicio:.2f} segundos.")

        # Salvar embeddings
        with open("../temp/embeddings_output.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        # Salvar índices válidos
        with open("../temp/indices_validos.pkl", "wb") as f:
            pickle.dump(indices_validos, f)

        print(f"[INFO] Embeddings e índices salvos com sucesso.")
    
    elif action == 1:
        # ler os tokens da frase de um arquivo CSV
        tokens = pd.read_csv("../temp/frase_processada.csv")["tokens"].tolist()[0]
        print(f"[INFO] Tokens lidos: {tokens}")

        # Carregar o modelo Word2Vec
        w2v_model = Word2Vec.load("../modelos/modelo_word2vec.model")

        vetor = vetor_medio(tokens, w2v_model)
        print(f"[INFO] Vetor médio calculado: {vetor}")

        # Reshape porque o modelo espera 2D: (1, tamanho_vetor)
        vetor = vetor.reshape(1, -1)
        print(f"[INFO] Vetor reshaped: {vetor.shape}")

        # Salvar o vetor em um arquivo CSV
        pd.DataFrame(vetor).to_csv("../temp/vetor_word2vec.csv", index=False, header=False)

        # Ler o vetor salvo no arquivo CSV
        vetor_carregado = pd.read_csv("../temp/vetor_word2vec.csv", header=None).values
        print(f"[INFO] Vetor carregado do CSV: {vetor_carregado}")
        
