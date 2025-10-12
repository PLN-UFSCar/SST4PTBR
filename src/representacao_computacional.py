# Word2Vec
from gensim.models import Word2Vec
import numpy as np

def aplica_word2vec(df, nome_coluna, estrategia='skip-gram'):
    """
    Treina um modelo Word2Vec a partir de uma coluna com listas de tokens.

    Parâmetros:
    - df: DataFrame contendo a coluna de texto tokenizado.
    - coluna: nome da coluna com as listas de tokens.
    - sg: estratégia

    Retorno:
    - modelo Word2Vec treinado.
    """
    # Verifica se a coluna existe no DataFrame
    if nome_coluna not in df.columns:
        raise ValueError(f"A coluna {nome_coluna} não foi encontrada no DataFrame.")

    # Extrai os textos tokenizados (precisa ser uma lista de listas)
    frases = df[nome_coluna].tolist()
    
    # Parâmetros do modelo
    tamanho_vetor = 100
    janela = 5          # Tamanho da janela de contexto para o treinamento
    min_count = 1       # Ignora palavras raras (frequência menor que o valor especificado)
    epocas = 10         # Quantidade de vezes que vai tentar otimizar o modelo
    num_threads = 4     # Número de threads para acelerar o treinamento
    sg = 1 if estrategia == 'skip-gram' else 0

    # Treina o modelo Word2Vec
    modelo = Word2Vec(
        sentences=frases,
        vector_size=tamanho_vetor,
        window=janela,
        min_count=min_count,
        sg=sg,
        epochs=epocas,
        workers=num_threads
    )

    # Função para calcular o vetor médio de um texto
    def vetor_medio(palavras, modelo, tamanho_vetor):
        vetores = [modelo.wv[palavra] for palavra in palavras if palavra in modelo.wv]
        if len(vetores) == 0:
            return np.zeros(tamanho_vetor)
        # Combina vetores para representar o texto como um todo
        return np.mean(vetores, axis=0)    
  
    # Para cada texto, calcula a média dos vetores das palavras no texto 
    embeddings = [vetor_medio(texto, modelo, tamanho_vetor) for texto in frases]
    
    return modelo, embeddings

