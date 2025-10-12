from gensim.models import Word2Vec
import joblib
import pickle
import numpy as np
import re
import subprocess
# from nltk.tokenize import word_tokenize
import pandas as pd


def vetor_medio(tokens, modelo, tamanho_vetor=100):
    vetores = [modelo.wv[t] for t in tokens if t in modelo.wv]
    if not vetores:
        return np.zeros(tamanho_vetor)
    return np.mean(vetores, axis=0)


# Carrega modelo Word2Vec
w2v_model = Word2Vec.load("modelo_word2vec.model")

# Carrega classificador treinado (SVM, Random Forest etc.)
clf = joblib.load("modelo_word2vec.pkl")

while True:

    frase = input("Digite uma frase para análise: ")
    # tokens = pre_processamento_frase(frase)
    subprocess.run(["conda", "run", "-n", "ironia", "python", "scripts/preprocessamento.py", frase, "False", "True", "False", "False"], check=True)
    tokens = pd.read_csv("frase_processada.csv")["tokens"].tolist()[0]
    vetor = vetor_medio(tokens, w2v_model)

    # Reshape porque o modelo espera 2D: (1, tamanho_vetor)
    vetor = vetor.reshape(1, -1)

    # Previsão
    pred = clf.predict(vetor)
    prob = clf.predict_proba(vetor)[0]

    if pred[0] == 1:
        print(f"Ironia detectada (confiança: {prob[1]:.2f})")
    else:
        print(f"Sem ironia detectada (confiança: {prob[0]:.2f})")