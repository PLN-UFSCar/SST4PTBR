import pandas as pd
import wn
import sys
import torch
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import string
import spacy

wn.add("../lexico/own-pt.tar.gz")

def pre_processamento(frase):
  frase = frase.lower()
  frase = frase.translate(str.maketrans('', '', string.punctuation))
  return frase

class WSD:
    def __init__(self):
        self.extractor = pipeline(
            "feature-extraction",
            model="neuralmind/bert-base-portuguese-cased",
            tokenizer="neuralmind/bert-base-portuguese-cased",
            framework="pt"
        )

    def gerarEmbeddingSentido(self, sentido):
        #Combinando token CLS e a media dos embeddings do sentido
        embs = self.extractor(sentido)[0]
        embs = np.array(embs)

        embscls = embs[0]
        embssentido = np.mean(embs[1:-1], axis=0)

        return  (0.8 * embscls + 0.2 * embssentido)
        #Usando so token CLS
        #  embs = self.extractor(sentido)[0]
        #  return embs[0]

    def gerarEmbeddingsPalavra(self, contexto, palavra):
        tkscontexto = self.extractor.tokenizer.tokenize(contexto)
        tkspalavra = self.extractor.tokenizer.tokenize(palavra)

        indicespalavra = self.acharPalavraContexto(tkscontexto, tkspalavra)

        if not indicespalavra:
            print(f"Palavra '{palavra}' não encontrada no contexto")
            return None

        embscontexto = self.extractor(contexto)[0]

        indicespalavrasajustado = [i + 1 for i in indicespalavra]

        embspalavra = [embscontexto[i] for i in indicespalavrasajustado]

        return np.mean(embspalavra, axis=0)

    def acharPalavraContexto(self, tkscontexto, tkspalavra):
        for i in range(len(tkscontexto) - len(tkspalavra) + 1):
            if tkscontexto[i:i+len(tkspalavra)] == tkspalavra:
                return list(range(i, i + len(tkspalavra)))
        return []

    def compararPalavraSentido(self, contexto, palavra, sentido):
        embspalavra = self.gerarEmbeddingsPalavra(contexto, palavra)
        embssentido = self.gerarEmbeddingSentido(sentido)
        if embspalavra is None:
            return 0.0
        return cosine_similarity([embspalavra], [embssentido])[0][0]

class OWNPT:
    def getSenses(self, word):
        senses = wn.senses(word)
        sensesstr = []
        for sense in senses:
            definicao = sense.synset().definition()
            if definicao:
                sensesstr.append(definicao)
        return sensesstr

def selecionarSentido(contexto, palavra, wsd, ownpt):
    doc = nlp(palavra)
    palavra_aux = doc[0].lemma_
    # palavra = pre_processamento(palavra)
    sentidos = ownpt.getSenses(palavra_aux)

    # Para ser ambígua, uma palavra precisa ter dois ou mais sentidos
    if (sentidos == None or (sentidos != None and len(sentidos) <= 1)):
        return None

    maior_sim = -1
    melhor_sentido = None

    for sentido in sentidos:
        sim = wsd.compararPalavraSentido(contexto, palavra, pre_processamento(sentido))
        if sim is not None and sim > maior_sim:
            maior_sim = sim
            melhor_sentido = sentido

    return melhor_sentido

if __name__ == '__main__':
    nlp = spacy.load("pt_core_news_sm")
    contexto = str(sys.argv[1])
    palavra = str(sys.argv[2])
    
    wsd = WSD()
    ownpt = OWNPT()

    melhor_sentido = selecionarSentido(contexto, palavra, wsd, ownpt)
    print(melhor_sentido)
        
