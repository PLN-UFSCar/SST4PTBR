import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from hyphen import Hyphenator
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
nltk.download('punkt')

class Avaliacao:
    def __init__(self):
        self.hifen = Hyphenator('pt_BR')
        self.extractor = pipeline(
            "feature-extraction",
            model="neuralmind/bert-base-portuguese-cased",
            tokenizer="neuralmind/bert-base-portuguese-cased",
            framework="pt"
        )
        
    def contarSilabas(self, palavra):
        silabas = self.hifen.syllables(palavra.lower())
        if(silabas):
            return len(silabas)
        else:
            return 1
    
    def flesch(self, texto):
        frases = sent_tokenize(texto, language='portuguese')
        palavras = []
        totalsilabas = 0

        for palavra in word_tokenize(texto, language='portuguese'):
            if(palavra.isalpha()):
                palavras.append(palavra)
                totalsilabas += self.contarSilabas(palavra)

        totalfrases = len(frases)
        totalpalavras = len(palavras)

        if(totalfrases == 0 or totalpalavras == 0):
            return 0.0
        
        mediapalavrasfrase = totalpalavras / totalfrases
        mediasilabaspalavra = totalsilabas / totalpalavras

        flesch = 248.835 - 1.015 * mediapalavrasfrase - 84.6 * mediasilabaspalavra 
        return round(flesch, 2)

    def gerarEmbeddings(self, texto):
        embs = self.extractor(texto)[0]
        return np.mean(embs, axis=0)

    def verificarManteveSentido(self, textoinicial, textofinal):

        embsTextoInicial = self.gerarEmbeddings(textoinicial)
        embsTextoFinal = self.gerarEmbeddings(textofinal)

        similaridade = cosine_similarity([embsTextoInicial], [embsTextoFinal])[0][0]

        if(similaridade >= 0.8):
            return (True, similaridade)
        return (False, similaridade)
    
    def avaliarReescrita(self, textooriginal, textoreescrito):
        fleschoriginal = self.flesch(textooriginal)
        fleschreescrito = self.flesch(textoreescrito)
        mantevesentido, similaridade = self.verificarManteveSentido(textooriginal, textoreescrito)
        melhorialegibilidade = fleschreescrito - fleschoriginal
        
        return {
            'Flesch texto original': fleschoriginal,
            'Flesch texto reescrito': fleschreescrito,
            'Melhora na legibilidade (ou piora)': round(melhorialegibilidade, 2),
            'Manteve o sentido?': mantevesentido,
            'Similaridade': similaridade
        }