import wn
import sys
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import string
import spacy

wn.add("../lexicon/own-pt.tar.gz")

def pre_processing(sentence):
  sentence = sentence.lower()
  sentence = sentence.translate(str.maketrans('', '', string.punctuation))
  return sentence

class WSD:
    def __init__(self):
        self.extractor = pipeline(
            "feature-extraction",
            model="neuralmind/bert-base-portuguese-cased",
            tokenizer="neuralmind/bert-base-portuguese-cased",
            framework="pt"
        )

    def generate_sense_embedding(self, sense):
        # Combining CLS token and the average of sense embeddings
        embs = self.extractor(sense)[0]
        embs = np.array(embs)

        embs_cls = embs[0]
        embs_sense = np.mean(embs[1:-1], axis=0)

        return (0.8 * embs_cls + 0.2 * embs_sense)

    def generate_word_embeddings(self, context, word):
        tks_context = self.extractor.tokenizer.tokenize(context)
        tks_word = self.extractor.tokenizer.tokenize(word)
        word_indexes = self.find_word_in_context(tks_context, tks_word)

        if not word_indexes:
            print(f"Word '{word}' not found in context")
            return None

        embs_context = self.extractor(context)[0]
        adjusted_word_indexes = [i + 1 for i in word_indexes]
        embs_word = [embs_context[i] for i in adjusted_word_indexes]

        return np.mean(embs_word, axis=0)

    def find_word_in_context(self, tks_context, tks_word):
        for i in range(len(tks_context) - len(tks_word) + 1):
            if tks_context[i:i+len(tks_word)] == tks_word:
                return list(range(i, i + len(tks_word)))
        return []

    def compare_word_sense(self, context, word, sense):
        embs_word = self.generate_word_embeddings(context, word)
        embs_sense = self.generate_sense_embedding(sense)
        if embs_word is None:
            return 0.0
        return cosine_similarity([embs_word], [embs_sense])[0][0]

class OWNPT:
    def get_senses(self, word):
        senses = wn.senses(word)
        definitions = []
        for sense in senses:
            definition = sense.synset().definition()
            if definition:
                definitions.append(definition)
        return definitions

def select_sense(context, word, wsd, ownpt):
    doc = nlp(word)
    aux_word = doc[0].lemma_
    senses = ownpt.get_senses(aux_word)

    # To be ambiguous, a word needs to have two or more senses
    if (senses == None or (senses != None and len(senses) <= 1)):
        return None

    highest_sim = -1
    best_sense = None

    for sense in senses:
        sim = wsd.compare_word_sense(context, word, pre_processing(sense))
        if sim is not None and sim > highest_sim:
            highest_sim = sim
            best_sense = sense

    return best_sense

if __name__ == '__main__':
    nlp = spacy.load("pt_core_news_sm")
    context = str(sys.argv[1])
    word = str(sys.argv[2])
    
    wsd = WSD()
    ownpt = OWNPT()

    best_sense = select_sense(context, word, wsd, ownpt)
    print(best_sense)