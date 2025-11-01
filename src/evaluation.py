import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from hyphen import Hyphenator
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
nltk.download('punkt_tab')

class Evaluation:
    def __init__(self):
        self.hyphen = Hyphenator('pt_BR')
        self.extractor = pipeline(
            "feature-extraction",
            model="neuralmind/bert-base-portuguese-cased",
            tokenizer="neuralmind/bert-base-portuguese-cased",
            framework="pt"
        )
        
    def count_syllables(self, word):
        syllables = self.hyphen.syllables(word.lower())
        if(syllables):
            return len(syllables)
        else:
            return 1
    
    def flesch(self, text):
        sentences = sent_tokenize(text, language='portuguese')
        words = []
        total_syllables = 0

        for word in word_tokenize(text, language='portuguese'):
            if(word.isalpha()):
                words.append(word)
                total_syllables += self.count_syllables(word)

        total_sentences = len(sentences)
        total_words = len(words)

        if(total_sentences == 0 or total_words == 0):
            return 0.0
        
        average_words_sentence = total_words / total_sentences
        average_syllables_word = total_syllables / total_words

        flesch = 248.835 - 1.015 * average_words_sentence - 84.6 * average_syllables_word 
        return round(flesch, 2)

    def generate_embeddings(self, text):
        embs = self.extractor(text)[0]
        return np.mean(embs, axis=0)

    def check_maintained_meaning(self, initial_text, final_text):
        embs_initial_text = self.generate_embeddings(initial_text)
        embs_final_text = self.generate_embeddings(final_text)

        similarity = cosine_similarity([embs_initial_text], [embs_final_text])[0][0]

        if(similarity >= 0.8):
            return (True, similarity)
        return (False, similarity)
    
    def evaluate_rewrite(self, original_text, rewritten_text):
        flesch_original = self.flesch(original_text)
        flesch_rewritten = self.flesch(rewritten_text)
        maintained_meaning, similarity = self.check_maintained_meaning(original_text, rewritten_text)
        readability_improvement = flesch_rewritten - flesch_original
        
        return {
            'Flesch original text': flesch_original,
            'Flesch rewritten text': flesch_rewritten,
            'Readability improvement (or worsening)': round(readability_improvement, 2),
            'Maintained meaning?': maintained_meaning,
            'Similarity': similarity
        }