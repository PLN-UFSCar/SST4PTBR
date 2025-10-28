import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import spacy
import re
import pandas as pd
import sys

def extract_title_from_url(url: str) -> str:
    """
    Extracts the 'slug' from the URL (final part) and converts to readable title, replacing '-' with space.
    """
    # Remove all '/' slashes at the end of the string
    url = url.rstrip('/')
    # Split the url
    url = url.split('/')
    slug = url[-1].replace('-', ' ')

    return(slug)

def clean_text(text, keep_punctuation=False) -> str:
    """
    Cleans a text by removing non-textual characters, with option to keep punctuation.
    If `keep_punctuation` is True, keeps common punctuations like !, ?, ,, ., :, ;.

    Parameters:
        text (str): Text to be processed.
        keep_punctuation (bool): If True, keeps punctuations like !, ?, ., ,, :, ;. Default is False.

    Returns:
        str: Clean text, with or without punctuation, depending on the parameter.
    """
    if not isinstance(text, str):
        return text
    if keep_punctuation:
        # Keep punctuation marks (like !, ?, ,, ., etc.)
        return re.sub(r'[^\w\s\.\,\!\?\:\;]', '', text)
    else:
        # Remove everything that is not a letter (a-z) or whitespace (\s)        
        return re.sub(r'[^\w\s]', '', text)

def tokenize_text(text):
    """
    Tokenizes a string into words, using NLTK.
    """
    try:
        # To use word_tokenize(text) from NLTK
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Error trying to download tokenizer 'punkt': {e}")

    if not isinstance(text, str):
        return text
    return word_tokenize(text)

def remove_stopwords(tokens):
    try:
        nltk.download('stopwords', quiet=True)
        stopwords_pt = set(stopwords.words('portuguese'))
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        stopwords_pt = set()  # Fallback: empty set

    if not isinstance(tokens, list):
        return tokens
    return [t for t in tokens if t not in stopwords_pt]

def apply_stemming(tokens):
    try:
        # Download necessary data for Portuguese stemmer to work
        nltk.download('rslp', quiet=True)
    except Exception as e:
        print(f"Error downloading RSLP stemmer: {e}")
    
    stemmer = RSLPStemmer()
    return [stemmer.stem(t) for t in tokens]

def apply_lemmatization(tokens, nlp):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]    
    
def preprocessing(df, 
                  keep_punctuation: bool = False, 
                  tokenize_text: bool = True,
                  use_stemming: bool = False,
                  use_lemmatization: bool = False):
    # Convert strings to lowercase letters 
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    # Remove article_link (same information as 'headline' attribute)
    df.drop('article_link', axis=1, inplace=True)

    # Remove everything that is not a letter (a-z) or whitespace (\s) 
    df = df.map(lambda x: clean_text(x, keep_punctuation))

    # Remove numbers
    df = df.map(lambda x: re.sub(r'\d+', '', x) if isinstance(x, str) else x)

    # Tokenize text
    if tokenize_text:
        df = df.map(lambda x: tokenize_text(x))

    # Remove stop words (Portuguese)
    df = df.map(lambda x: remove_stopwords(x) if isinstance(x, list) else x)

    # Apply stemming or lemmatization
    if use_lemmatization or (use_stemming and use_lemmatization):
        nlp = spacy.load("pt_core_news_sm")  # lightweight model for Portuguese
        df = df.map(lambda x: apply_lemmatization(x, nlp) if isinstance(x, list) else x)

    if use_stemming == True and use_lemmatization == False:
        df = df.map(lambda x: apply_stemming(x) if isinstance(x, list) else x)

    # Transform label from True and False to 1 and 0
    df["is_sarcastic"] = df["is_sarcastic"].astype(int)

    # display(df)
    return df


def preprocess_sentence(sentence: str, 
                       keep_punctuation: bool = False, 
                       tokenize_text: bool = True,
                       use_stemming: bool = False,
                       use_lemmatization: bool = False):
    """
    Preprocesses a sentence according to the provided options.
    """
    sentence = sentence.lower() if isinstance(sentence, str) else sentence

    sentence = clean_text(sentence, keep_punctuation)

    sentence = re.sub(r'\d+', '', sentence) if isinstance(sentence, str) else sentence

    if tokenize_text:
        sentence = tokenize_text(sentence)

    sentence = remove_stopwords(sentence) if isinstance(sentence, list) else sentence

    if use_lemmatization or (use_stemming and use_lemmatization):
        nlp = spacy.load("pt_core_news_sm")
        sentence = apply_lemmatization(sentence, nlp) if isinstance(sentence, list) else sentence

    if use_stemming and not use_lemmatization:
        sentence = apply_stemming(sentence) if isinstance(sentence, list) else sentence

    return sentence

if __name__ == "__main__":
    preprocess_sentence(
        sys.argv[1],  # Sentence to be processed
    )