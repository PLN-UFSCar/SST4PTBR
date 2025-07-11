import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import spacy
import re
import pandas as pd
import sys

def extrai_titulo_url(url: str) -> str:
    """
    Extrai o 'slug' da URL (parte final) e converte em título legível, substituindo '-' por espaço.
    """
    # Remover todas as barras '/' no final da string
    url = url.rstrip('/')
    # Divide a url
    url = url.split('/')
    slug = url[-1].replace('-', ' ')

    return(slug)

def limpa_texto(texto, manter_pontuacao=False) -> str:
    """
    Limpa um texto removendo caracteres não textuais, com opção de manter pontuação.
    Caso `manter_pontuacao` seja True, mantém pontuações comuns como !, ?, ,, ., :, ;.

    Parâmetros:
        texto (str): Texto a ser processado.
        manter_pontuacao (bool): Se True, mantém pontuações como !, ?, ., ,, :, ;. Default é False.

    Retorno:
        str: Texto limpo, com ou sem pontuação, dependendo do parâmetro.
    """
    if not isinstance(texto, str):
        return texto
    if manter_pontuacao:
        # Mantém sinais de pontuação (como !, ?, ,, ., etc.)
        return re.sub(r'[^\w\s\.\,\!\?\:\;]', '', texto)
    else:
        # Remove tudo que não for letra (a-z) ou espaço em branco (\s)        
        return re.sub(r'[^\w\s]', '', texto)

def tokeniza_texto(texto):
    """
    Tokeniza uma string em palavras, usando NLTK.
    """
    try:
        # Para usar word_tokenize(texto) do NLTK
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Erro ao tentar baixar o tokenizer 'punkt': {e}")

    if not isinstance(texto, str):
        return texto
    return word_tokenize(texto)

def remove_stopwords(tokens):
    try:
        nltk.download('stopwords', quiet=True)
        stopwords_pt = set(stopwords.words('portuguese'))
    except Exception as e:
        print(f"Erro ao carregar as stopwords: {e}")
        stopwords_pt = set()  # Fallback: conjunto vazio

    if not isinstance(tokens, list):
        return tokens
    return [t for t in tokens if t not in stopwords_pt]

def aplica_stemming(tokens):
    try:
        # Baixa os dados necessários para o stemmer português funcionar
        nltk.download('rslp', quiet=True)
    except Exception as e:
        print(f"Erro ao baixar o stemmer RSLP: {e}")
    
    stemmer = RSLPStemmer()
    return [stemmer.stem(t) for t in tokens]

def aplica_lemmatization(tokens, nlp):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]    
    
def pre_processamento(df, 
                      manter_pontuacao: bool = False, 
                      tokenizar_texto: bool = True,
                      usar_stemming: bool = False,
                      usar_lemmatization: bool = False):
    # Converte strings para letras minúsculas 
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    # Normaliza a URL: Extrai o título da URL
    # df['article_link'] = df['article_link'].map(extrai_titulo_url)

    # Remove o article_link (mesma informação do que o atributo 'headline')
    df.drop('article_link', axis=1, inplace=True)

    # Remove tudo que não for letra (a-z) ou espaço em branco (\s) 
    df = df.map(lambda x: limpa_texto(x, manter_pontuacao))

    # Remove números
    df = df.map(lambda x: re.sub(r'\d+', '', x) if isinstance(x, str) else x)

    # Tokeniza o texto
    if tokenizar_texto:
        df = df.map(lambda x: tokeniza_texto(x))

    # Remove stop words (português)
    df = df.map(lambda x: remove_stopwords(x) if isinstance(x, list) else x)

    # Aplica o stemming ou lemmatization
    if usar_lemmatization or (usar_stemming and usar_lemmatization):
        nlp = spacy.load("pt_core_news_sm")  # modelo leve para português
        df = df.map(lambda x: aplica_lemmatization(x, nlp) if isinstance(x, list) else x)

    if usar_stemming == True and usar_lemmatization == False:
        df = df.map(lambda x: aplica_stemming(x) if isinstance(x, list) else x)

    # Transforma o label de True e False para 1 e 0
    df["is_sarcastic"] = df["is_sarcastic"].astype(int)

    # display(df)
    return df


def pre_processamento_frase(frase: str, 
                          manter_pontuacao: bool = False, 
                          tokenizar_texto: bool = True,
                          usar_stemming: bool = False,
                          usar_lemmatization: bool = False):
    """
    Pré-processa uma frase de acordo com as opções fornecidas.
    """
    frase = frase.lower() if isinstance(frase, str) else frase

    frase = limpa_texto(frase, manter_pontuacao)

    frase = re.sub(r'\d+', '', frase) if isinstance(frase, str) else frase

    if tokenizar_texto:
        frase = tokeniza_texto(frase)

    frase = remove_stopwords(frase) if isinstance(frase, list) else frase

    if usar_lemmatization or (usar_stemming and usar_lemmatization):
        nlp = spacy.load("pt_core_news_sm")
        frase = aplica_lemmatization(frase, nlp) if isinstance(frase, list) else frase

    if usar_stemming and not usar_lemmatization:
        frase = aplica_stemming(frase) if isinstance(frase, list) else frase

    return frase


if __name__ == "__main__":

    

    pre_processamento_frase(
        sys.argv[1],  # Frase a ser processada
    )