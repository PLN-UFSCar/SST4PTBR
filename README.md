# Detecção de sarcasmos e ambiguidades textuais com enriquecimento de prompts para transformação linguística

A tarefa de PLN proposta tem como objetivo a detecção e eliminação de ironias e ambiguidades lexicais em textos, visando torná-los mais claros e objetivos.

Para atingir o objetivo, são realizadas três tarefas: a primeira de detecção de frases possivelmente irônicas no texto. Em seguida, detectando o sentido de palavras possivelmente ambíguas no texto. Por fim, utiliza-se as informações das etapas anteriores para enriquecer um prompt para reescrever o texto.

## Análise quantitativa

No arquivo `analise.xlsx` está disponível a planilha com a análise quantitativa feita pelos membros do grupo.

## Pré-requisitos

> [!IMPORTANT]
> Tenha o docker e o docker compose instalados 

- Garanta que você tenha o docker e o docker compose instalados em sua máquina. [Acesse aqui a documentação do docker](https://www.docker.com/).

## Passo a passo

### Faça o clone do projeto
```
git clone https://github.com/PLN-UFSCar/PLN-SO-2025-01-Proj-G03.git
```

### Entre no diretório do projeto
```
cd PLN-SO-2025-01-Proj-G03
```

### Clone o Dataset
Clone o repositório [PLNCrawler](https://github.com/schuberty/PLNCrawler/tree/master) que contem o dataset utilizado.
```
git clone https://github.com/schuberty/PLNCrawler.git
```

### Faça o download do lexico disponível em [OpenWordnet-PT](https://github.com/own-pt/openWordnet-PT/releases), cujo nome é `own-pt.tar.gz`, e o coloque dentro da pasta `lexico` do projeto.

### Execute o projeto usando o docker
```
docker compose up -d --build
```

### Acesse o [notebook principal](http://localhost:8888/lab/tree/notebooks/main.ipynb).
Se for a sua primeira vez executando o código, coloque a variável carregar_modelo como False na sessão "Segunda abordagem para detecção: Fine-tuning de um modelo Sentence Transformer". Dessa forma, o modelo é treinado do zero.
```python
carregar_modelo = False
```

Além disso, você pode escolher a sua estratégia (word2vec ou sequence transformers). Para isso, mude as variáveis da célula conforme a sua necessidade.
```python
usar_word2vec = False
usar_sequence_transformer = not usar_word2vec
```

Você também deverá colocar sua chave da API do Google Gemini na variável `API_KEY`.
```python
API_KEY = 'sua_chave_aqui'
```
