# Detecção de sarcasmos e ambiguidades textuais com enriquecimento de prompts para transformação linguística

A tarefa de PLN proposta tem como objetivo a detecção e eliminação de ironias e ambiguidades lexicais em textos, visando torná-los mais claros e objetivos.

Para atingir o objetivo, são realizadas três tarefas: a primeira de detecção de frases possivelmente irônicas no texto. Em seguida, detectando o sentido de palavras possivelmente ambíguas no texto. Por fim, utiliza-se as informações das etapas anteriores para enriquecer um prompt para reescrever o texto.

## Autores
[Beatriz Patrício](https://github.com/BeatrizPat)  
[Carlos de França](https://github.com/carlos-hfm)  
[Luis Felipe Souza](https://github.com/LuisFSouza)  
[Paula Caires](https://github.com/paulacaires)  
[Pedro Alves](https://github.com/pedrohaas)

## Pré-requisitos

> [!IMPORTANT]
> A versão do Python é 3.10

- Instale o [Conda](https://anaconda.org/anaconda/conda)
  
Os passos foram executados no terminal de uma máquina Linux, mas também se aplicam à Windows com as devidas adaptações.

## Passo a passo

Essas instruções fornecerão uma cópia do projeto em funcionamento na sua máquina local para fins de desenvolvimento e testes.

### Faça o clone do projeto
```
git clone https://github.com/PLN-UFSCar/PLN-SO-2025-01-Proj-G03.git
```

### Entre no diretório do projeto
```
cd PLN-SO-2025-01-Proj-G03/ironia
```

### Crie os três ambientes virtuais necessários
_Lembre-se de ter Conda instalado na sua máquina!_
```
conda env create -f word2vec_env.yml
conda env create -f ambiguidade_env.yml
conda env create -f main_env.yml
```
Aceite os Termos de Serviço do Conda se solicitado.

### Ative o ambiente ironia_env
```
conda activate ironia_env
```

### Abra o Jupyter Notebook
Para executar o Notebook da main
```
jupyter notebook
```

### Entre no arquivo main.ipynb
Se for a sua primeira vez executando o código, coloque a variável carregar_modelo como False na sessão "Segunda abordagem para detecção: Fine-tuning de um modelo Sentence Transformer". Dessa forma, o modelo é treinado do zero.

```
carregar_modelo = False
```

Além disso, você pode escolher a sua estratégia (word2vec ou sequence transformers). Para isso, mude as variáveis da célula conforme a sua necessidade.
```
usar_word2vec = False
usar_sequence_transformer = not usar_word2vec
```
