# SST4PTBR - Detection of sarcasm and textual ambiguities with prompt enrichment for linguistic transformation

The proposed NLP task aims at the detection and elimination of sarcasm and lexical ambiguities in texts, aiming to make them clearer and more objective.

To achieve this goal, three tasks are performed: first, detecting possibly sarcastic sentences in the text. Second, detecting the meaning of possibly ambiguous words in the text. Finally, the information from the previous steps is used to enrich a prompt for rewriting the text.

## Quantitative analysis

The spreadsheet with the quantitative analysis done by the project members is available in the `analisys.xlsx` file.

## Prerequisites

> [!IMPORTANT]
> Have docker and docker compose installed

- Ensure you have docker and docker compose installed on your machine. [Access the docker documentation here](https://www.docker.com/).

## Step-by-step

### Clone the project
```
git clone https://github.com/PLN-UFSCar/SST4PTBR.git
```

### Enter the project directory
```
cd SST4PTBR
```

### Clone the Dataset
Clone the [PLNCrawler](https://github.com/schuberty/PLNCrawler/tree/master) repository which contains the dataset used.
```
git clone https://github.com/schuberty/PLNCrawler.git
```

### Download the lexicon available at [OpenWordnet-PT](https://github.com/own-pt/openWordnet-PT/releases), named `own-pt.tar.gz`, and place it inside the project's `lexicon` folder.

### Run the project using docker
```
docker compose up -d --build
```

### Access the [main notebook](http://localhost:8888/lab/tree/notebooks/main.ipynb).
If it's your first time running the code, set the `load_model` variable to `False` in the session "Second approach for detection: Fine-tuning of a Sentence Transformer model". This way, the model is trained from scratch.
```python
load_model = False
```

Additionally, you can choose your strategy (word2vec or sequence transformers). To do this, change the variables in the cell according to your needs.
```python
use_word2vec = False
use_sequence_transformer = not use_word2vec
```

You will also need to put your Google Gemini API key in the `API_KEY` variable.
```python
API_KEY = 'your_key_here'
```
