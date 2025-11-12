# SST4PTBR - Detection of sarcasm and textual ambiguities with prompt enrichment for linguistic transformation

The proposed NLP task aims at the detection and elimination of sarcasm and lexical ambiguities in texts, aiming to make them clearer and more objective.

## Steps

To achieve this goal, three tasks are performed: first, detecting possibly sarcastic sentences in the text. Second, detecting the meaning of possibly ambiguous words in the text. Finally, the information from the previous steps is used to enrich a prompt for rewriting the text.

<div align = "center">
    <img src="https://github.com/user-attachments/assets/f27b9432-874b-4afb-bd59-d3c5fe6af968">
</div>

## Examples
| Original text | Rewritten text |
|---|---|
| Claro que adorei a sua "ajuda" na arrumação da casa. É impressionante como o simples ato de você ficar sentado no sofá fez tudo se organizar magicamente. | Eu não gostei da sua ação durante a arrumação da residência. Sua permanência sentada no sofá não causou a organização das coisas. |
| O trânsito na cidade na véspera de feriado é muito fluido. É um prazer dirigir por horas a uma velocidade média de 5 km/h, apreciando a paisagem dos outros carros. O som das buzinas forma uma sinfonia harmoniosa que acalma os nervos. Chegar ao seu destino é apenas um detalhe. A verdadeira aventura é o caminho. | O trânsito na cidade na véspera de feriado é muito lento e congestionado. É incômodo dirigir por horas a uma velocidade média de 5 km/h, vendo os outros carros ao redor. O som das buzinas cria um barulho alto que causa estresse. Atingir o destino final é o objetivo principal. O percurso é difícil. |
| Decidi adotar um filhote de cachorro para ter mais companhia em casa. Ele é um anjinho, super calmo e obediente. Meu novo chinelo, que ele carinhosamente roeu, ficou com um design muito mais arrojado. As poças de xixi em lugares aleatórios da casa dão um toque de surpresa ao meu dia. Ensinar a dar a pata tem sido um processo rápido. | Eu decidi adotar um filhote de cachorro para ter mais companhia na minha residência. Ele não é calmo nem obediente. Meu chinelo novo foi roído por ele. O chinelo ficou danificado. Ele fez poças de urina em locais aleatórios da residência. Isso causa transtorno durante o meu dia. Ensinar a dar a pata tem sido um processo demorado. |

## Quantitative analysis

A spreadsheet with a quantitative analysis of the system's results using metrics is available in the `evaluation` folder.

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

### Run the project using docker
```
docker compose up -d --build
```

### Access the [main notebook](http://localhost:8888/lab/tree/notebooks/main.ipynb).
Access the main notebook and follow the steps described in this.
