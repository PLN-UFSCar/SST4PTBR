    #Imagem do miniconda
    FROM continuumio/miniconda3:latest

    # Define o diretório de trabalho
    WORKDIR /app

    #Copia os arquivos de ambientes para o container
    COPY ambientes/ /app/ambientes/

    #Cria os ambientes conda a partir dos arquivos yml
    RUN conda env create -f /app/ambientes/ambiguidade_env.yml && \
        conda env create -f /app/ambientes/main_env.yml && \
        conda env create -f /app/ambientes/word2vec_env.yml && \
        conda env create -f /app/ambientes/transformers_env.yml && \
        conda run -n ambiguidade python -m spacy download pt_core_news_sm && \
        conda run -n main python -m spacy download pt_core_news_sm

    #Instala o jupyterlab no ambiente base
    RUN conda install -n base -c conda-forge jupyterlab

    #Ativa os ambientes e instala os kernels do jupyterlab
    RUN /opt/conda/envs/ambiguidade/bin/python -m ipykernel install --user --name ambiguidade --display-name "Python (Ambiguidade)" && \
        /opt/conda/envs/main/bin/python -m ipykernel install --user --name main --display-name "Python (Main)" && \
        /opt/conda/envs/word2vec_env/bin/python -m ipykernel install --user --name word2vec_env --display-name "Python (Word2Vec)"

    COPY . /app/

    EXPOSE 8888

    CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]