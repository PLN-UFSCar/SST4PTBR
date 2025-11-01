# Miniconda image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment files to container
COPY envs/ /app/envs/

# Create conda environments and install pt_core_news_sm
RUN conda env create -f /app/envs/ambiguity_env.yml && \
    conda env create -f /app/envs/main_env.yml && \
    conda run -n ambiguity python -m spacy download pt_core_news_sm && \
    conda run -n main python -m spacy download pt_core_news_sm

# Install jupyterlab in base environment
RUN conda install -n base -c conda-forge jupyterlab

# Activate environments and install jupyterlab kernels
RUN /opt/conda/envs/ambiguity/bin/python -m ipykernel install --user --name ambiguity --display-name "Python (Ambiguity)" && \
    /opt/conda/envs/main/bin/python -m ipykernel install --user --name main --display-name "Python (Main)"

COPY . /app/

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]