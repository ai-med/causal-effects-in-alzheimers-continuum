FROM continuumio/miniconda3:4.9.2

COPY requirements.yaml /

RUN set -xe \
    && conda install -q -y tini=0.18.0 \
    && conda env create -n jupyter -q --file /requirements.yaml \
    && conda clean --all -f -y \
    && rm -f /requirements.yaml

RUN set -xe \
    && buildDeps="fonts-liberation2" \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y $buildDeps --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /notebooks/data && echo 'backend: Agg' > /notebooks/matplotlibrc
COPY adni*.ipynb run-synthetic-ukb.sh /notebooks/
COPY causalad /notebooks/causalad
WORKDIR /notebooks

RUN conda run --no-capture-output -n jupyter \
    python -c 'from causalad.pystan_models.model import compile_stan_models; compile_stan_models()'

EXPOSE 8888
VOLUME ["/notebooks/data"]

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
# Run juypter in a conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "jupyter", "jupyter", "notebook", "--notebook-dir=/notebooks", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]