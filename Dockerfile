FROM debian:bullseye-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        fonts-liberation2 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        openssh-client \
        procps \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget "https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh" \
    && echo "8cb16ef82fe18d466850abb873c7966090b0fbdcf1e80842038e0b4e6d8f0b66  ./Mambaforge-4.12.0-2-Linux-x86_64.sh" > shasum \
    && sha256sum --check --status shasum \
    && mkdir -p /opt \
    && bash Mambaforge-4.12.0-2-Linux-x86_64.sh -b -p /opt/conda \
    && rm "Mambaforge-4.12.0-2-Linux-x86_64.sh" shasum \
    && /opt/conda/bin/mamba clean --all -f -y \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && ln -s /opt/conda/etc/profile.d/mamba.sh /etc/profile.d/mamba.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

COPY requirements.yaml /requirements.yaml
RUN /opt/conda/bin/mamba env create -n jupyter --file /requirements.yaml \
    && /opt/conda/bin/mamba clean --all -f -y \
    && /opt/conda/envs/jupyter/bin/pip install --no-cache neurocombat==0.2.12 \
    && mkdir -p /workspace/data /workspace/outputs ~/.config/matplotlib \
    && echo 'backend: Agg' > ~/.config/matplotlib/matplotlibrc \
    && rm /requirements.yaml

COPY . /build
SHELL ["/bin/bash", "-c"]
RUN cd /build \
    && . /opt/conda/bin/activate jupyter \
    && python setup.py bdist_wheel \
    && pip install --no-index -f dist/ causalad \
    && mv adni-experiments.sh ukb-experiments.sh /workspace/ \
    && rm -fr /build

WORKDIR /workspace

VOLUME ["/workspace/data", "/workspace/outputs"]
