# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# Define ARGs
ARG ENV_NAME=data_science_ml_preprocessor3
ARG PYTHON_VER=3.10

# Prevent tzdata from asking for user input
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Set environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV ENV_NAME=$ENV_NAME

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    libstdc++6 \
    libgcc1 \
    libc6 \
    libc6-dev \
    libc-dev-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -a -y

# Create a new Conda environment
COPY .devcontainer/environment.yml /tmp/environment.yml
RUN sed -i "s/name: \${ENV_NAME}/name: $ENV_NAME/" /tmp/environment.yml && \
    sed -i "s/- python=\${PYTHON_VER}/- python=$PYTHON_VER/" /tmp/environment.yml && \
    conda env create -f /tmp/environment.yml \
    && conda clean -a -y

# Set up shell initialization for Conda
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME" >> /etc/bash.bashrc

# Upgrade pip and install additional packages within the Conda environment
RUN /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

# Set working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["/bin/bash"]