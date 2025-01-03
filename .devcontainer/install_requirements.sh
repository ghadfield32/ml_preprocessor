#!/usr/bin/env bash
ENV_NAME=$1
PYTHON_VER=$2
CPU=$(uname -m)

# Install Miniconda
# apt-get update updates the package list.
# wget downloads the Miniconda installer script.
# /bin/bash executes the installer script.
# Options:
#    Change the URL to download a different version of Miniconda if needed.
apt update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update 

wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CPU}.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && export PATH=/opt/conda/bin:$PATH \
    && conda init bash \
    && conda install conda-build

# Set environment
# Create a new Conda environment and activate it.
. /root/.bashrc \
    && conda create -y --name $ENV_NAME python=$PYTHON_VER 

echo "export QUARTO_PYTHON=/opt/conda/envs/${ENV_NAME}/bin/python3" >> ~/.bashrc
echo "conda activate $ENV_NAME" >> ~/.bashrc

conda activate $ENV_NAME

# Install the Python packages
# pip3 install installs the packages listed in requirements.txt.
pip3 install -r /requirements/requirements.txt