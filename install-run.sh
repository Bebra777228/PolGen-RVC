#!/bin/bash

set -e

title="PolGen"
echo $title

if [ ! -d "env" ]; then
    principal=$(pwd)
    CONDA_ROOT_PREFIX="$HOME/miniconda3"
    INSTALL_ENV_DIR="$principal/env"
    MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Linux-x86_64.sh"
    CONDA_EXECUTABLE="$CONDA_ROOT_PREFIX/bin/conda"

    if [ ! -f "$CONDA_EXECUTABLE" ]; then
        echo "Miniconda not found. Starting download and installation..."
        echo "Downloading Miniconda..."
        curl -o miniconda.sh $MINICONDA_DOWNLOAD_URL
        if [ ! -f "miniconda.sh" ]; then
            echo "Download failed. Please check your internet connection and try again."
            exit 1
        fi

        echo "Installing Miniconda..."
        bash miniconda.sh -b -p $CONDA_ROOT_PREFIX
        if [ $? -ne 0 ]; then
            echo "Miniconda installation failed."
            exit 1
        fi
        rm miniconda.sh
        echo "Miniconda installation complete."
    else
        echo "Miniconda already installed. Skipping installation."
    fi
    echo

    echo "Creating Conda environment..."
    $CONDA_EXECUTABLE create --no-shortcuts -y -k --prefix "$INSTALL_ENV_DIR" python=3.9
    if [ $? -ne 0 ]; then
        exit 1
    fi
    echo "Conda environment created successfully."
    echo

    if [ -f "$INSTALL_ENV_DIR/bin/python" ]; then
        echo "Installing specific pip version..."
        $INSTALL_ENV_DIR/bin/python -m pip install "pip<24.1"
        if [ $? -ne 0 ]; then
            exit 1
        fi
        echo "Pip installation complete."
        echo
    fi

    echo "Installing dependencies..."
    source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    conda activate "$INSTALL_ENV_DIR" || exit 1
    pip install --upgrade setuptools || exit 1
    pip install -r "$principal/requirements.txt" || exit 1
    conda deactivate
    echo "Dependencies installation complete."
    echo
fi

$INSTALL_ENV_DIR/bin/python download_models.py
echo

$INSTALL_ENV_DIR/bin/python app.py --open
echo
read -p "Press any key to continue..." -n1 -s
exit 0

error() {
    echo "An error occurred during installation. Please check the output above for details."
    read -p "Press any key to continue..." -n1 -s
    exit 1
}
trap error ERR
