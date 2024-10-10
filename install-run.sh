#!/bin/bash

set -e

title="PolGen"
echo $title

principal=$(pwd)
CONDA_ROOT_PREFIX="$HOME/miniconda3"
INSTALL_ENV_DIR="$principal/env"
MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh"
CONDA_EXECUTABLE="$CONDA_ROOT_PREFIX/bin/conda"
step=1

# Function to handle errors
error() {
    echo "An error occurred during the process. Exiting the script..."
    read -p "Press any key to continue..." -n1 -s
    exit 1
}
trap error ERR

# Check internet connection
echo "[~!$step~] - Checking internet connection..."
if ping -c 1 google.com &> /dev/null; then
    echo "Internet connection is available."
    INTERNET_AVAILABLE=1
else
    echo "No internet connection detected."
    INTERNET_AVAILABLE=0
fi
echo
step=$((step + 1))

if [ ! -d "env" ]; then
    echo "[~!$step~] - Checking for Miniconda installation..."
    if [ ! -f "$CONDA_EXECUTABLE" ]; then
        echo "Miniconda is not installed. Starting download and installation..."
        echo "Downloading Miniconda..."
        curl -o miniconda.sh $MINICONDA_DOWNLOAD_URL
        if [ ! -f "miniconda.sh" ]; then
            echo "Miniconda download failed. Please check your internet connection and try again."
            error
        fi

        echo "Installing Miniconda..."
        bash miniconda.sh -b -p $CONDA_ROOT_PREFIX
        if [ $? -ne 0 ]; then
            echo "Miniconda installation failed."
            error
        fi
        rm miniconda.sh
        echo "Miniconda installation completed successfully."
    else
        echo "Miniconda is already installed. Skipping installation."
    fi
    echo
    step=$((step + 1))

    echo "[~!$step~] - Creating Conda environment..."
    $CONDA_EXECUTABLE create --no-shortcuts -y -k --prefix "$INSTALL_ENV_DIR" python=3.10
    if [ $? -ne 0 ]; then
        error
    fi
    echo "Conda environment created successfully."
    echo
    step=$((step + 1))

    echo "[~!$step~] - Installing specific pip version..."
    if [ -f "$INSTALL_ENV_DIR/bin/python" ]; then
        $INSTALL_ENV_DIR/bin/python -m pip install "pip<24.1"
        if [ $? -ne 0 ]; then
            error
        fi
        echo "Pip installed successfully."
        echo
    fi
    step=$((step + 1))

    echo "[~!$step~] - Installing dependencies..."
    source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    conda activate "$INSTALL_ENV_DIR" || error
    pip install --upgrade setuptools || error
    pip install -r "$principal/requirements.txt" || error
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 || error
    conda deactivate
    echo "Dependencies installed successfully."
    echo
    step=$((step + 1))
fi

echo "[~!$step~] - Checking for required models..."
hubert_base="$principal/rvc/models/embedders/hubert_base.pt"
fcpe="$principal/rvc/models/predictors/fcpe.pt"
rmvpe="$principal/rvc/models/predictors/rmvpe.pt"

if [ ! -f "$hubert_base" ] || [ ! -f "$fcpe" ] || [ ! -f "$rmvpe" ]; then
    echo "Required models were not found. Installing models..."
    echo
    $INSTALL_ENV_DIR/bin/python download_models.py
    if [ $? -ne 0 ]; then
        error
    fi
fi
echo
step=$((step + 1))

echo "[~!$step~] - Running Interface..."
if [ "$INTERNET_AVAILABLE" == "1" ]; then
    echo "Running app.py..."
    $INSTALL_ENV_DIR/bin/python app.py --open
else
    echo "Running app_offline.py..."
    $INSTALL_ENV_DIR/bin/python app_offline.py --open
fi
if [ $? -ne 0 ]; then
    error
fi
step=$((step + 1))

echo "Script completed successfully."
read -p "Press any key to continue..." -n1 -s
exit 0