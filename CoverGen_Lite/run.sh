# Install Debian packages
sudo apt-get update
sudo apt-get install -qq -y build-essential ffmpeg aria2

# Upgrade pip and setuptools
pip install --upgrade pip
pip install --upgrade setuptools

# Install wheel package (built-package format for Python)
pip install wheel

# Install Python packages using pip
pip install -r requirements.txt

# Run application locally at http://127.0.0.1:7860
python app.py
