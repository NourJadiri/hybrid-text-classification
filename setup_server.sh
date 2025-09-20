#!/bin/bash

# Setup script for hybrid-text-classification project on remote server
# This script sets up the complete environment for training

set -e  # Exit on any error

echo "=== Starting server setup for hybrid-text-classification project ==="

# Update and upgrade the system
echo ">>> Updating and upgrading system packages..."
apt update -y
apt upgrade -y

# Install essential build tools and dependencies
echo ">>> Installing essential build dependencies..."
apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    git \
    unzip \
    graphviz \
    graphviz-dev \
    pkg-config

# Configure git user settings
echo ">>> Configuring git user settings..."
git config --global user.name "Nour Jadiri"
git config --global user.email "ick62377@gmail.com"
echo "Git user configured: Nour Jadiri <ick62377@gmail.com>"

# Install pyenv if not already installed
echo ">>> Installing pyenv..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
    
    # Add pyenv to PATH
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    
    # Add to shell profile for persistence
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
else
    echo "pyenv already installed"
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi

# Install Python 3.12.3
echo ">>> Installing Python 3.12.3 via pyenv..."
pyenv install -s 3.12.3
pyenv global 3.12.3

# Verify Python installation
echo ">>> Verifying Python installation..."
python --version
which python

# Assume we're running from within the project directory
# No need to create or check directories - everything is already here

# Create virtual environment
echo ">>> Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo ">>> Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ">>> Upgrading pip..."
pip install --upgrade pip

# Unzip data archive if it exists
echo ">>> Looking for and extracting data archive..."
if [ -f "all-texts-unified.zip" ]; then
    echo "Found all-texts-unified.zip, extracting..."
    unzip -o all-texts-unified.zip
    echo "Data extraction completed"
else
    echo "all-texts-unified.zip not found - skipping data extraction"
fi

# Install PyTorch first (for better dependency resolution)
echo ">>> Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install pygraphviz separately (often problematic)
echo ">>> Installing pygraphviz..."
pip install pygraphviz

# Install requirements from requirements.txt
echo ">>> Installing project dependencies..."
pip install -r requirements.txt
echo "All dependencies installed successfully"

# Create activation script for convenience
echo ">>> Creating convenience activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Convenience script to activate the project environment

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

source .venv/bin/activate

echo "Environment activated!"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
EOF

chmod +x activate_env.sh

# Verify installation
echo ">>> Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import pygraphviz; print('pygraphviz installed successfully')"

echo ""
echo "=== Setup completed successfully! ==="
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source activate_env.sh"
echo ""
echo "Or manually:"
echo "  export PYENV_ROOT=\"\$HOME/.pyenv\""
echo "  export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
echo "  eval \"\$(pyenv init --path)\""
echo "  eval \"\$(pyenv init -)\""
echo "  source .venv/bin/activate"
echo ""
echo "All files should be in the current project directory:"
echo "- all-texts-unified.zip (data archive)"
echo "- requirements.txt (dependencies)"
echo "- Your training scripts and source code"