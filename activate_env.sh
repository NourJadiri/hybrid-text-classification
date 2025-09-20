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
