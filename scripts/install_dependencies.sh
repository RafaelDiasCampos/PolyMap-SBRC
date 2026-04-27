# !/bin/sh

# Create Python venv
echo "Criando venv Python"
python3 -m venv .venv

source .venv/bin/activate

# Install dependencies
echo "Instalando dependências"
pip3 install -r requirements.txt
pip3 install torch torchvision
