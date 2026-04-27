#!/bin/sh

# Check install dependencies
echo "Instalando dependências"
./scripts/install_dependencies.sh

# Activate Python venv
source .venv/bin/activate

# Delete old results
rm results/*

# Run experiments
echo "Executando experimentos"
python3 reduced_experiments.py

# Show results
echo "Finalizada a execução dos experimentos. Os resultados podem ser encontrados na pasta 'results'"
