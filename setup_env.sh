#!/bin/zsh

# Agrega el directorio raíz del proyecto al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Users/diego/Documents/tesis_standalone/tesis_unet_2"

# Activa el entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Entorno configurado para tesis_unet_2"
echo "PYTHONPATH actualizado"
if [ -d "venv" ]; then
    echo "Entorno virtual activado"
fi
