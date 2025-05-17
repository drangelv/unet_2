# Guía de Instalación Detallada

## Instalación por Sistema Operativo

### macOS (Apple Silicon)
1. Instalar Homebrew (si no está instalado):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Instalar Python 3.8+ usando Homebrew:
```bash
brew install python@3.8
```

3. Crear y activar entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Instalar dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS (Intel)
Seguir los mismos pasos que para Apple Silicon, pero usar PyTorch con soporte CPU:
```bash
pip install torch torchvision torchaudio
```

### Linux (NVIDIA GPU)
1. Instalar CUDA Toolkit y cuDNN según tu GPU:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn)

2. Crear entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar PyTorch con soporte CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Instalar el resto de dependencias:
```bash
pip install -r requirements.txt
```

### Windows (NVIDIA GPU)
1. Instalar [Anaconda](https://www.anaconda.com/download)

2. Crear entorno conda:
```bash
conda create -n tesis_unet python=3.8
conda activate tesis_unet
```

3. Instalar CUDA Toolkit a través de conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Instalar dependencias adicionales:
```bash
pip install -r requirements.txt
```

## Verificación de la Instalación

Ejecutar el siguiente script para verificar la instalación:
```python
import torch
import pytorch_lightning as pl
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Verificar disponibilidad de hardware
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Solución de Problemas Comunes

### Error CUDA "out of memory"
- Reducir batch_size en config.py
- Liberar memoria GPU entre ejecuciones
- Verificar otros procesos usando la GPU

### Error MPS "not available"
- Verificar versión de macOS (12.3+)
- Actualizar PyTorch a 2.0+
- Reinstalar PyTorch con soporte MPS

### Error al cargar h5py
- Instalar/actualizar hdf5:
  - macOS: `brew install hdf5`
  - Linux: `sudo apt-get install libhdf5-dev`
  - Windows: Usar conda: `conda install h5py`

### Problemas con matplotlib
- macOS: `pip install matplotlib`
- Linux: `sudo apt-get install python3-matplotlib`
- Windows: `conda install matplotlib`

## Configuración del Entorno de Desarrollo

### VS Code
Extensiones recomendadas:
- Python
- Pylance
- Jupyter
- Python Test Explorer

Settings recomendados (settings.json):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm
Configuración recomendada:
1. Abrir el proyecto
2. File -> Settings -> Project -> Python Interpreter
3. Add Interpreter -> Virtualenv Environment -> Existing
4. Seleccionar el intérprete en venv/bin/python

## Actualizaciones y Mantenimiento

### Actualizar dependencias
```bash
pip install --upgrade -r requirements.txt
```

### Limpiar cache y archivos temporales
```bash
# Linux/macOS
find . -type d -name "__pycache__" -exec rm -r {} +
rm -rf .pytest_cache
rm -rf .ipynb_checkpoints

# Windows
del /s /q __pycache__
del /s /q .pytest_cache
del /s /q .ipynb_checkpoints
```
