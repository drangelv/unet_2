# UNet3 para Predicción de Heatmaps
> Proyecto de tesis de maestría para predicción de heatmaps usando arquitectura UNet3 optimizada para GPU NVIDIA y Apple Silicon.

## 📋 Descripción
Este proyecto implementa un modelo UNet3 para la predicción de heatmaps, optimizado tanto para GPUs NVIDIA como para chips Apple Silicon (M1/M2/M3). El modelo puede ser configurado para predecir múltiples frames de salida a partir de una secuencia de frames de entrada.

## 🚀 Características
- Soporte para múltiples plataformas (NVIDIA CUDA, Apple Silicon MPS, CPU)
- **Control flexible de frames de predicción (1-6 frames)**
- Métricas avanzadas (CSI, FAR, HSS, MSE)
- Visualización de predicciones
- Cross-validation
- Configuración flexible
- Notebook para pruebas interactivas

## 🛠️ Requisitos
- Python 3.8+
- PyTorch 2.0+
- CUDA Toolkit 11.x+ (para GPUs NVIDIA)
- macOS 12.3+ (para Apple Silicon)

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd tesis_unet_2
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

El proyecto utiliza un archivo de configuración centralizado en `config/config.py` donde puedes ajustar:

- Hardware (MPS/CUDA/CPU)
- Parámetros del modelo (frames de entrada/salida)
- Configuración de entrenamiento (batch size, epochs)
- Rutas de datos
- Métricas y función de pérdida

### Configuración Principal
```python
# config/config.py

# Hardware
HARDWARE_CONFIG = {
    'device': 'mps',  # Opciones: 'mps', 'cuda', 'cpu'
    'gpu_id': 0,
}

# Modelo
MODEL_CONFIG = {
    'input_frames': 4,     # Frames de entrada
    'output_frames': 1,    # Frames de salida
    'initial_filters': 48,
}

# Entrenamiento
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-3,
}
```

## 🏃‍♂️ Uso

### 1. Crear Dataset Trusted
Primero, necesitas crear un dataset trusted a partir de los datos crudos. Puedes especificar el número de frames de entrada y salida:

```bash
# Usando valores por defecto (12 frames entrada, 6 salida)
python 01_create_dataset.py

# Especificando número de frames
python 01_create_dataset.py --input-frames 24 --output-frames 12

# Especificando rutas personalizadas
python 01_create_dataset.py --input-frames 8 --output-frames 4 --input-path datos/raw.h5 --output-path datos/processed.h5
```

### 2. Entrenamiento
Una vez creado el dataset trusted, puedes entrenar el modelo:

```bash
# Usando configuración por defecto (desde config.py)
python 02_train.py

# Control flexible de frames de predicción (NUEVO)
python 02_train.py --output-frames 1   # Predecir solo 1 frame
python 02_train.py --output-frames 3   # Predecir 3 frames consecutivos  
python 02_train.py --output-frames 6   # Predecir todos los 6 frames

# Cambiar número de frames en tiempo de ejecución
python 02_train.py --input-frames 24 --output-frames 12

# Usar un dataset específico
python 02_train.py --dataset inputs/mi_dataset_custom.h5

# Visualizar muestras sin entrenar
python 02_train.py --visualize_only

# Cargar modelo existente para evaluación
python 02_train.py --load_model path/to/model.ckpt
```

> **💡 Nuevo:** El proyecto ahora permite controlar dinámicamente cuántos frames predecir (1-6) sin necesidad de crear datasets separados. Ver [FLEXIBLE_FRAMES.md](FLEXIBLE_FRAMES.md) para más detalles.

### Testing Interactivo
Usar el notebook `notebooks/model_testing.ipynb` para:
- Cargar modelos entrenados
- Realizar predicciones
- Visualizar resultados
- Calcular métricas

## 📁 Estructura del Proyecto
```
tesis_unet_2/
├── 01_create_dataset.py      # Script para crear dataset trusted
├── 02_train.py              # Script principal de entrenamiento
├── config/
│   └── config.py            # Configuración centralizada
├── src/
│   ├── data/
│   │   ├── dataset.py       # Dataset para datos crudos
│   │   └── trusted_dataset.py # Dataset optimizado para datos procesados
│   ├── metrics/
│   │   └── metrics.py        # Métricas personalizadas
│   ├── models/
│   │   └── unet3.py         # Modelo UNet3
│   └── visualization/
│       └── visualizer.py     # Visualización
├── notebooks/
│   └── model_testing.ipynb   # Notebook para pruebas
├── requirements.txt          # Dependencias
```

## 📊 Métricas Implementadas

El modelo implementa las siguientes métricas:
- MSE (Error Cuadrático Medio)
- CSI (Critical Success Index)
- FAR (False Alarm Rate)
- HSS (Heidke Skill Score)

Las métricas binarias utilizan un umbral configurable para clasificación (0 por defecto).

## 💾 Guardado de Modelos

Los modelos se guardan automáticamente en:
- `saved_models/checkpoints/`: Checkpoints durante entrenamiento
- `saved_models/`: Mejor modelo y modelo final
- `logs/`: Logs de entrenamiento y métricas

## 📊 Visualización de Resultados

Los resultados se guardan en:
- `logs/predicciones/`: Predicciones del modelo
- `logs/visualizaciones/`: Visualizaciones de datos
- `logs/test_results/`: Métricas detalladas

## 🤝 Contribuciones
Para contribuir al proyecto:
1. Fork el repositorio
2. Crear una rama para tu característica
3. Commit los cambios
4. Push a la rama
5. Crear un Pull Request

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE.md para detalles.

## ✉️ Contacto
[Tu Nombre] - [tu.email@dominio.com]

## 🙏 Agradecimientos
- [Institución/Universidad]
- [Director de Tesis]
- [Otros colaboradores]
