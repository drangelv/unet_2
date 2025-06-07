# Control Flexible de Frames de Predicción

Este proyecto ahora permite controlar dinámicamente cuántos frames consecutivos predecir de los 6 frames disponibles en el dataset.

## Configuración

### Método 1: Archivo de configuración
Edita `config/config.py` para cambiar el número de frames de salida:

```python
MODEL_CONFIG = {
    'model_name': 'unet3',
    'input_frames': 12,      # Siempre 12 frames de entrada
    'output_frames': 1,      # Número de frames a predecir (1-6)
    'initial_filters': 48,
    'bilinear': True
}
```

### Método 2: Argumentos de línea de comandos
Usa el parámetro `--output-frames` al ejecutar el script:

```bash
# Predecir solo 1 frame
python 02_train.py --output-frames 1 --visualize_only

# Predecir 3 frames consecutivos
python 02_train.py --output-frames 3 --visualize_only

# Predecir todos los 6 frames disponibles
python 02_train.py --output-frames 6 --visualize_only
```

## Funcionamiento

- El dataset base (`data_trusted_12x6.h5`) contiene 6 frames consecutivos para predicción
- El sistema extrae automáticamente solo los primeros `output_frames` frames
- Los timestamps también se ajustan automáticamente
- No es necesario crear datasets separados para diferentes configuraciones

## Ejemplos de Uso

### Predicción de 1 frame (más rápido, menos información temporal)
```bash
python 02_train.py --output-frames 1
```

### Predicción de 3 frames (balance entre velocidad y información temporal)
```bash
python 02_train.py --output-frames 3
```

### Predicción de 6 frames (máxima información temporal, más lento)
```bash
python 02_train.py --output-frames 6
```

## Ventajas

1. **Flexibilidad**: Experimenta con diferentes horizontes de predicción sin cambiar datasets
2. **Eficiencia**: Usa menos recursos computacionales con menos frames
3. **Compatibilidad**: Mantiene compatibilidad con el dataset existente
4. **Comparación**: Permite comparar fácilmente modelos con diferentes configuraciones

## Verificación

Para verificar que el sistema funciona correctamente:

```bash
# Verificar con 1 frame
python 02_train.py --output-frames 1 --visualize_only

# Verificar con 3 frames  
python 02_train.py --output-frames 3 --visualize_only

# Verificar con 6 frames
python 02_train.py --output-frames 6 --visualize_only
```

Cada ejecución mostrará:
- Configuración del modelo
- Número de frames usados vs disponibles
- Shapes correctos en los datos
- Visualizaciones generadas

## Limitaciones

- Máximo 6 frames de predicción (limitado por el dataset)
- Mínimo 1 frame de predicción
- Los frames siempre se toman consecutivamente desde el inicio
