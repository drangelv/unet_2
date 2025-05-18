"""
Configuración del entorno para el proyecto UNet de predicción de heatmaps
"""

# Configuración del Hardware
HARDWARE_CONFIG = {
    'device': 'mps',  # Opciones: 'mps' para Mac Silicon, 'cuda' para GPU NVIDIA, 'cpu' para CPU
    'gpu_id': 0,      # ID de la GPU a utilizar si hay múltiples
}

# Configuración del Modelo
MODEL_CONFIG = {
    'input_frames': 12,     # Número de frames de entrada
    'output_frames': 6,    # Número de frames de salida
    'initial_filters': 48, # Número de filtros iniciales en UNet Original 48
    'bilinear': True,     # Usar interpolación bilinear en upsampling
}

# Configuración de Entrenamiento
TRAINING_CONFIG = {
    'batch_size': 4,
    'epochs': 3,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 8,
    'gradient_clip_val': 0.5,
    'accumulate_grad_batches': 2,
}

# Configuración de Datos
DATA_CONFIG = {
    'data_path': "inputs/combined_data_final.h5",  # Dataset crudo
    'trusted_data_path': "inputs/data_trusted_12x6.h5",  # Dataset procesado
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'shuffle_seed': 42,
}

# Configuración de Métricas
METRICS_CONFIG = {
    'loss_function': 'mse',  # Opciones: 'mse', 'mae', 'huber'
    'threshold': 0.0,        # Umbral para binarización en métricas de clasificación
    'metrics': [
        'mse',              # Error cuadrático medio
        'csi',              # Critical Success Index
        'far',              # False Alarm Rate
        'hss',              # Heidke Skill Score
    ]
}

# Configuración de Logging y Checkpoints
LOGGING_CONFIG = {
    'save_dir': './saved_models',
    'log_dir': './logs',
    'model_name': 'UNet3',
    'save_top_k': 2,
    'log_every_n_steps': 10,
}
