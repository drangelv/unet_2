#!/usr/bin/env python
"""
Script principal para entrenamiento del modelo UNet3
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import multiprocessing

import sys
sys.path.append('.')

from config.config import (MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, 
                         METRICS_CONFIG, HARDWARE_CONFIG, LOGGING_CONFIG)
from src.models.unet3 import UNet3
from src.models.unet4 import UNet4
from src.models.last12 import Last12
from src.data.trusted_dataset import TrustedHeatmapDataModule
from src.visualization.visualizer import visualize_sample, visualize_prediction
from src.utils.system_info import print_system_info
import sys
sys.path.append('.')

def get_model(model_name):
    """Retorna el modelo según el nombre especificado"""
    if model_name == "unet3":
        return UNet3(
            n_channels=MODEL_CONFIG['input_frames'],
            n_classes=MODEL_CONFIG['output_frames'],
            bilinear=MODEL_CONFIG['bilinear']
        )
    elif model_name == "unet4":
        return UNet4(
            n_channels=MODEL_CONFIG['input_frames'],
            n_classes=MODEL_CONFIG['output_frames'],
            bilinear=MODEL_CONFIG['bilinear']
        )
    elif model_name == "last12":
        return Last12(
            n_channels=MODEL_CONFIG['input_frames'],
            n_outputs=MODEL_CONFIG['output_frames']
        )
    else:
        raise ValueError(f"Modelo {model_name} no reconocido")

def setup_hardware():
    """Configura el hardware según la configuración y disponibilidad"""
    # Verificar MPS (Apple Silicon)
    if HARDWARE_CONFIG['device'] == 'mps':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("Usando aceleración MPS (Apple Silicon)")
            HARDWARE_CONFIG['device'] = 'mps'
            return "mps"
        else:
            print("MPS no disponible, usando CPU")
            HARDWARE_CONFIG['device'] = 'cpu'
            return "cpu"
    
    # Verificar CUDA (NVIDIA GPU)
    elif HARDWARE_CONFIG['device'] == 'cuda':
        if torch.cuda.is_available():
            cuda_id = HARDWARE_CONFIG['gpu_id']
            if cuda_id >= torch.cuda.device_count():
                print(f"GPU {cuda_id} no disponible, usando GPU 0")
                HARDWARE_CONFIG['gpu_id'] = 0
            print(f"Usando GPU NVIDIA: {torch.cuda.get_device_name(HARDWARE_CONFIG['gpu_id'])}")
            return f"cuda:{HARDWARE_CONFIG['gpu_id']}"
        else:
            print("CUDA no disponible, usando CPU")
            HARDWARE_CONFIG['device'] = 'cpu'
            return "cpu"
    
    # CPU como fallback
    else:
        print("Usando CPU")
        HARDWARE_CONFIG['device'] = 'cpu'
        return "cpu"

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de UNet3 para predicción de heatmaps')
    parser.add_argument('--config', type=str, default=None,
                       help='Ruta a archivo de configuración alternativo')
    parser.add_argument('--visualize_only', action='store_true',
                       help='Solo visualizar muestras sin entrenar')
    parser.add_argument('--load_model', type=str, default='',
                       help='Ruta a modelo guardado para evaluación')
    parser.add_argument('--input-frames', type=int, default=None,
                       help='Número de frames de entrada (por defecto usa el valor de config.py)')
    parser.add_argument('--output-frames', type=int, default=None,
                       help='Número de frames de salida (por defecto usa el valor de config.py)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Ruta al dataset trusted a usar (por defecto usa el valor de config.py)')
    args = parser.parse_args()

    # Actualizar configuración si se especificaron argumentos
    if args.input_frames is not None:
        MODEL_CONFIG['input_frames'] = args.input_frames
    if args.output_frames is not None:
        MODEL_CONFIG['output_frames'] = args.output_frames
    if args.dataset is not None:
        DATA_CONFIG['trusted_data_path'] = args.dataset
    elif args.input_frames is not None or args.output_frames is not None:
        # Si se cambió el número de frames pero no se especificó dataset, ajustar la ruta
        DATA_CONFIG['trusted_data_path'] = f"inputs/data_trusted_{MODEL_CONFIG['input_frames']}x{MODEL_CONFIG['output_frames']}.h5"

    # Configurar hardware
    device = setup_hardware()
    print(f"Usando dispositivo: {device}")

    # Imprimir información del sistema
    print_system_info()
    print(f"\nConfiguración del modelo:")
    print(f"- Frames de entrada: {MODEL_CONFIG['input_frames']}")
    print(f"- Frames de salida: {MODEL_CONFIG['output_frames']}")
    print(f"- Dataset: {DATA_CONFIG['trusted_data_path']}")

    # Crear directorios necesarios
    os.makedirs(LOGGING_CONFIG['save_dir'], exist_ok=True)
    os.makedirs(LOGGING_CONFIG['log_dir'], exist_ok=True)

    # Crear data module
    data_module = TrustedHeatmapDataModule(
        data_path=DATA_CONFIG['trusted_data_path'],
        batch_size=TRAINING_CONFIG['batch_size'],
        train_ratio=DATA_CONFIG['train_split'],
        val_ratio=DATA_CONFIG['val_split']
    )

    # Solo visualizar si se solicita
    if args.visualize_only:
        data_module.setup()
        print("\nVisualizando muestras del dataset:")
        for i in range(3):
            visualize_sample(data_module.train_dataset, i, 
                           os.path.join(LOGGING_CONFIG['log_dir'], "visualizaciones"))
        return

    # Cargar modelo existente o crear nuevo
    if args.load_model:
        print(f"\nCargando modelo desde: {args.load_model}")
        model = UNet3.load_from_checkpoint(args.load_model)
        
        # Realizar predicciones
        data_module.setup()
        print("\nGenerando predicciones:")
        for i in range(5):
            visualize_prediction(
                model, 
                data_module.test_dataset, 
                i,
                os.path.join(LOGGING_CONFIG['log_dir'], "predicciones")
            )
        return

    # Crear modelo nuevo
    model = get_model(MODEL_CONFIG['model_name'])

    # Configurar callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(LOGGING_CONFIG['save_dir'], "checkpoints"),
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=LOGGING_CONFIG['save_top_k'],
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=TRAINING_CONFIG['early_stopping_patience'],
        mode="min"
    )

    # Configurar trainer
    trainer_config = {
        'max_epochs': TRAINING_CONFIG['epochs'],
        'callbacks': [checkpoint_callback, early_stop_callback],
        'gradient_clip_val': TRAINING_CONFIG['gradient_clip_val'],
        'accumulate_grad_batches': TRAINING_CONFIG['accumulate_grad_batches'],
        'log_every_n_steps': LOGGING_CONFIG['log_every_n_steps'],
        'default_root_dir': LOGGING_CONFIG['log_dir']
    }

    # Configuración específica según el dispositivo
    if HARDWARE_CONFIG['device'] == 'cuda':
        trainer_config.update({
            'accelerator': 'cuda',
            'devices': [HARDWARE_CONFIG['gpu_id']]
        })
    elif HARDWARE_CONFIG['device'] == 'mps':
        trainer_config.update({
            'accelerator': 'mps',
            'devices': 1
        })
    else:
        trainer_config.update({
            'accelerator': 'cpu'
        })

    trainer = pl.Trainer(**trainer_config)

    # Entrenar
    print("\nIniciando entrenamiento:")
    trainer.fit(model, data_module)

    # Evaluar
    print("\nEvaluando en conjunto de test:")
    trainer.test(model, data_module)

    # Guardar algunas predicciones
    print("\nGenerando visualizaciones finales:")
    data_module.setup()
    for i in range(5):
        visualize_prediction(
            model, 
            data_module.test_dataset, 
            i,
            os.path.join(LOGGING_CONFIG['log_dir'], "predicciones_finales")
        )

    print("\n¡Entrenamiento completado!")
    print(f"Modelo guardado en: {LOGGING_CONFIG['save_dir']}")
    print(f"Logs disponibles en: {LOGGING_CONFIG['log_dir']}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
