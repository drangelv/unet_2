"""
Callbacks para visualización durante el entrenamiento
"""

import os
import torch
import pytorch_lightning as pl
from src.visualization.visualizer import visualize_prediction


class VisualizationCallback(pl.Callback):
    """Callback para guardar visualizaciones durante el entrenamiento"""
    
    def __init__(self, save_dir, num_samples=3):
        """
        Args:
            save_dir (str): Directorio donde guardar las visualizaciones
            num_samples (int): Número de muestras a visualizar
        """
        super().__init__()
        self.save_dir = save_dir
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        """Guardar visualizaciones al final de cada época de validación"""
        try:
            # Obtener algunas muestras del conjunto de validación
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            
            # Asumiendo que batch es una tupla de tensores (inputs, targets)
            # o es un tensor único que contiene ambos
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[:2]  # Tomar solo los primeros dos elementos si hay más
            else:
                # Si el batch es un tensor único, intentar dividirlo
                inputs = batch[:, :pl_module.input_frames]
                targets = batch[:, pl_module.input_frames:]

            # Generar y guardar predicciones
            for i in range(min(self.num_samples, len(inputs))):
                input_seq = inputs[i:i+1]
                target_seq = targets[i:i+1]
                
                # Realizar predicción
                with torch.no_grad():
                    prediction = pl_module(input_seq.to(pl_module.device))
                
                # Guardar visualización
                visualize_prediction(
                    model=None,  # No necesitamos el modelo aquí
                    dataset=None,  # No necesitamos el dataset aquí
                    sample_idx=i,
                    save_dir=self.save_dir,
                    input_seq=input_seq,
                    target_seq=target_seq,
                    pred_seq=prediction)
        except Exception as e:
            print(f"Error en VisualizationCallback: {str(e)}")
            # No interrumpir el entrenamiento por errores de visualización
