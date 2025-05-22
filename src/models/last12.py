"""
Modelo que usa el último frame de entrada como predicción
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import h5py
import numpy as np
from datetime import datetime
from config.config import LOGGING_CONFIG, TRAINING_CONFIG

class Last12(pl.LightningModule):
    """
    Modelo simple que asume que el último frame de entrada será igual
    a las siguientes 6 predicciones.
    """
    def __init__(self, n_channels=12, n_outputs=6):
        super().__init__()
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.save_hyperparameters()
        
        # Agregar listas para almacenar predicciones y valores reales
        self.test_predictions = []
        self.test_targets = []
        self.test_inputs = []
        self.test_step_outputs = []
        self.test_timestamps = []  # Nueva lista para timestamps
        
    def forward(self, x):
        # Tomar el último frame y repetirlo n_outputs veces
        last_frame = x[:, -1:, :, :]  # Shape: (batch, 1, height, width)
        prediction = last_frame.repeat(1, self.n_outputs, 1, 1)
        return prediction
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch  # Ignorar timestamps
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss.item(), prog_bar=True)
        # No retornamos loss porque no hay parámetros para entrenar
        return None
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch  # Ignorar timestamps
        y_hat = self(x)
        val_loss = nn.MSELoss()(y_hat, y).item()
        
        # Calcular métricas
        y_cpu = y.cpu()
        y_hat_cpu = y_hat.cpu()
        
        # Aplicar umbral para convertir a binario (0 o 1)
        threshold = 0.5
        y_binary = (y_cpu > threshold).float()
        y_hat_binary = (y_hat_cpu > threshold).float()
        
        # Calcular métricas
        metrics = self.calculate_metrics(y_binary, y_hat_binary)
        
        # Logging
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mse', metrics['mse'], prog_bar=True)
        self.log('val_csi', metrics['csi'], prog_bar=True)
        self.log('val_far', metrics['far'], prog_bar=True)
        self.log('val_hss', metrics['hss'], prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y, timestamps = batch  # Ahora también recibimos los timestamps
        y_hat = self(x)
        
        # Calcular métricas
        metrics = self.calculate_metrics(y, y_hat)
        
        # Almacenar resultados para análisis posterior
        self.test_predictions.append(y_hat.cpu())
        self.test_targets.append(y.cpu())
        self.test_inputs.append(x.cpu())
        self.test_timestamps.extend(timestamps)  # Guardamos los timestamps
        
        return metrics

    def on_test_epoch_start(self):
        """Inicializar listas para almacenar resultados"""
        self.test_step_outputs = []
        self.test_predictions = []
        self.test_targets = []
        self.test_inputs = []
        self.test_timestamps = []  # Nueva lista para timestamps

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
            
        # Calcular promedios de métricas
        avg_metrics = {}
        for metric in self.test_step_outputs[0].keys():
            values = [x[metric] for x in self.test_step_outputs]
            avg_metrics[metric] = torch.stack(values).mean()
            
        # Log métricas finales
        for name, value in avg_metrics.items():
            value_float = value.item() if isinstance(value, torch.Tensor) else value
            self.log(f'test_{name}', value_float)
        
        # Crear archivo H5 con predicciones
        try:
            # Concatenar todos los batches
            all_predictions = torch.cat(self.test_predictions, dim=0).numpy()
            all_targets = torch.cat(self.test_targets, dim=0).numpy()
            all_inputs = torch.cat(self.test_inputs, dim=0).numpy()
            all_timestamps = np.array(self.test_timestamps, dtype='S26')  # Convertir timestamps a bytes
            
            # Escalar las predicciones de 0-1 a 0-100
            all_predictions = all_predictions * 100.0
            
            # Crear directorio de logs si no existe
            model_log_dir = os.path.join(LOGGING_CONFIG['log_dir'], 'last12')
            os.makedirs(model_log_dir, exist_ok=True)
            
            # Guardar en archivo H5
            h5_path = os.path.join(model_log_dir, 'test_results.h5')
            
            with h5py.File(h5_path, 'w') as f:
                # Guardar datos con sus timestamps asociados
                inputs_group = f.create_group('inputs')
                inputs_group.create_dataset('data', data=all_inputs)
                inputs_group.create_dataset('timestamps', data=all_timestamps[:12])  # primeros 12 timestamps
                
                targets_group = f.create_group('targets')
                targets_group.create_dataset('data', data=all_targets)
                targets_group.create_dataset('timestamps', data=all_timestamps[12:18])  # siguientes 6 timestamps
                
                predictions_group = f.create_group('predictions')
                predictions_group.create_dataset('data', data=all_predictions)
                predictions_group.create_dataset('timestamps', data=all_timestamps[12:18])  # mismos timestamps que targets
                
                # Guardar metadatos
                f.attrs['input_frames'] = self.n_channels
                f.attrs['output_frames'] = self.n_outputs
                f.attrs['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
                f.attrs['model_type'] = 'last12'
                
                # Guardar métricas
                for name, value in avg_metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    f.attrs[f'metric_{name}'] = value
            
            print(f"\nResultados de test guardados en: {h5_path}")
            
        except Exception as e:
            print(f"Error al guardar resultados del test: {str(e)}")
        finally:
            # Limpiar memoria
            self.test_step_outputs.clear()
            self.test_predictions.clear()
            self.test_targets.clear()
            self.test_inputs.clear()
            self.test_timestamps.clear()
    
    def configure_optimizers(self):
        # No hay parámetros para optimizar en este modelo
        return None
    
    def calculate_metrics(self, y_true, y_pred):
        """Calcula métricas para evaluación"""
        # MSE
        mse = nn.MSELoss()(y_pred, y_true)
        
        # Calcular TP, FP, FN para otras métricas
        TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
        FP = torch.sum((y_pred == 1) & (y_true == 0)).float()
        FN = torch.sum((y_pred == 0) & (y_true == 1)).float()
        TN = torch.sum((y_pred == 0) & (y_true == 0)).float()
        
        # CSI (Critical Success Index)
        csi = TP / (TP + FN + FP + 1e-8)
        
        # FAR (False Alarm Ratio)
        far = FP / (TP + FP + 1e-8)
        
        # HSS (Heidke Skill Score)
        expected_random = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / (TP + TN + FP + FN + 1e-8)
        hss = (TP + TN - expected_random) / (TP + TN + FP + FN - expected_random + 1e-8)
        
        return {
            'mse': mse.item(),
            'csi': csi.item(),
            'far': far.item(),
            'hss': hss.item()
        }
