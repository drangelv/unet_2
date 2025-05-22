"""
Implementación del modelo UNet3 y sus componentes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..metrics.metrics import calculate_metrics
from config.config import MODEL_CONFIG, METRICS_CONFIG, LOGGING_CONFIG, TRAINING_CONFIG
from datetime import datetime
import os
import h5py
from datetime import datetime

# Verificar configuración de frames
assert MODEL_CONFIG['input_frames'] > 0, "El número de frames de entrada debe ser mayor que 0"
assert MODEL_CONFIG['output_frames'] > 0, "El número de frames de salida debe ser mayor que 0"
INPUT_FRAMES = MODEL_CONFIG['input_frames']
OUTPUT_FRAMES = MODEL_CONFIG['output_frames']

class DoubleConv(nn.Module):
    """Bloque de doble convolución con batch normalization"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Operación de downsampling seguida de doble convolución"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Operación de upsampling seguida de doble convolución"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Compensar diferencias en dimensiones
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenar
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Capa de convolución final"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3(pl.LightningModule):
    """
    Implementación de UNet3 usando PyTorch Lightning
    
    Args:
        n_channels (int): Número de canales de entrada
        n_classes (int): Número de canales de salida
        bilinear (bool): Usar interpolación bilinear en upsampling
        learning_rate (float): Tasa de aprendizaje inicial
    """
    def __init__(self, n_channels=MODEL_CONFIG['input_frames'], 
                 n_classes=MODEL_CONFIG['output_frames'],
                 bilinear=MODEL_CONFIG['bilinear'],
                 learning_rate=1e-3):
        super(UNet3, self).__init__()
        
        # Verificar que los parámetros coinciden con la configuración
        if n_channels != MODEL_CONFIG['input_frames']:
            print(f"Advertencia: n_channels ({n_channels}) no coincide con MODEL_CONFIG['input_frames'] ({MODEL_CONFIG['input_frames']})")
        if n_classes != MODEL_CONFIG['output_frames']:
            print(f"Advertencia: n_classes ({n_classes}) no coincide con MODEL_CONFIG['output_frames'] ({MODEL_CONFIG['output_frames']})")
            
        self.save_hyperparameters()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.learning_rate = learning_rate

        # Parámetros del modelo
        self.initial_filters = MODEL_CONFIG['initial_filters']
        factor = 2 if bilinear else 1

        # Capas del modelo
        self.inc = DoubleConv(n_channels, self.initial_filters)
        self.down1 = Down(self.initial_filters, self.initial_filters*2)
        self.down2 = Down(self.initial_filters*2, self.initial_filters*4)
        self.down3 = Down(self.initial_filters*4, self.initial_filters*8 // factor)
        
        self.up1 = Up(self.initial_filters*8, self.initial_filters*4 // factor, bilinear)
        self.up2 = Up(self.initial_filters*4, self.initial_filters*2 // factor, bilinear)
        self.up3 = Up(self.initial_filters*2, self.initial_filters, bilinear)
        self.outc = OutConv(self.initial_filters, n_classes)

        # Métricas
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Log información del modelo
        self.example_input_array = torch.zeros(1, n_channels, 128, 128)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"UNet3 creada con {params:,} parámetros entrenables")

        # Agregar listas para almacenar predicciones y valores reales
        self.test_predictions = []
        self.test_targets = []
        self.test_inputs = []

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def _get_loss(self, y_hat, y):
        """Calcula la función de pérdida según la configuración"""
        if METRICS_CONFIG['loss_function'] == 'mse':
            return F.mse_loss(y_hat, y)
        elif METRICS_CONFIG['loss_function'] == 'mae':
            return F.l1_loss(y_hat, y)
        elif METRICS_CONFIG['loss_function'] == 'huber':
            return F.smooth_l1_loss(y_hat, y)
        else:
            raise ValueError(f"Función de pérdida no soportada: {METRICS_CONFIG['loss_function']}")

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self._get_loss(y_hat, y)
        
        # Log metrics
        metrics = calculate_metrics(y_hat, y, METRICS_CONFIG['threshold'])
        for name, value in metrics.items():
            self.log(f'train_{name}', value, on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=x.shape[0])
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self._get_loss(y_hat, y)
        
        # Asegurar que todo esté en el mismo dispositivo
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)
        
        metrics = calculate_metrics(y_hat, y, METRICS_CONFIG['threshold'])
        metrics['loss'] = loss
        self.validation_step_outputs.append(metrics)
        
        return metrics

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        # Calcular promedios
        avg_metrics = {}
        for metric in self.validation_step_outputs[0].keys():
            values = [x[metric] for x in self.validation_step_outputs]
            avg_metrics[metric] = torch.stack(values).mean()
            
        # Log métricas
        for name, value in avg_metrics.items():
            # Convertir a float para logging
            value_float = value.item() if isinstance(value, torch.Tensor) else value
            self.log(f'val_{name}', value_float, prog_bar=True)
            
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y, timestamps = batch
        y_hat = self(x)
        
        # Asegurar que todo esté en el mismo dispositivo
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)
        
        # Guardar predicciones, targets y timestamps para H5
        self.test_predictions.append(y_hat.cpu().detach())
        self.test_targets.append(y.cpu().detach())
        self.test_inputs.append(x.cpu().detach())
        
        # Calcular métricas
        metrics = calculate_metrics(y_hat, y, METRICS_CONFIG['threshold'])
        self.test_step_outputs.append(metrics)
        
        # Log métricas individuales
        for name, value in metrics.items():
            self.log(f'test_{name}', value, on_step=False, on_epoch=True)
            
        return metrics

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
            
            # Escalar las predicciones de 0-1 a 0-100
            all_predictions = all_predictions * 100.0
            
            # Crear directorio de logs si no existe
            model_log_dir = os.path.join(LOGGING_CONFIG['log_dir'], 'unet3')
            os.makedirs(model_log_dir, exist_ok=True)
            
            # Guardar en archivo H5 sin timestamp
            h5_path = os.path.join(model_log_dir, 'test_results.h5')
            
            with h5py.File(h5_path, 'w') as f:
                # Guardar datos
                f.create_dataset('predictions', data=all_predictions)
                f.create_dataset('targets', data=all_targets) 
                f.create_dataset('inputs', data=all_inputs)
                
                # Guardar metadatos
                f.attrs['input_frames'] = self.n_channels
                f.attrs['output_frames'] = self.n_classes
                f.attrs['timestamp'] = datetime.now().strftime('%Y/%m/%d %H:%M')
                f.attrs['model_type'] = 'unet3'
                
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 10,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                div_factor=10,
                final_div_factor=100
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
