"""
Modelo que usa el último frame de entrada como predicción
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        x, y, _ = batch  # Ignorar timestamps
        y_hat = self(x)
        test_loss = nn.MSELoss()(y_hat, y).item()
        
        # Calcular métricas como en validation
        y_cpu = y.cpu()
        y_hat_cpu = y_hat.cpu()
        
        threshold = 0.5
        y_binary = (y_cpu > threshold).float()
        y_hat_binary = (y_hat_cpu > threshold).float()
        
        metrics = self.calculate_metrics(y_binary, y_hat_binary)
        
        # Logging
        self.log('test_loss', test_loss)
        self.log('test_mse', metrics['mse'])
        self.log('test_csi', metrics['csi'])
        self.log('test_far', metrics['far'])
        self.log('test_hss', metrics['hss'])
        
        return test_loss
    
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
