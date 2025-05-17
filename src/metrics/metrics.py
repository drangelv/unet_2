"""
Módulo de métricas personalizadas para evaluación del modelo
"""
import torch
import torch.nn.functional as F
from torchmetrics import Metric
import numpy as np

class BinaryMetrics(Metric):
    """Implementación de métricas binarias para predicción de precipitación"""
    
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.add_state("tp", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
    
    def _binarize(self, x):
        return (x > self.threshold).float()
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = self._binarize(preds)
        target = self._binarize(target)
        
        self.tp += torch.sum((preds == 1) & (target == 1))
        self.fp += torch.sum((preds == 1) & (target == 0))
        self.fn += torch.sum((preds == 0) & (target == 1))
        self.tn += torch.sum((preds == 0) & (target == 0))

class CSI(BinaryMetrics):
    """Critical Success Index (Threat Score)"""
    def compute(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-8)

class FAR(BinaryMetrics):
    """False Alarm Rate"""
    def compute(self):
        return self.fp / (self.tp + self.fp + 1e-8)

class HSS(BinaryMetrics):
    """Heidke Skill Score"""
    def compute(self):
        total = self.tp + self.tn + self.fp + self.fn
        random_hits = ((self.tp + self.fn) * (self.tp + self.fp) + 
                      (self.tn + self.fp) * (self.tn + self.fn)) / (total + 1e-8)
        return (self.tp + self.tn - random_hits) / (total - random_hits + 1e-8)

def get_metrics_dict(threshold=0.0, device='cpu'):
    """Retorna un diccionario con todas las métricas configuradas"""
    return {
        'csi': CSI(threshold=threshold).to(device),
        'far': FAR(threshold=threshold).to(device),
        'hss': HSS(threshold=threshold).to(device)
    }

def calculate_metrics(y_pred, y_true, threshold=0.0):
    """
    Calcula todas las métricas para evaluación
    
    Args:
        y_pred: Predicciones del modelo
        y_true: Valores reales
        threshold: Umbral para binarización
    
    Returns:
        Dict con los resultados de las métricas
    """
    metrics = {}
    device = y_pred.device
    
    # MSE
    metrics['mse'] = F.mse_loss(y_pred, y_true)
    
    # Métricas binarias
    binary_metrics = get_metrics_dict(threshold, device)
    for name, metric in binary_metrics.items():
        metric.update(y_pred, y_true)
        metrics[name] = metric.compute()
    
    return metrics
