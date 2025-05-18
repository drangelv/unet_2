"""
Módulo para manejo de datos y dataset de heatmaps desde el dataset trusted
"""
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datetime import datetime
from .split_registry import update_split_info, DATA_SPLIT_INFO
from sklearn.model_selection import train_test_split

class TrustedHeatmapDataset(Dataset):
    """Dataset para cargar secuencias desde el dataset trusted"""
    
    def __init__(self, file_path, subset='train', transform=None, 
                 normalize=True, indices=None):
        """
        Args:
            file_path (str): Ruta al archivo H5 trusted
            subset (str): 'train', 'val', o 'test'
            transform: Transformaciones a aplicar
            normalize (bool): Si normalizar los datos
            indices (list): Índices específicos a usar (opcional)
        """
        self.file_path = file_path
        self.transform = transform
        self.normalize = normalize
        self.subset = subset
        
        # Abrir archivo para obtener metadatos
        with h5py.File(self.file_path, "r") as h5_file:
            self.input_frames = h5_file.attrs['input_frames']
            self.output_frames = h5_file.attrs['output_frames']
            self.n_sequences = len(h5_file['inputs'])
        
        # Usar índices específicos o todos
        self.valid_indices = indices if indices is not None else list(range(self.n_sequences))
        print(f"Dataset {subset} creado con {len(self.valid_indices)} secuencias")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_idx = self.valid_indices[idx]
        seq_name = f"sequence_{seq_idx:05d}"
        
        # Cargar secuencia
        with h5py.File(self.file_path, "r") as h5_file:
            # Cargar frames
            input_tensor = torch.tensor(h5_file['inputs'][seq_name][:], dtype=torch.float32)
            target_tensor = torch.tensor(h5_file['targets'][seq_name][:], dtype=torch.float32)
            
            # Cargar timestamps
            input_timestamps = [ts.decode() for ts in h5_file['timestamps'][f"{seq_name}_input"][:]]
            target_timestamps = [ts.decode() for ts in h5_file['timestamps'][f"{seq_name}_target"][:]]
            timestamps = input_timestamps + target_timestamps

        # Normalización
        if self.normalize:
            input_tensor = input_tensor / 100.0
            target_tensor = target_tensor / 100.0

        # Transformaciones
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor, timestamps

class TrustedHeatmapDataModule(pl.LightningDataModule):
    """Módulo de datos para PyTorch Lightning usando el dataset trusted"""
    
    def __init__(self, data_path, batch_size=32, num_workers=4, 
                 train_ratio=0.7, val_ratio=0.15):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        
        # Verificar y cargar metadatos
        with h5py.File(self.data_path, "r") as h5_file:
            self.input_frames = h5_file.attrs['input_frames']
            self.output_frames = h5_file.attrs['output_frames']
        print(f"DataModule configurado para secuencias {self.input_frames}x{self.output_frames}")

    def prepare_data(self):
        """Verifica que el archivo de datos existe"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset trusted no encontrado: {self.data_path}")

    def setup(self, stage=None):
        """Prepara los conjuntos de datos"""
        # Crear dataset completo temporal para obtener índices
        full_dataset = TrustedHeatmapDataset(
            file_path=self.data_path,
            subset='full'
        )
        
        # Dividir índices
        indices = np.arange(len(full_dataset))
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=self.train_ratio,
            random_state=DATA_SPLIT_INFO['random_seed']
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=self.val_ratio/(self.val_ratio + self.test_ratio),
            random_state=DATA_SPLIT_INFO['random_seed']
        )
        
        # Registrar la división
        update_split_info(
            total_samples=len(full_dataset),
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx
        )
        
        # Crear datasets
        self.train_dataset = TrustedHeatmapDataset(
            file_path=self.data_path,
            subset='train',
            indices=train_idx
        )
        
        self.val_dataset = TrustedHeatmapDataset(
            file_path=self.data_path,
            subset='val',
            indices=val_idx
        )
        
        self.test_dataset = TrustedHeatmapDataset(
            file_path=self.data_path,
            subset='test',
            indices=test_idx
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
