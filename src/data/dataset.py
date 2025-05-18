"""
Módulo para manejo de datos y dataset de heatmaps
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

class HeatmapDataset(Dataset):
    """Dataset para cargar y procesar heatmaps desde archivo H5"""
    
    def __init__(self, file_path, input_frames, output_frames, subset='train', 
                 transform=None, normalize=True, indices=None):
        """
        Args:
            file_path (str): Ruta al archivo H5
            input_frames (int): Número de frames de entrada
            output_frames (int): Número de frames de salida
            subset (str): 'train', 'val', o 'test'
            transform: Transformaciones a aplicar
            normalize (bool): Si normalizar los datos
            indices (list): Índices específicos a usar (opcional)
        """
        self.file_path = file_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.transform = transform
        self.normalize = normalize
        self.subset = subset
        self.timestamps = []

        # Obtener lista de keys y ordenarlas
        with h5py.File(self.file_path, "r") as h5_file:
            self.keys_list = list(h5_file.keys())
            
        # Ordenar por timestamp
        self.keys_list.sort(key=lambda x: datetime.strptime(x, "%Y_%m_%d_%H_%M_%S"))
        
        # Guardar timestamps formateados
        for key in self.keys_list:
            dt = datetime.strptime(key, "%Y_%m_%d_%H_%M_%S")
            self.timestamps.append(dt.strftime("%Y-%m-%d %H:%M"))
            
        # Usar índices específicos o calcular índices válidos
        if indices is not None:
            self.valid_indices = indices
        else:
            self.valid_indices = list(range(len(self.keys_list) - 
                                          (input_frames + output_frames - 1)))
            
        print(f"Dataset {subset} creado con {len(self.valid_indices)} muestras válidas")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_idx = self.valid_indices[idx]

        # Cargar secuencia
        # Aquí se carga la data cruda desde el archivo HDF5
        with h5py.File(self.file_path, "r") as h5_file:
            # Frames de entrada
            input_frames = []
            for i in range(self.input_frames):
                key = self.keys_list[start_idx + i]
                frame = h5_file[key][:]
                frame = np.clip(frame, 0, 100)  # Clip valores
                input_frames.append(frame)

            # Frames objetivo
            target_frames = []
            for i in range(self.output_frames):
                key = self.keys_list[start_idx + self.input_frames + i]
                frame = h5_file[key][:]
                frame = np.clip(frame, 0, 100)
                target_frames.append(frame)

        # Convertir a tensores
        input_tensor = torch.tensor(np.array(input_frames), dtype=torch.float32)
        if self.output_frames == 1:
            target_tensor = torch.tensor(target_frames[0], dtype=torch.float32).unsqueeze(0)
        else:
            target_tensor = torch.tensor(np.array(target_frames), dtype=torch.float32)

        # Normalización
        if self.normalize:
            input_tensor = input_tensor / 100.0
            target_tensor = target_tensor / 100.0

        # Transformaciones
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        # Obtener timestamps para esta secuencia
        sequence_timestamps = self.timestamps[start_idx:start_idx + 
                                           self.input_frames + self.output_frames]

        return input_tensor, target_tensor, sequence_timestamps

class HeatmapDataModule(pl.LightningDataModule):
    """Módulo de datos para PyTorch Lightning"""
    
    def __init__(self, data_path, batch_size=32, input_frames=4, output_frames=1,
                 num_workers=4, train_ratio=0.7, val_ratio=0.15):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Validar número de frames
        if input_frames <= 0:
            raise ValueError(f"input_frames debe ser positivo, se recibió: {input_frames}")
        if output_frames <= 0:
            raise ValueError(f"output_frames debe ser positivo, se recibió: {output_frames}")
            
        self.input_frames = input_frames
        self.output_frames = output_frames
        print(f"DataModule configurado con {input_frames} frames de entrada y {output_frames} frames de salida")
        
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

    def prepare_data(self):
        """Verifica que el archivo de datos existe"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Archivo de datos no encontrado: {self.data_path}")

    def setup(self, stage=None):
        """Prepara los conjuntos de datos"""
        # Crear dataset completo temporal para obtener índices
        full_dataset = HeatmapDataset(
            file_path=self.data_path,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
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
        self.train_dataset = HeatmapDataset(
            file_path=self.data_path,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
            subset='train',
            indices=train_idx
        )
        
        self.val_dataset = HeatmapDataset(
            file_path=self.data_path,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
            subset='val',
            indices=val_idx
        )
        
        self.test_dataset = HeatmapDataset(
            file_path=self.data_path,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
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
