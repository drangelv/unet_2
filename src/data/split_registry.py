"""
Registro de la división de datos para reproducibilidad
"""
import json
from datetime import datetime

# Información sobre la división de datos
DATA_SPLIT_INFO = {
    "dataset_path": "/Users/diego/Documents/Tesis SmaAt/SmaAt-UNet/data/combined_data_final.h5",
    "split_date": "2024-05-17",
    "total_samples": None,  # Se llenará al ejecutar
    "splits": {
        "train": {
            "ratio": 0.7,
            "samples": None,  # Se llenará al ejecutar
            "indices": []  # Se llenará al ejecutar
        },
        "validation": {
            "ratio": 0.15,
            "samples": None,
            "indices": []
        },
        "test": {
            "ratio": 0.15,
            "samples": None,
            "indices": []
        }
    },
    "random_seed": 42,
    "split_method": "random",  # Alternativas: "temporal", "random"
}

def save_split_info(split_info, filepath="data_split_info.json"):
    """Guarda la información de la división de datos"""
    with open(filepath, 'w') as f:
        json.dump(split_info, f, indent=4)

def load_split_info(filepath="data_split_info.json"):
    """Carga la información de la división de datos"""
    with open(filepath, 'r') as f:
        return json.load(f)

def update_split_info(total_samples, train_indices, val_indices, test_indices):
    """Actualiza la información con los índices reales usados"""
    DATA_SPLIT_INFO["total_samples"] = total_samples
    DATA_SPLIT_INFO["splits"]["train"]["samples"] = len(train_indices)
    DATA_SPLIT_INFO["splits"]["train"]["indices"] = train_indices.tolist()
    DATA_SPLIT_INFO["splits"]["validation"]["samples"] = len(val_indices)
    DATA_SPLIT_INFO["splits"]["validation"]["indices"] = val_indices.tolist()
    DATA_SPLIT_INFO["splits"]["test"]["samples"] = len(test_indices)
    DATA_SPLIT_INFO["splits"]["test"]["indices"] = test_indices.tolist()
    DATA_SPLIT_INFO["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Guardar la información actualizada
    save_split_info(DATA_SPLIT_INFO)
