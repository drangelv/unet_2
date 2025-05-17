"""
Módulo para obtener y mostrar información del sistema
"""
import os
import platform
import torch
import multiprocessing
from config.config import MODEL_CONFIG

def print_system_info():
    """Imprime información detallada del sistema y hardware"""
    print("\n===== INFORMACIÓN DEL SISTEMA =====")
    print(f"Sistema Operativo: {platform.system()} {platform.version()}")
    print(f"Procesador: {platform.processor()}")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Python versión: {platform.python_version()}")
    print(f"PyTorch versión: {torch.__version__}")
    
    print("\n===== CONFIGURACIÓN DE HARDWARE =====")
    if torch.backends.mps.is_available():
        print("Metal Performance Shaders (MPS) disponible")
        print("Dispositivo MPS disponible para aceleración")
        try:
            x = torch.ones(1, device="mps")
            print("  Prueba MPS: ✓ Operación exitosa")
        except:
            print("  Prueba MPS: ✗ Error en operación")
    else:
        print("Metal Performance Shaders (MPS) no disponible")
    
    if torch.cuda.is_available():
        print("\nCUDA disponible")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Versión CUDA: {torch.version.cuda}")
        try:
            x = torch.ones(1, device="cuda")
            print("  Prueba CUDA: ✓ Operación exitosa")
        except:
            print("  Prueba CUDA: ✗ Error en operación")
    else:
        print("\nCUDA no disponible")
        
    print("\n===== CONFIGURACIÓN DE MULTIPROCESSING =====")
    print(f"Método de inicio: {multiprocessing.get_start_method(allow_none=True)}")
    num_cores = os.cpu_count()
    print(f"Núcleos disponibles: {num_cores}")
    print(f"Número de workers recomendado: {min(num_cores - 1, 8)}")
    
    print("\n===== CONFIGURACIÓN DEL MODELO =====")
    print(f"Frames de entrada: {MODEL_CONFIG['input_frames']}")
    print(f"Frames de salida: {MODEL_CONFIG['output_frames']}")
    print(f"Filtros iniciales: {MODEL_CONFIG['initial_filters']}")
    print(f"Upsampling bilinear: {MODEL_CONFIG['bilinear']}")
