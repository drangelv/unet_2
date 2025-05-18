"""
Script para crear dataset trusted de secuencias NxM a partir del dataset crudo
"""
import os
import h5py
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse

def filter_last_n_months(keys_list, n_months=3):
    """
    Filtra los timestamps para mantener solo los últimos n meses
    
    Args:
        keys_list (list): Lista de timestamps en formato "YYYY_MM_DD_HH_MM_SS"
        n_months (int): Número de meses a mantener
        
    Returns:
        list: Lista filtrada de timestamps
    """
    # Convertir todos los timestamps a objetos datetime
    dates = [datetime.strptime(key, "%Y_%m_%d_%H_%M_%S") for key in keys_list]
    
    # Encontrar la fecha más reciente
    latest_date = max(dates)
    
    # Calcular la fecha límite (3 meses antes)
    cutoff_date = latest_date - timedelta(days=30 * n_months)
    
    # Filtrar las keys
    filtered_keys = [key for key, date in zip(keys_list, dates) if date >= cutoff_date]
    
    print(f"Total de frames originales: {len(keys_list)}")
    print(f"Frames después de filtrar por {n_months} meses: {len(filtered_keys)}")
    print(f"Rango de fechas: {datetime.strptime(filtered_keys[0], '%Y_%m_%d_%H_%M_%S')} a {datetime.strptime(filtered_keys[-1], '%Y_%m_%d_%H_%M_%S')}")
    
    return filtered_keys

def create_trusted_dataset(input_file, output_file, input_frames=12, output_frames=6):
    """
    Crea un dataset trusted con secuencias de 12 frames de entrada y 6 de salida
    
    Args:
        input_file (str): Ruta al archivo H5 crudo
        output_file (str): Ruta donde guardar el dataset trusted
        input_frames (int): Número de frames de entrada (default: 12)
        output_frames (int): Número de frames de salida (default: 6)
    """
    print(f"Creando dataset trusted {input_frames}x{output_frames}...")
    
    # Abrir archivo de entrada
    with h5py.File(input_file, "r") as h5_input:
        # Obtener y ordenar timestamps
        keys_list = list(h5_input.keys())
        keys_list.sort(key=lambda x: datetime.strptime(x, "%Y_%m_%d_%H_%M_%S"))
        
        # Filtrar por últimos 3 meses
        keys_list = filter_last_n_months(keys_list, n_months=3)
        
        # Calcular número de secuencias válidas
        n_sequences = len(keys_list) - (input_frames + output_frames - 1)
        print(f"Se crearán {n_sequences} secuencias válidas")
        
        # Crear archivo de salida
        with h5py.File(output_file, "w") as h5_output:
            # Crear grupos para entrada y salida
            input_group = h5_output.create_group("inputs")
            target_group = h5_output.create_group("targets")
            timestamps_group = h5_output.create_group("timestamps")
            
            # Guardar configuración
            h5_output.attrs['input_frames'] = input_frames
            h5_output.attrs['output_frames'] = output_frames
            h5_output.attrs['created_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5_output.attrs['date_range'] = f"{keys_list[0]} to {keys_list[-1]}"
            
            # Crear secuencias
            for seq_idx in tqdm(range(n_sequences), desc="Creando secuencias"):
                # Obtener frames de entrada
                input_frames_data = []
                input_timestamps = []
                for i in range(input_frames):
                    key = keys_list[seq_idx + i]
                    frame = h5_input[key][:]
                    frame = np.clip(frame, 0, 100)  # Clip valores
                    input_frames_data.append(frame)
                    input_timestamps.append(key)
                
                # Obtener frames objetivo
                target_frames_data = []
                target_timestamps = []
                for i in range(output_frames):
                    key = keys_list[seq_idx + input_frames + i]
                    frame = h5_input[key][:]
                    frame = np.clip(frame, 0, 100)
                    target_frames_data.append(frame)
                    target_timestamps.append(key)
                
                # Guardar secuencia
                seq_name = f"sequence_{seq_idx:05d}"
                input_group.create_dataset(seq_name, data=np.array(input_frames_data))
                target_group.create_dataset(seq_name, data=np.array(target_frames_data))
                
                # Guardar timestamps como atributos
                timestamps_group.create_dataset(
                    f"{seq_name}_input", 
                    data=np.array(input_timestamps, dtype='S20')
                )
                timestamps_group.create_dataset(
                    f"{seq_name}_target", 
                    data=np.array(target_timestamps, dtype='S20')
                )
    
    print(f"Dataset trusted creado exitosamente en: {output_file}")

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Crear dataset trusted a partir de dataset crudo')
    parser.add_argument('--input-frames', type=int, default=12,
                      help='Número de frames de entrada (default: 12)')
    parser.add_argument('--output-frames', type=int, default=6,
                      help='Número de frames de salida (default: 6)')
    parser.add_argument('--input-path', type=str, default="inputs/combined_data_final.h5",
                      help='Ruta al archivo H5 crudo (default: inputs/combined_data_final.h5)')
    parser.add_argument('--output-path', type=str, default="inputs/data_trusted_12x6.h5",
                      help='Ruta donde guardar el dataset trusted')
    args = parser.parse_args()

    # Ajustar ruta de salida según los frames especificados
    if args.output_path == "inputs/data_trusted_12x6.h5":
        args.output_path = f"inputs/data_trusted_{args.input_frames}x{args.output_frames}.h5"
    
    # Crear dataset trusted
    create_trusted_dataset(
        input_file=args.input_path,
        output_file=args.output_path,
        input_frames=args.input_frames,
        output_frames=args.output_frames
    )
