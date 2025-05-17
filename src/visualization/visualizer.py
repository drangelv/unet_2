"""
Módulo para visualización de heatmaps y resultados
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

def setup_custom_cmap():
    """Configura un colormap personalizado para visualización de heatmaps"""
    import matplotlib.colors as mcolors
    
    # Definir colores para el mapa
    colors = ['#FFFFFF', '#A6F28F', '#3DEC3A', '#FFE985', '#FFA647',
              '#F52C2C', '#BC0000', '#700000']
    
    # Crear colormap personalizado
    n_bins = 100
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    return custom_cmap

def visualize_sample(dataset, index, save_dir="visualizaciones", show=False):
    """
    Visualiza una muestra del dataset
    
    Args:
        dataset: Dataset de heatmaps
        index: Índice de la muestra a visualizar
        save_dir: Directorio donde guardar la visualización
        show: Si mostrar la figura además de guardarla
    """
    inputs, target, timestamps = dataset[index]
    
    # Desnormalizar si es necesario
    if inputs.max() <= 1.0:
        inputs = inputs * 100.0
        target = target * 100.0
    
    # Configurar visualización
    n_cols = min(4, inputs.shape[0] + target.shape[0])
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*4, 4))
    
    custom_cmap = setup_custom_cmap()
    
    # Mostrar frames de entrada
    for i in range(min(inputs.shape[0], n_cols-1)):
        im = axes[i].imshow(inputs[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        axes[i].set_title(f'Input {i+1}\n{timestamps[i]}')
        axes[i].axis('off')
    
    # Mostrar frame objetivo
    im = axes[-1].imshow(target[0].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
    axes[-1].set_title(f'Target\n{timestamps[-1]}')
    axes[-1].axis('off')
    
    # Barra de color común
    plt.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                 pad=0.01, fraction=0.05, label='Intensidad')
    
    plt.tight_layout()
    plt.suptitle(f'Muestra {index}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Guardar
    os.makedirs(save_dir, exist_ok=True)
    plt_filename = os.path.join(save_dir, f"sample_{index}.jpg")
    plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Visualización guardada en: {plt_filename}")

def visualize_prediction(model, dataset, index, save_dir="predicciones", show=False):
    """
    Visualiza una predicción del modelo
    
    Args:
        model: Modelo UNet3 entrenado
        dataset: Dataset de test
        index: Índice de la muestra a visualizar
        save_dir: Directorio donde guardar la visualización
        show: Si mostrar la figura además de guardarla
    """
    model.eval()
    inputs, target, timestamps = dataset[index]
    
    # Preparar input
    inputs_batch = inputs.unsqueeze(0).to(model.device)
    
    # Realizar predicción
    with torch.no_grad():
        prediction = model(inputs_batch)
    
    # Mover a CPU
    prediction = prediction.cpu().squeeze(0)
    
    # Desnormalizar si es necesario
    if inputs.max() <= 1.0:
        inputs = inputs * 100.0
        target = target * 100.0
        prediction = prediction * 100.0
    
    # Configurar visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    custom_cmap = setup_custom_cmap()
    
    # Último frame de entrada
    im1 = axes[0].imshow(inputs[-1].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
    axes[0].set_title(f'Último input\n{timestamps[-2]}')
    axes[0].axis('off')
    
    # Predicción
    im2 = axes[1].imshow(prediction[0].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
    axes[1].set_title(f'Predicción\n{timestamps[-1]}')
    axes[1].axis('off')
    
    # Target
    im3 = axes[2].imshow(target[0].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
    axes[2].set_title(f'Target\n{timestamps[-1]}')
    axes[2].axis('off')
    
    # Barra de color común
    plt.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal',
                 pad=0.01, fraction=0.05, label='Intensidad')
    
    plt.tight_layout()
    plt.suptitle(f'Predicción para {timestamps[-1]}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Guardar
    os.makedirs(save_dir, exist_ok=True)
    plt_filename = os.path.join(save_dir, f"prediction_{index}.jpg")
    plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Predicción guardada en: {plt_filename}")
    
    return prediction

def plot_training_history(log_dir, save_dir="resultados"):
    """
    Visualiza el historial de entrenamiento desde los logs
    
    Args:
        log_dir: Directorio con los logs de entrenamiento
        save_dir: Directorio donde guardar las gráficas
    """
    # Esta función requerirá implementación específica según
    # el formato de los logs guardados durante el entrenamiento
    pass
