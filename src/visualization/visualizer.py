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
    
    # Definir colores con transparencia
    colors = [(0, 0, 0, 0),          # Transparente al inicio
             (0, 0, 1, 0.7),        # Azul con 0.7 de alpha en el medio
             (1, 0, 0, 0.7)]        # Rojo con 0.7 de alpha al final
    positions = [0, 0.5, 1]
    
    # Crear colormap personalizado
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "TransparentBlueRed", 
        list(zip(positions, colors)), 
        N=100
    )
    
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
    
    # Mover a CPU y desnormalizar si es necesario
    prediction = prediction.cpu().squeeze(0)
    if prediction.max() <= 1.0:
        prediction = prediction * 100.0
        target = target * 100.0
        inputs = inputs * 100.0
    
    # Desnormalizar si es necesario
    if inputs.max() <= 1.0:
        inputs = inputs * 100.0
        target = target * 100.0
        prediction = prediction * 100.0
    
    # Configurar visualización
    total_plots = 1 + len(prediction) + len(target)  # último input + predicciones + targets
    timestamp_inicio = timestamps[-len(prediction)-1]
    timestamp_fin = timestamps[-1]
    
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(2, total_plots, height_ratios=[1, 1], hspace=0.3)
    
    custom_cmap = setup_custom_cmap()
    
    # Primer fila: Input + Predicciones
    # Último frame de entrada
    ax = plt.subplot(gs[0, 0])
    im = ax.imshow(inputs[-1].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
    ax.set_title(f'Input frame {len(inputs)}\n{timestamps[-len(prediction)-1]}')
    ax.axis('off')
    
    # Predicciones
    for i in range(len(prediction)):
        ax = plt.subplot(gs[0, i+1])
        im = ax.imshow(prediction[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        ax.set_title(f'Prediction {i+1}\n{timestamps[-len(prediction)+i]}')
        ax.axis('off')
    
    # Segunda fila: Targets
    for i in range(len(target)):
        ax = plt.subplot(gs[1, i])
        im = ax.imshow(target[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        ax.set_title(f'Target frame {i+1}\n{timestamps[-len(target)+i]}')
        ax.axis('off')
    
    # Título general
    plt.suptitle(f'Predicción para secuencia {timestamp_inicio} - {timestamp_fin}', 
                y=1.05, fontsize=14)
    
    # Barra de color común, ajustada fuera del área de graficado
    # Los valores [0.92, 0.15, 0.02, 0.7] representan [left, bottom, width, height]
    # donde 0.92 asegura que el colorbar esté fuera del área de las gráficas
    # y no se sobreponga con ninguna de ellas
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Intensidad (0-100)', fontsize=10)
    
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
