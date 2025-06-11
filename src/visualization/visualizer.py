"""
Módulo para visualización de heatmaps y resultados
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from config.config import MODEL_CONFIG

def format_timestamp(timestamp):
    """Formatea un timestamp a formato yyyy/mm/dd hh:mm"""
    try:
        # Intentar diferentes formatos de entrada
        formats = ['%Y_%m_%d_%H_%M_%S', '%Y%m%d%H%M', '%Y_%m_%d_%H_%M']
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp, fmt)
                return dt.strftime('%Y/%m/%d %H:%M')
            except ValueError:
                continue
        # Si ningún formato coincide, intentar limpiar el string
        cleaned = timestamp.replace('_', '')
        dt = datetime.strptime(cleaned[:12], '%Y%m%d%H%M')
        return dt.strftime('%Y/%m/%d %H:%M')
    except Exception:
        return timestamp

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
        axes[i].set_title(f'Input {i+1}\n{format_timestamp(timestamps[i])}')
        axes[i].axis('off')
    
    # Mostrar frame objetivo
    im = axes[-1].imshow(target[0].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
    axes[-1].set_title(f'Target\n{format_timestamp(timestamps[-1])}')
    axes[-1].axis('off')
    
    # Barra de color común
    plt.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                 pad=0.01, fraction=0.05, label='Intensidad')
    
    plt.tight_layout()
    plt.suptitle(f'Muestra {index}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Crear subdirectorio con el nombre del modelo
    model_save_dir = os.path.join(save_dir, MODEL_CONFIG['model_name'])
    os.makedirs(model_save_dir, exist_ok=True)
    plt_filename = os.path.join(model_save_dir, f"sample_{index}.jpg")
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
    
    # Función helper para dibujar el rectángulo central
    def draw_central_rectangle(ax, image):
        height, width = image.shape
        center_y, center_x = height // 2, width // 2
        rect_size = 32 // 2  # La mitad del tamaño del rectángulo (32x32)
        rect = plt.Rectangle((center_x - rect_size, center_y - rect_size),
                          32, 32, fill=False, color='black', linewidth=1)
        ax.add_patch(rect)
    
    # Configurar visualización
    n_rows = 4  # 2 filas para inputs, 1 para ground truth, 1 para predicciones
    n_cols = 6  # 6 columnas para cada fila
    
    # Crear figura con espacio extra a la derecha para la colorbar
    fig = plt.figure(figsize=(22, 12))  # Aumentado el alto para 4 filas
    
    # Crear grid para los subplots, dejando espacio para la colorbar
    gs = plt.GridSpec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.2],
                     left=0.05, right=0.95, bottom=0.05, top=0.92,
                     wspace=0.2, hspace=0.4)
    
    # Configurar colormap
    custom_cmap = setup_custom_cmap()
    
    # Configurar el título principal
    last_input_ts = format_timestamp(timestamps[-1])
    plt.suptitle(f'Predicción para {last_input_ts}', y=1.02, fontsize=14)
    
    # Mostrar frames de entrada (primeros 6)
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(inputs[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        draw_central_rectangle(ax, inputs[i].numpy())
        ax.set_title(f'Input {i+1}\n{format_timestamp(timestamps[i])}')
        ax.axis('off')
    
    # Mostrar frames de entrada (últimos 6)
    for i in range(6, 12):
        ax = fig.add_subplot(gs[1, i-6])
        im = ax.imshow(inputs[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        draw_central_rectangle(ax, inputs[i].numpy())
        ax.set_title(f'Input {i+1}\n{format_timestamp(timestamps[i])}')
        ax.axis('off')
    
    # Añadir líneas punteadas horizontales
    fig.add_artist(plt.Line2D([0.05, 0.90], [0.51, 0.51], color='white', linestyle='--', transform=fig.transFigure))
    fig.add_artist(plt.Line2D([0.05, 0.90], [0.26, 0.26], color='white', linestyle='--', transform=fig.transFigure))
    
    # Mostrar frames de ground truth (dinámico según número disponible)
    n_output_frames = target.shape[0]
    for i in range(n_output_frames):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(target[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        draw_central_rectangle(ax, target[i].numpy())
        ax.set_title(f'Ground Truth {i+1}\n{format_timestamp(timestamps[12+i])}')
        ax.axis('off')
    
    # Llenar las columnas restantes con espacios vacíos si hay menos de 6 frames
    for i in range(n_output_frames, 6):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
    
    # Mostrar frames de predicción (dinámico según número disponible)
    for i in range(n_output_frames):
        ax = fig.add_subplot(gs[3, i])
        im = ax.imshow(prediction[i].numpy(), cmap=custom_cmap, vmin=0, vmax=100)
        draw_central_rectangle(ax, prediction[i].numpy())
        ax.set_title(f'Prediction {i+1}\n{format_timestamp(timestamps[12+i])}')
        ax.axis('off')
    
    # Llenar las columnas restantes con espacios vacíos si hay menos de 6 frames
    for i in range(n_output_frames, 6):
        ax = fig.add_subplot(gs[3, i])
        ax.axis('off')
    
    # Añadir colorbar en la última columna del GridSpec
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Intensidad (0-100)', fontsize=10)
    
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
