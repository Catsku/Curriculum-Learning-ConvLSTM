import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import math


def nash_sutcliffe_efficiency_3d(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calcula o Nash-Sutcliffe Efficiency (NSE) para tensores 3D (batch, height, width).
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    numerator = np.sum((y_true - y_pred) ** 2)
    y_true_mean = np.mean(y_true)
    denominator = np.sum((y_true - y_true_mean) ** 2)

    if denominator == 0:
        return 0

    return 1 - (numerator / denominator)

def evaluate_predictions(y_true, y_pred):
    """
    Avalia as previsão, calculando as métricas.
    Retorna métricas detalhadas por amostra.
    """
    metrics = {}

    # Extrair os tensores e arrays das tuplas
    y_true_arrays = []
    for item in y_true:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            tensor = item[1]  # Segundo elemento é o tensor
        else:
            tensor = item

        if torch.is_tensor(tensor):
            y_true_arrays.append(tensor.cpu().detach().numpy())
        else:
            y_true_arrays.append(tensor)

    y_pred_arrays = []
    for item in y_pred:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            tensor = item[1]  # Segundo elemento é o tensor
        else:
            tensor = item

        if torch.is_tensor(tensor):
            y_pred_arrays.append(tensor.cpu().detach().numpy())
        else:
            y_pred_arrays.append(tensor)

    n_samples = len(y_true_arrays)

    temporal_metrics = {
        'mse': [],
        'mae': [],
        'rmse': [],
        'r2': [],
        'nash_sutcliffe_efficiency': [],
        'ssim': [],
        'correlation_pearsonr': [],
        'max_true': [],
        'max_pred': []
    }

    for i in range(n_samples):
        y_true_i = y_true_arrays[i]
        y_pred_i = y_pred_arrays[i]

        # VERIFICAR se são realmente arrays numpy
        if not isinstance(y_true_i, np.ndarray) or not isinstance(y_pred_i, np.ndarray):
            print(f"Aviso: Item {i} não é numpy array. True type: {type(y_true_i)}, Pred type: {type(y_pred_i)}")
            # Pular amostras inválidas
            for key in temporal_metrics.keyxs():
                temporal_metrics[key].append(np.nan)
            continue

        # Remover dimensão de canal se necessário
        if y_true_i.ndim == 3 and y_true_i.shape[0] == 1:  # [1, height, width]
            y_true_i = np.squeeze(y_true_i, axis=0)
        if y_pred_i.ndim == 3 and y_pred_i.shape[0] == 1:  # [1, height, width]
            y_pred_i = np.squeeze(y_pred_i, axis=0)

        # VERIFICAÇÃO DE VALORES VÁLIDOS
        if np.any(np.isnan(y_true_i)) or np.any(np.isnan(y_pred_i)):
            # Pular amostras com NaN
            for key in temporal_metrics.keys():
                temporal_metrics[key].append(np.nan)
            continue

        # Métricas básicas
        mse = np.mean((y_true_i - y_pred_i) ** 2)
        mae = np.mean(np.abs(y_true_i - y_pred_i))
        rmse = np.sqrt(mse)

        temporal_metrics['mse'].append(mse)
        temporal_metrics['mae'].append(mae)
        temporal_metrics['rmse'].append(rmse)

        # R² CORRETO
        numerador2 = np.sum((y_true_i - np.mean(y_true_i))*(y_pred_i - np.mean(y_pred_i)))
        denominador2 = math.sqrt(np.sum((y_true_i - np.mean(y_true_i)) ** 2)) * math.sqrt(np.sum((y_pred_i - np.mean(y_pred_i)) ** 2))
        r2 = (numerador2 / denominador2)**2 if denominador2 > 0 else 0
        temporal_metrics['r2'].append(r2)

        # NASH SUTCLIFFE EFFICIENCY
        nse = nash_sutcliffe_efficiency_3d(y_true_i, y_pred_i)
        temporal_metrics['nash_sutcliffe_efficiency'].append(nse)

        # SSIM com tratamento de erro
        try:
            if np.all(y_true_i == y_true_i[0, 0]) or np.all(y_pred_i == y_pred_i[0, 0]):
                ssim_val = 0
            else:
                min_side = min(y_true_i.shape[0], y_true_i.shape[1])
                win_size = min(7, min_side)
                win_size = win_size if win_size % 2 == 1 else win_size - 1
                data_range = max(y_true_i.max() - y_true_i.min(), 1e-6)
                ssim_val = ssim(y_true_i, y_pred_i, data_range=data_range, win_size=win_size)
        except:
            ssim_val = 0

        temporal_metrics['ssim'].append(ssim_val)

        # Correlação de Pearson
        try:
            if np.all(y_true_i == y_true_i[0, 0]) or np.all(y_pred_i == y_pred_i[0, 0]):
                corr = np.nan
            else:
                corr, _ = pearsonr(y_true_i.flatten(), y_pred_i.flatten())
        except:
            corr = np.nan

        temporal_metrics['correlation_pearsonr'].append(corr)

        # Valores máximos
        temporal_metrics['max_true'].append(y_true_i.max())
        temporal_metrics['max_pred'].append(y_pred_i.max())

    # Adicionar todas as métricas por amostra
    metrics.update(temporal_metrics)

    # Calcular médias
    for key in temporal_metrics.keys():
        valid_values = [x for x in temporal_metrics[key] if not np.isnan(x)]
        metrics[f'{key}_mean'] = np.nanmean(valid_values) if valid_values else np.nan

    return metrics

def plot_metrics_comparison(metrics_dict1, metrics_dict2, model_name1, model_name2="Persistência", model_name_dir='semNome'):
    """
    Plota comparação detalhada de métricas entre dois modelos com pontos por amostra.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 12))

    # Métricas para comparar
    metrics_to_compare = ['mse', 'mae', 'rmse', 'r2', 'nash_sutcliffe_efficiency', 'ssim', 'correlation_pearsonr']

    for i, metric in enumerate(metrics_to_compare, 1):
        plt.subplot(3, 3, i)

        # Preparar dados
        data1 = metrics_dict1.get(metric, [])
        data2 = metrics_dict2.get(metric, [])

        if data1 and data2:
            # Criar dados para scatter plot
            x_positions = np.concatenate([
                np.ones(len(data1)),  # Modelo 1
                np.ones(len(data2)) * 2  # Modelo 2
            ])

            y_values = np.concatenate([data1, data2])
            colors = ['blue'] * len(data1) + ['red'] * len(data2)

            plt.scatter(x_positions, y_values, alpha=0.6, c=colors, s=50)

            # Boxplot sobreposto
            boxplot_data = [data1, data2]
            box = plt.boxplot(boxplot_data, positions=[1, 2], widths=0.3, patch_artist=True)

            # Colorir boxes
            for patch, color in zip(box['boxes'], ['lightblue', 'lightcoral']):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            plt.xticks([1, 2], [model_name1, model_name2])
            plt.title(f'{metric.upper()} - Comparação por Amostra')
            plt.ylabel(metric)

            # Adicionar valores médios
            mean1 = np.nanmean(data1)
            mean2 = np.nanmean(data2)
            plt.axhline(y=mean1, color='darkblue', linestyle='--', linewidth=2,
                        label=f'{model_name1} Mean: {mean1:.3f}')
            plt.axhline(y=mean2, color='darkred', linestyle='--', linewidth=2, label=f'{model_name2} Mean: {mean2:.3f}')
            plt.legend()

    plt.tight_layout()
    # ⭐⭐ CONFIGURAÇÃO OTIMIZADA PARA MÉTRICAS
    plot_name = f"{model_name_dir}/evaluation_metrics.png"
    plt.savefig(plot_name,
                bbox_inches='tight',
                dpi=300,
                facecolor='white',  # Fundo branco para melhor impressão
                edgecolor='none',  # Sem bordas
                pad_inches=0.1)  # Pouco padding
    plt.close()


def plot_comprehensive_comparison(y_true_denorm, y_pred_denorm, x_denorm, timestamps, top_n=3, n_inputs=5,
                                  random_seed=42, model_name_dir='semNome'):
    """
    Plota comparação abrangente com sequências selecionadas aleatoriamente.
    Usa escala de cores baseada apenas nos registros que serão exibidos.
    """
    # Configurar seed para reprodutibilidade
    np.random.seed(random_seed)

    # Selecionar sequências aleatoriamente
    n_samples = len(y_true_denorm)
    if top_n > n_samples:
        top_n = n_samples
        print(f"Aviso: Reduzindo top_n para {n_samples} (número total de amostras)")

    # Selecionar índices aleatórios únicos
    random_indices = np.random.choice(n_samples, size=top_n, replace=False)

    print(f"Exibindo {top_n} sequências aleatórias: índices {random_indices}")

    # ⭐⭐ CALCULAR ESCALA APENAS DOS REGISTROS QUE SERÃO EXIBIDOS
    all_values = []

    # Coletar valores apenas das amostras selecionadas
    for sample_idx in random_indices:
        # Inputs da amostra selecionada
        input_frames = x_denorm[sample_idx]
        for frame_idx, frame_tensor in input_frames:
            all_values.append(frame_tensor.squeeze().numpy())

        # Predição da amostra selecionada
        pred_idx, pred_tensor = y_pred_denorm[sample_idx]
        all_values.append(pred_tensor.squeeze().numpy())

        # Target da amostra selecionada
        true_idx, true_tensor = y_true_denorm[sample_idx]
        all_values.append(true_tensor.squeeze().numpy())

    # Concatenar todos os valores das amostras selecionadas
    all_values_concat = np.concatenate([arr.flatten() for arr in all_values])
    vmin_global = np.min(all_values_concat)
    vmax_global = np.max(all_values_concat)

    print(f"Escala dos {top_n} registros exibidos: {vmin_global:.2f} a {vmax_global:.2f} mm")

    # ⭐⭐ CORREÇÃO: Criar figura com espaço para a barra de cores
    fig = plt.figure(figsize=(25, 4 * top_n))

    # Criar grid com espaço para barra de cores
    gs = fig.add_gridspec(top_n, 8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.05])

    # Criar subplots
    axes = []
    for row in range(top_n):
        row_axes = []
        for col in range(7):  # 7 colunas para imagens
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Adicionar barra de cores no último grid
    cbar_ax = fig.add_subplot(gs[:, 7])  # Ocupa todas as linhas da última coluna

    fig.suptitle(f'Comparação Detalhada: {n_inputs} Inputs + Predição + Target ({top_n} sequências aleatórias)\n'
                 f'Escala: {vmin_global:.1f} a {vmax_global:.1f} mm',
                 fontsize=16, fontweight='bold')

    img = None  # Para referência da barra de cores

    for row, sample_idx in enumerate(random_indices):
        # Extrair dados da amostra
        true_idx, true_tensor = y_true_denorm[sample_idx]
        pred_idx, pred_tensor = y_pred_denorm[sample_idx]
        input_frames = x_denorm[sample_idx]  # Lista de frames de input

        # Plotar os frames de input
        for col in range(n_inputs):
            if col < len(input_frames):
                frame_idx, frame_tensor = input_frames[col]
                frame_data = frame_tensor.squeeze()

                ax = axes[row][col]

                # ⭐⭐ USAR ESCALA DOS REGISTROS EXIBIDOS
                img = ax.imshow(frame_data, cmap='viridis', vmin=vmin_global, vmax=vmax_global)

                # CORRIGIR TÍTULO - mostrar timestamp correto
                timestamp_str = timestamps[int(frame_idx)] if frame_idx < len(timestamps) else f"Idx {frame_idx}"
                ax.set_title(f'{timestamp_str}', fontsize=9, pad=5)
                ax.axis('off')

                # Adicionar valor máximo local
                max_val = frame_data.max()
                ax.text(0.95, 0.05, f'max: {max_val:.1f}',
                        transform=ax.transAxes, fontsize=8, color='white',
                        ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

        # Plotar predição (coluna n_inputs)
        ax = axes[row][n_inputs]
        pred_data = pred_tensor.squeeze()

        # ⭐⭐ USAR ESCALA DOS REGISTROS EXIBIDOS
        img = ax.imshow(pred_data, cmap='viridis', vmin=vmin_global, vmax=vmax_global)

        # CORRIGIR TÍTULO
        pred_timestamp = timestamps[int(pred_idx)] if pred_idx < len(timestamps) else f"Idx {pred_idx}"
        ax.set_title(f'Predição\n{pred_timestamp}', fontsize=10, color='blue', pad=5)
        ax.axis('off')

        max_pred = pred_data.max()
        ax.text(0.95, 0.05, f'max: {max_pred:.1f}',
                transform=ax.transAxes, fontsize=8, color='white',
                ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

        # Plotar target (coluna n_inputs + 1)
        ax = axes[row][n_inputs + 1]
        true_data = true_tensor.squeeze()

        # ⭐⭐ USAR ESCALA DOS REGISTROS EXIBIDOS
        img = ax.imshow(true_data, cmap='viridis', vmin=vmin_global, vmax=vmax_global)

        # CORRIGIR TÍTULO
        true_timestamp = timestamps[int(true_idx)] if true_idx < len(timestamps) else f"Idx {true_idx}"
        ax.set_title(f'Target\n{true_timestamp}', fontsize=10, color='green', pad=5)
        ax.axis('off')

        max_true = true_data.max()
        ax.text(0.95, 0.05, f'max: {max_true:.1f}',
                transform=ax.transAxes, fontsize=8, color='white',
                ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

        # Adicionar número da amostra na primeira coluna
        axes[row][0].set_ylabel(f'Amostra {sample_idx + 1}', fontsize=12, fontweight='bold',
                                rotation=0, ha='right', va='center')

    # ⭐⭐ ADICIONAR LABELS DAS COLUNAS DINAMICAMENTE
    column_labels = []
    for i in range(n_inputs):
        column_labels.append(f'Input t-{n_inputs - 1 - i}')
    column_labels.extend(['Predição', 'Target'])

    for col, label in enumerate(column_labels):
        # Usar a primeira linha para os títulos das colunas
        axes[0][col].set_title(label, fontsize=11, fontweight='bold', pad=12)

    # ⭐⭐ CORREÇÃO: Barra de cores integrada ao layout
    if img is not None:
        cbar = fig.colorbar(img, cax=cbar_ax, label='Precipitação (mm)')
        cbar.set_label('Precipitação (mm)', fontsize=12, fontweight='bold')

    # ⭐⭐ REMOVER ajustes manuais - tight_layout agora funciona
    plt.tight_layout()

    # CONFIGURAÇÃO OTIMIZADA PARA VISUALIZAÇÃO DE IMAGENS
    plot_filename = f"{model_name_dir}/input_predict_visualization.png"
    plt.savefig(plot_filename,
                bbox_inches='tight',
                dpi=200,
                facecolor='white',
                edgecolor='none')

    plt.close()
    print(f"Visualização salva em: {plot_filename}")
