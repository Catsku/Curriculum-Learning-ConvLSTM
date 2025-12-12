import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_reconstruction(y_true, y_pred, n_features=1):
    metrics = {}

    # Número de amostras
    n_samples = y_true.size

    # Métricas por pixel
    metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['PSNR'] = 20 * np.log10(y_true.max() / metrics['RMSE'])

    # Cálculo do R² tradicional
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    metrics['R2'] = r2

    # Cálculo do R² Ajustado
    if n_samples > n_features + 1:
        adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    else:
        adjusted_r2 = np.nan  # Evita divisão por zero se n_samples <= n_features + 1
    metrics['Adjusted_R2'] = adjusted_r2

    # Métricas espaciais
    min_side = min(y_true.shape[-2], y_true.shape[-1])  # Pega o menor lado (H ou W)
    win_size = min(7, min_side)  # Não pode ser maior que a imagem
    win_size = win_size if win_size % 2 == 1 else win_size - 1  # Garante que seja ímpar

    metrics['SSIM'] = ssim(
        y_true,
        y_pred,
        data_range=y_true.max() - y_true.min(),
        win_size=win_size,
        channel_axis=1  # Se y_true/y_pred forem [batch, channels, H, W]
    )

    metrics['Spatial_Correlation'] = pearsonr(y_true.flatten(),
                                              y_pred.flatten())[0]

    # Métricas para precipitação (ex.: limiar de 1mm/h)
    threshold = 1.0
    bin_true = y_true >= threshold
    bin_pred = y_pred >= threshold

    tp = np.sum(bin_true & bin_pred)
    fp = np.sum(~bin_true & bin_pred)
    fn = np.sum(bin_true & ~bin_pred)

    metrics['CSI'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    metrics['POD'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['FAR'] = fp / (tp + fp) if (tp + fp) > 0 else 0

    return metrics

def plot_metrics(metrics_dict, model_name="Modelo"):
    """
    Plota gráficos para visualizar as métricas de avaliação.

    Args:
        metrics_dict (dict): Dicionário com as métricas calculadas.
        model_name (str): Nome do modelo para título dos gráficos.
    """
    # Configuração do estilo
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))

    # ----------------------------
    # Gráfico 1: Métricas Básicas
    # ----------------------------
    basic_metrics = ['MSE', 'MAE', 'RMSE', 'PSNR']
    basic_values = [metrics_dict[m] for m in basic_metrics]

    plt.subplot(2, 2, 1)
    sns.barplot(x=basic_metrics, y=basic_values, hue=basic_metrics, palette="Blues_d", legend=False)
    plt.title(f'{model_name} - Métricas de Erro')
    plt.ylabel('Valor')

    # Adiciona os valores nas barras
    for i, v in enumerate(basic_values):
        plt.text(i, v + 0.01 * max(basic_values), f"{v:.4f}", ha='center')

    # ----------------------------
    # Gráfico 2: R² e R² Ajustado
    # ----------------------------
    r2_metrics = ['R2', 'Adjusted_R2']
    r2_values = [metrics_dict[m] for m in r2_metrics]

    plt.subplot(2, 2, 2)
    sns.barplot(x=r2_metrics, y=r2_values, hue=r2_metrics, palette="Greens_d", legend=False)
    plt.ylim(0, 1)
    plt.title(f'{model_name} - Coeficientes de Determinação')
    plt.ylabel('Valor')

    # Linha de referência para R²
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
    plt.text(0.5, 0.72, 'Bom ajuste (R² > 0.7)', color='r', ha='center')

    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

    # ----------------------------
    # Gráfico 3: Métricas Espaciais
    # ----------------------------
    spatial_metrics = ['SSIM', 'Spatial_Correlation']
    spatial_values = [metrics_dict[m] for m in spatial_metrics]

    plt.subplot(2, 2, 3)
    sns.barplot(x=spatial_metrics, y=spatial_values, hue=spatial_metrics, palette="Oranges_d", legend=False)
    plt.ylim(0, 1)
    plt.title(f'{model_name} - Métricas Espaciais')
    plt.ylabel('Valor')

    # Linhas de referência
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Bom (SSIM > 0.8)')
    plt.axhline(y=0.6, color='y', linestyle='--', alpha=0.5, label='Aceitável')
    plt.legend()

    for i, v in enumerate(spatial_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

    # ----------------------------
    # Gráfico 4: Métricas de Eventos
    # ----------------------------
    event_metrics = ['CSI', 'POD', 'FAR']
    event_values = [metrics_dict[m] for m in event_metrics]

    plt.subplot(2, 2, 4)
    bars = sns.barplot(x=event_metrics, y=event_values, hue=event_metrics, palette="Reds_d", legend=False)
    plt.ylim(0, 1)
    plt.title(f'{model_name} - Métricas de Eventos (Threshold = 1mm/h)')
    plt.ylabel('Valor')

    # Linhas de referência
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Bom (CSI > 0.5)')
    plt.axhline(y=0.7, color='b', linestyle='--', alpha=0.5, label='Bom (POD > 0.7)')
    plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Aceitável (FAR < 0.4)')
    plt.legend()

    for i, v in enumerate(event_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.show()
