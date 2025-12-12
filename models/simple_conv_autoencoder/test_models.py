import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models import SimpleConvAutoencoder24x21
from tensor_dataset import Tensor3dDataset
from models.simple_conv_autoencoder.norm_denorm import normalize_3d_tensor, denormalize_3d_tensor
from performance_mesure import plot_metrics, evaluate_reconstruction

# Configurações iniciais
model_name = "best_model_simpleConv_HuberLoss_lr0_001_RJ_nomalized"
model_path = f"./saved_models/{model_name}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega os dados
tensorPrecip = torch.load('../../data/tensores/complete/tensor_radar_RJ_precipitacao.pt', weights_only=True)

dataset = Tensor3dDataset(tensorPrecip, transform=normalize_3d_tensor)
normalization_max_val = dataset.max_val

# Divisão do dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Carrega o modelo
model = SimpleConvAutoencoder24x21().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Função para avaliação completa
def evaluate_model(loader, model, denormalize=False, max_val=None):
    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for inputs in loader:
            # Garante o formato correto [batch, 1, height, width]
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)

            outputs = model(inputs)

            if denormalize and max_val is not None:
                inputs = denormalize_3d_tensor(inputs.cpu(), max_val)
                outputs = denormalize_3d_tensor(outputs.cpu(), max_val)

            all_inputs.append(inputs.cpu())
            all_outputs.append(outputs.cpu())

    return torch.cat(all_inputs), torch.cat(all_outputs)


# Avaliação normalizada
y_true_norm, y_pred_norm = evaluate_model(test_loader, model, denormalize=False)
metrics_norm = evaluate_reconstruction(y_true_norm.numpy(), y_pred_norm.numpy(), n_features=1)
print("Métricas normalizadas:")
plot_metrics(metrics_norm)

# Avaliação desnormalizada
y_true_denorm, y_pred_denorm = evaluate_model(test_loader, model, denormalize=True, max_val=normalization_max_val)
metrics_denorm = evaluate_reconstruction(y_true_denorm.numpy(), y_pred_denorm.numpy(), n_features=1)
print("\nMétricas desnormalizadas:")
plot_metrics(metrics_denorm)
# Visualização de exemplos
test_inputs = next(iter(test_loader))
if test_inputs.dim() == 3:
    test_inputs = test_inputs.unsqueeze(1)
test_inputs = test_inputs.to(device)

with torch.no_grad():
    test_outputs = model(test_inputs)

# Desnormaliza para visualização
inputs_vis = denormalize_3d_tensor(test_inputs.cpu(), normalization_max_val)
outputs_vis = denormalize_3d_tensor(test_outputs.cpu(), normalization_max_val)

# Converte para numpy e remove dimensão do canal
inputs_np = inputs_vis.numpy()[:, 0]  # Formato [batch, height, width]
outputs_np = outputs_vis.numpy()[:, 0]

# Filtra apenas amostras com precipitação > 1 em pelo menos um pixel
valid_indices = [i for i in range(inputs_np.shape[0])
                if np.any(inputs_np[i] > 1.0)]
n_samples = min(20, len(valid_indices))  # Máximo de 20 amostras válidas

if n_samples == 0:
    print("Nenhuma amostra com precipitação > 1 encontrada.")

    # Prepara para plotagem
    n_samples = min(20, inputs_vis.shape[0])  # Máximo de 20 amostras
    inputs_vis = inputs_vis.numpy()[:, 0]  # Remove dimensão do canal
    outputs_vis = outputs_vis.numpy()[:, 0]

    # Configuração da plotagem
    plt.figure(figsize=(20, 10))
    plt.suptitle('Comparação entre Inputs Originais e Outputs Preditos', y=1.02, fontsize=16)

    # Plot dos inputs
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(inputs_vis[i], cmap='viridis', vmin=0, vmax=inputs_vis.max())
        plt.axis('off')
        if i == 0:
            plt.ylabel('Inputs', rotation=0, labelpad=40, fontsize=12)

    # Plot dos outputs
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(outputs_vis[i], cmap='viridis', vmin=0, vmax=inputs_vis.max())
        plt.axis('off')
        if i == 0:
            plt.ylabel('Outputs', rotation=0, labelpad=40, fontsize=12)

    plt.tight_layout()
    plt.show()
else:
    # Configuração da plotagem
    plt.figure(figsize=(20, 10))
    plt.suptitle('Comparação entre Inputs Originais e Outputs Preditos (apenas amostras com precipitação > 1)',
                 y=1.02, fontsize=16)

    # Plot dos inputs
    for plot_idx, data_idx in enumerate(valid_indices[:n_samples]):
        plt.subplot(2, n_samples, plot_idx + 1)
        plt.imshow(inputs_np[data_idx], cmap='viridis', vmin=0, vmax=inputs_np.max())
        plt.axis('off')
        if plot_idx == 0:
            plt.ylabel('Inputs', rotation=0, labelpad=40, fontsize=12)

    # Plot dos outputs
    for plot_idx, data_idx in enumerate(valid_indices[:n_samples]):
        plt.subplot(2, n_samples, plot_idx + n_samples + 1)
        plt.imshow(outputs_np[data_idx], cmap='viridis', vmin=0, vmax=inputs_np.max())
        plt.axis('off')
        if plot_idx == 0:
            plt.ylabel('Outputs', rotation=0, labelpad=40, fontsize=12)

    plt.tight_layout()
    plt.show()