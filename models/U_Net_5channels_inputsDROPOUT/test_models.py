import torch
import matplotlib.pyplot as plt
import csv
from models import UNetAutoencoder
from models.U_Net_5channels_inputsDROPOUT.norm_denorm import denormalize_3d_tensor
from performance_mesure import evaluate_reconstruction, plot_metrics_comparison
from train_5imagesUNET import load_sequences_from_tensor


def load_timestamps(csv_path):
    with open(csv_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        # Remove cabeçalho e pega apenas a coluna do timestamp (assumindo que é a primeira coluna)
        timestamps = [row[0] for row in lista[1:]]
    return timestamps

# Configurações iniciais
model_name = "CompleteDB_future1stamp_UNetConv_HuberLoss_lr0_0003_RJ_nomalized"
model_path = f"./saved_models/best_model_{model_name}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_stamps_aread = 0

# Carrega os dados
train_loader, val_loader, test_loader, test_dataset, normalization_max_val = load_sequences_from_tensor(
    "../../data/csvs/indices_sequencias_RJ.csv",
    "../../data/tensores/complete/tensor_radar_RJ_precipitacao.pt",
                                                                    top_n= 5,
                                                                    times_aread= n_stamps_aread,
                                                                    batch_size=32)
# Carrega o modelo
model = UNetAutoencoder().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def evaluate_model(loader, model, denormalize=False, max_val=None):
    all_inputs = []
    all_future_register = []
    all_outputs = []

    with torch.no_grad():
        for batch in loader:
            # Garante o formato correto [batch, channels, height, width]
            if isinstance(batch, list):  # Se for retornado como lista pelo DataLoader
                inputs = torch.stack([item[0][1] for item in batch])
                future_registers = torch.stack([item[1][1] for item in batch])

                indices_inputs = [item[0][0] for item in batch]   # Índices dos inputs
                indices_future_registers = [item[1][0] for item in batch] # Índices dos targets
            else:
                print("problema na recepção dos dados no LOADER")
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            if future_registers.dim() == 3:
                future_registers = future_registers.unsqueeze(1)

            inputs = inputs.to(device)
            future_registers = future_registers.to(device)

            outputs = model(inputs)

            if denormalize and max_val is not None:
                inputs = denormalize_3d_tensor(inputs.cpu(), max_val)
                outputs = denormalize_3d_tensor(outputs.cpu(), max_val)
                future_registers = denormalize_3d_tensor(future_registers.cpu(), max_val)
            for i in range(len(indices_inputs)):
                all_inputs.append((indices_inputs[i],inputs[i].cpu()))
                all_outputs.append((indices_future_registers[i],outputs[i].cpu()))
                all_future_register.append((indices_future_registers[i], future_registers[i].cpu()))

    return all_inputs, all_outputs, all_future_register

# Avaliação desnormalizada
x_denorm, y_pred_denorm, y_true_denorm = evaluate_model(test_loader, model, denormalize=True, max_val=normalization_max_val)

predict_metrics_denorm = evaluate_reconstruction(y_true_denorm, y_pred_denorm)
persist_metrics_denorm = evaluate_reconstruction(y_true_denorm, x_denorm)
print("\nMétricas desnormalizadas:")
plot_metrics_comparison(predict_metrics_denorm, persist_metrics_denorm, "UNet", "Persist_model")

# Carrega os timestamps
timestamps = load_timestamps("../../data/csvs/timestampsRJ.csv")

# 2. Encontrar os 5 índices com maiores valores médios em y_true_denorm
# Assumindo que y_true_denorm é uma lista de tuplas (timestamp_idx, image_tensor)
image_means = torch.stack([img.mean() for (idx, img) in y_true_denorm])
top_values, top_indices = torch.topk(image_means, 5)
top_indices = top_indices.numpy()

# 3. Plotagem com timestamps
plt.figure(figsize=(18, 10))
plt.suptitle('Comparação entre Inputs, Previsões e Valores Reais (Top 5 maiores precipitações)', fontsize=16)

for i, idx in enumerate(top_indices):
    # Extrai os dados
    timestamp_str = timestamps[y_true_denorm[idx][0]]  # Pega o timestamp do CSV usando o índice original
    input_img = x_denorm[idx][1].squeeze().numpy()
    pred_img = y_pred_denorm[idx][1].squeeze().numpy()
    true_img = y_true_denorm[idx][1].squeeze().numpy()

    # Valor máximo para escala consistente
    vmax = max(input_img.max(), pred_img.max(), true_img.max())

    # Plot Input
    plt.subplot(3, 5, i + 1)
    plt.imshow(input_img, cmap='viridis', vmin=0, vmax=vmax)
    plt.title(f'Input\n{timestamp_str}', fontsize=10)
    plt.axis('off')
    if i == 0:
        plt.ylabel('x_denorm', rotation=0, labelpad=40, fontsize=12)

    # Plot Prediction
    plt.subplot(3, 5, i + 6)
    plt.imshow(pred_img, cmap='viridis', vmin=0, vmax=vmax)
    plt.title(f'Predição\n{timestamp_str}', fontsize=10)
    plt.axis('off')
    if i == 0:
        plt.ylabel('y_pred_denorm', rotation=0, labelpad=40, fontsize=12)

    # Plot Ground Truth
    plt.subplot(3, 5, i + 11)
    plt.imshow(true_img, cmap='viridis', vmin=0, vmax=vmax)
    plt.title(f'Verdadeiro\n{timestamp_str}', fontsize=10)
    plt.axis('off')
    if i == 0:
        plt.ylabel('y_true_denorm', rotation=0, labelpad=40, fontsize=12)

plt.tight_layout()
plt.show()

# 4. Exibir informações complementares
print("\nValores máximos encontrados nos índices selecionados:")
for i, idx in enumerate(top_indices):
    timestamp_str = timestamps[y_true_denorm[idx][0]]
    print(f"Timestamp: {timestamp_str}")
    print(f"Input max: {x_denorm[idx][1].max():.2f}")
    print(f"Predição max: {y_pred_denorm[idx][1].max():.2f}")
    print(f"Verdadeiro max: {y_true_denorm[idx][1].max():.2f}")
    print("-" * 50)