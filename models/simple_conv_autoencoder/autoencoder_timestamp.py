import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import UNetAutoencoder
from tensor_dataset import Tensor3dDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import pandas as pd

# Configurações iniciais
model_name = "UNetAutoencoder_lr_0_0005"
os.makedirs('saved_models', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Função para exibir cabeçalho das épocas
def print_epoch_header(epoch, num_epochs):
    header = f" Epoch {epoch + 1}/{num_epochs} "
    print("\n" + "=" * 60)
    print(header.center(60, '='))
    print("=" * 60)


# Inicialização do modelo
model = UNetAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Carregamento dos dados
tensorPrecipSP = torch.load('../../data/tensores/complete/tensor_radar_SP_precipitacao.pt', weights_only=True)
timestamp_csv_path = '../../data/csvs/precipitacao_bacia_tamanduatei_sp_2019_2023_10min.csv'

# Carrega os timestamps separadamente
timestamps_df = pd.read_csv(timestamp_csv_path)
all_timestamps = timestamps_df['datahora'].tolist()

print("\nInformações sobre os dados:")
print(f"Total de timestamps: {len(all_timestamps)}")
print(f"Primeiros 5 timestamps: {all_timestamps[:5]}")
print(f"Formato do tensor de entrada: {tensorPrecipSP.shape}")

# Cria o dataset sem os timestamps
dataset = Tensor3dDataset(tensorPrecipSP)

# Divisão do dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Parâmetros de treinamento
num_epochs = 500
best_loss = float('inf')
patience = 10
patience_counter = 0
early_stop = False

# Listas para armazenamento
train_losses = []
val_losses = []

# Loop de treinamento principal
for epoch in range(num_epochs):
    if early_stop:
        break

    print_epoch_header(epoch, num_epochs)

    # Fase de treinamento
    model.train()
    train_loss = 0.0
    train_progress = tqdm(train_loader, desc='Training', leave=False)

    for batch in train_progress:
        # Certifique-se de que o batch é um tensor
        if isinstance(batch, (list, list)):
            images = batch[0]  # Assume que as imagens estão no primeiro elemento

        else:
            images = batch
        # Verifica e ajusta a dimensionalidade se necessário
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Adiciona dimensão de canal se necessário

        images = images.float().to(device)  # Converte para float e envia para o dispositivo

        optimizer.zero_grad()
        outputs = model(images)
        #print(f'Output dim: {outputs.size()} \n Images dim: {images.size()}')
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_progress.set_postfix({'train_batch_loss': f'{loss.item():.6f}'})

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Fase de validação
    model.eval()
    val_loss = 0.0
    val_progress = tqdm(test_loader, desc='Validating', leave=False)

    with torch.no_grad():
        for batch in val_progress:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            if images.dim() == 3:
                images = images.unsqueeze(1)

            images = images.float().to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item() * images.size(0)
            val_progress.set_postfix({'val_batch_loss': f'{loss.item():.6f}'})

    val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(val_loss)

    # Resumo da época
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Training Loss:   {train_loss:.6f}")
    print(f"  Validation Loss: {val_loss:.6f}")
    print(f"  Best Val Loss:   {best_loss:.6f}")

    # Lógica de early stopping
    if val_loss < best_loss:
        print(f"\n  Validation loss improved from {best_loss:.6f} to {val_loss:.6f}")
        print("  Saving model...")
        best_loss = val_loss
        patience_counter = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f'./saved_models/best_model_{model_name}.pth')
    else:
        patience_counter += 1
        print(f"\n  No improvement in validation loss ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("  Early stopping triggered!")
            early_stop = True

    if train_loss < 0.8 * val_loss:
        print("  Warning: Potential overfitting detected")

    print("-" * 60)

# Plotagem dos filtros_sequencias_outliers
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

# Carregar melhor modelo para avaliação
if early_stop:
    print('\nLoading the best model saved during training...')
    checkpoint = torch.load(f'./saved_models/best_model_{model_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Best model was saved at epoch {checkpoint["epoch"] + 1} with validation loss {checkpoint["loss"]:.6f}')

# Visualização dos filtros_sequencias_outliers
model.eval()
with torch.no_grad():
    test_iter = iter(test_loader)
    batch = next(test_iter)

    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch

    if images.dim() == 3:
        images = images.unsqueeze(1)

    images = images.float().to(device)
    outputs = model(images)

    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    # Obtém os índices originais do test_dataset
    test_indices = test_dataset.indices
    # Seleciona os primeiros 10 timestamps correspondentes
    selected_timestamps = [all_timestamps[i] for i in test_indices[:10]]


    def denormalize(tensor):
        return tensor * 0.5 + 0.5


    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 6))
    plt.suptitle('Timestamps Originais vs Reconstruídos', y=1.02)

    for idx in range(10):  # Mostra apenas as primeiras 10 imagens
        original = denormalize(images[idx].squeeze())
        reconstructed = denormalize(outputs[idx].squeeze())

        axes[0, idx].imshow(original, cmap='viridis')
        axes[0, idx].set_title(selected_timestamps[idx], fontsize=8)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(reconstructed, cmap='viridis')
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.show()