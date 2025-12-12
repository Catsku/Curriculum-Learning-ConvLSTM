import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


# Define the Autoencoder architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: 16x14x14
            nn.ReLU(True), #max(0,x)  True parameter indicates the value replacement on the original tensor
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: 32x7x7
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),  # Output: 64x1x1
            nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # Output: 32x7x7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: 16x14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # Output: 1x28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#MODEL NAME
model_name="0_0003"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = ConvAutoencoder().to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True) #Shuffle the data
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training parameters
num_epochs = 50
best_loss = float('inf')
patience = 5  # Number of epochs to wait before stopping
patience_counter = 0
early_stop = False

# Create directory for saving models
os.makedirs('saved_models', exist_ok=True)

# Lists to store losses for visualization
train_losses = []
val_losses = []

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()  #Set model to training mode
    train_loss = 0.0

    # Training phase
    for data in train_loader:
        images, _ = data
        images = images.to(device)

        optimizer.zero_grad() #restart optimizer gradient
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward() #calcula gradiente
        optimizer.step() #aplica gradiente nos pesos considerando o learning rate (definido na linha 50)

        train_loss += loss.item() * images.size(0)

    # Calculate average training loss and save
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    # #Desativa dropOut e BatchNorm usa estatistica detodo o dataset
    val_loss = 0.0
    with torch.no_grad():
        #Run images through the model and calculates val_loss
        for data in test_loader:
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item() * images.size(0)

    # Calculate average validation loss and save
    val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(val_loss)

    # Print epoch statistics
    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')


# ------------ Early-stop ------------
    # Check for improvement in validation loss
    if val_loss < best_loss:
        print(f'Validation loss improved from {best_loss:.6f} to {val_loss:.6f}. Saving model...')
        best_loss = val_loss
        patience_counter = 0  # Reset patience counter

        # Save the best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f'./saved_models/best_model_{model_name}.pth')
    else:
        patience_counter += 1
        print(f'No improvement in validation loss for {patience_counter}/{patience} epochs')

        # Check for early stopping condition
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs!')
            early_stop = True
            break

    # Check for overfitting (train loss much lower than validation loss)
    if train_loss < 0.8 * val_loss:
        print('Warning: Potential overfitting detected (train loss significantly lower than validation loss)')

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

# Load the best model for evaluation
if early_stop:
    print('Loading the best model saved during training...')
    checkpoint = torch.load('./saved_models/best_model.pth',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_epoch = checkpoint['epoch']
    print(f'Best model was saved at epoch {best_epoch + 1} with validation loss {checkpoint["loss"]:.6f}')

# Test the autoencoder with the best model
model.eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    outputs = model(images)

    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()


    # Adicione esta função para desnormalizar
    def denormalize(tensor):
        return tensor * 0.5 + 0.5  # Reverte de [-1, 1] para [0, 1]


    # Modifique o bloco de plotagem:
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for idx in range(10):
        original = denormalize(images[idx].squeeze())
        reconstructed = denormalize(outputs[idx].squeeze())
        axes[0, idx].imshow(original, cmap='gray')
        axes[0, idx].axis('off')
        axes[1, idx].imshow(reconstructed, cmap='gray')
        axes[1, idx].axis('off')
    plt.show()
    # Plot original and reconstructed images
"""    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for idx in range(10):
        axes[0, idx].imshow(images[idx].squeeze(), cmap='gray')
        axes[0, idx].axis('off')
        axes[1, idx].imshow(outputs[idx].squeeze(), cmap='gray')
        axes[1, idx].axis('off')
    plt.show()"""