import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


## Definir a mesma arquitetura do autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


## Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)

## Carregar o modelo salvo
model_path = "saved_models/best_model_0_0003.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo carregado do epoch {checkpoint['epoch'] + 1} com loss de validação {checkpoint['loss']:.6f}")
else:
    raise FileNotFoundError("Arquivo do modelo não encontrado. Verifique o caminho: ./saved_models/best_model.pth")

## Carregar dataset MNIST (apenas para visualização)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)  # Pegamos 10 imagens aleatórias


## Função para mostrar imagens
def show_images(original, reconstructed, n=20):
    plt.figure(figsize=(90, 36))
    for i in range(n):
        # Imagem original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Imagem reconstruída
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.title("Reconstruída")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


## Gerar visualizações
model.eval()  # Modo de avaliação
with torch.no_grad():
    # Pegar um batch de imagens de teste
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)

    # Gerar reconstruções
    outputs = model(images)

    # Mover para CPU e desnormalizar
    original_images = images.cpu().numpy()
    reconstructed_images = outputs.cpu().numpy()

    # Ajustar a escala das imagens para visualização
    original_images = 0.5 * original_images + 0.5  # Desnormalizar ([-1,1] -> [0,1])
    reconstructed_images = 0.5 * reconstructed_images + 0.5

    # Mostrar as imagens
    print("\nComparação entre imagens originais e reconstruídas:")
    show_images(original_images, reconstructed_images, n=5)  # Mostra 5 pares de imagens

    # Mostrar os valores dos pixels (primeira imagem)
    print("\nValores dos pixels (primeira imagem - 5x5 canto superior esquerdo):")
    print("Original:")
    print(original_images[0, 0, :5, :5])  # Mostra 5x5 pixels
    print("\nReconstruída:")
    print(reconstructed_images[0, 0, :5, :5])