import torch.nn as nn


class SimpleConvAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=(1, 1)),  # 16x31x30
            nn.ReLU(True), #max(0,x)
            nn.MaxPool2d(2),  # 16x15x15
            nn.Conv2d(16, 32, 3, padding=(1, 1)),  # 32x15x15
            nn.ReLU(True),
            nn.MaxPool2d(2)  # 32x7x7
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Primeiro estágio: Expansão de 7x7 para 14x14
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x14x14
            nn.ReLU(True),

            # Segundo estágio: Expansão de 14x14 para 28x28
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x28x28
            nn.ReLU(True),

            # Ajuste final para 31x30 usando Upsample (para dimensões não-padrão)
            nn.Upsample(size=(31, 30), mode='bilinear'),  # 16x31x30

            # Camada final para obter 1 canal
            nn.Conv2d(16, 1, kernel_size=3, padding=(1, 1)),  # 1x31x30
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x