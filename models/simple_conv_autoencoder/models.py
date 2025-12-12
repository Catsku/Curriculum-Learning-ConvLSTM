import torch.nn as nn

class SimpleConvAutoencoder31x30(nn.Module):
    def __init__(self):
        super(SimpleConvAutoencoder31x30, self).__init__()

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
            nn.Upsample(size=(31, 30), mode='nearest'),  # 16x31x30

            # Camada final para obter 1 canal
            nn.Conv2d(16, 1, kernel_size=3, padding=(1, 1)),  # 1x31x30
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class SimpleConvAutoencoder24x21(nn.Module):
    def __init__(self):
        super(SimpleConvAutoencoder24x21, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=(1, 1)),  # 16x24x21
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 16x12x10 (arredonda para baixo)
            nn.Conv2d(16, 32, 3, padding=(1, 1)),  # 32x12x10
            nn.ReLU(True),
            nn.MaxPool2d(2)  # 32x6x5
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Expansão de 6x5 para 12x10
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x12x10
            nn.ReLU(True),
            # Expansão de 12x10 para 24x20
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x24x20
            nn.ReLU(True),
            # Ajuste final para 24x21 usando Upsample
            nn.Upsample(size=(24, 21), mode='nearest'),  # 16x24x21
            # Camada final
            nn.Conv2d(16, 1, kernel_size=3, padding=(1, 1)),  # 1x24x21
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = self.conv_block(1, 16)
        self.enc_conv2 = self.conv_block(16, 32)
        self.enc_conv3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder (Upsampling)
        self.upconv3 = self.upconv_block(128, 64)
        self.dec_conv3 = self.conv_block(128, 64)
        self.upconv2 = self.upconv_block(64, 32)
        self.dec_conv2 = self.conv_block(64, 32)
        self.upconv1 = self.upconv_block(32, 16)
        self.dec_conv1 = self.conv_block(32, 16)

        # Final output layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

        # Padding calculation helper
        self.crop = CropConcat()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc_conv1(x)  # [B, 16, 24, 21]
        enc2 = self.enc_conv2(self.pool(enc1))  # [B, 32, 12, 10]
        enc3 = self.enc_conv3(self.pool(enc2))  # [B, 64, 6, 5]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))  # [B, 128, 3, 2]

        # Decoder path with skip connections
        dec3 = self.upconv3(bottleneck)  # [B, 64, 6, 5]
        dec3 = self.crop(dec3, enc3)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.upconv2(dec3)  # [B, 32, 12, 10]
        dec2 = self.crop(dec2, enc2)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)  # [B, 16, 24, 21]
        dec1 = self.crop(dec1, enc1)
        dec1 = self.dec_conv1(dec1)

        # Final output
        out = self.final_conv(dec1)  # [B, 1, 24, 21]

        return out


class CropConcat(nn.Module):
    def forward(self, x, enc):
        # Crop the encoder feature map to match decoder dimensions
        diffY = enc.size()[2] - x.size()[2]
        diffX = enc.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        return torch.cat([x, enc], dim=1)

