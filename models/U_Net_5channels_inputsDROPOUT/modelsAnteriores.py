import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = self.conv_block(5, 16)
        self.enc_conv2 = self.conv_block(16, 32)
        self.enc_conv3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder (Upsampling with ConvTranspose2d)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(32, 16)

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Padding calculation helper
        self.crop = CropConcat()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def upConv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),

        )

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        # Encoder path
        enc1 = self.enc_conv1(x)
        #print(f"enc1: {enc1.shape}")

        enc2 = self.enc_conv2(self.pool(enc1))
        #print(f"enc2: {enc2.shape}")

        enc3 = self.enc_conv3(self.pool(enc2))
        #print(f"enc3: {enc3.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        #print(f"bottleneck: {bottleneck.shape}")

        # Decoder path
        dec3 = self.upconv3(bottleneck)
        #print(f"dec3 upconv: {dec3.shape}")
        dec3 = self.crop(dec3, enc3)
        #print(f"dec3 after crop: {dec3.shape}")
        dec3 = self.dec_conv3(dec3)
        #print(f"dec3 final: {dec3.shape}")

        dec2 = self.upconv2(dec3)
        #print(f"dec2 upconv: {dec2.shape}")
        dec2 = self.crop(dec2, enc2)
        #print(f"dec2 after crop: {dec2.shape}")
        dec2 = self.dec_conv2(dec2)
        #print(f"dec2 final: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        #print(f"dec1 upconv: {dec1.shape}")
        dec1 = self.crop(dec1, enc1)
        #print(f"dec1 after crop: {dec1.shape}")
        dec1 = self.dec_conv1(dec1)
        #print(f"dec1 final: {dec1.shape}")

        # Final output
        out = self.final_conv(dec1)
        #print(f"Final output: {out.shape}")

        return out


class CropConcat(nn.Module):
    def forward(self, x, enc):
        # Crop the encoder feature map to match decoder dimensions
        diffY = enc.size()[2] - x.size()[2]
        diffX = enc.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        return torch.cat([x, enc], dim=1)




import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = self.conv_block(5, 16)
        self.enc_conv2 = self.conv_block(16, 32)
        self.enc_conv3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder (Upsampling with ConvTranspose2d)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(32, 16)

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Padding calculation helper
        self.crop = CropConcat()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def upConv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        # Encoder path
        enc1 = self.enc_conv1(x)
        #print(f"enc1: {enc1.shape}")

        enc2 = self.enc_conv2(self.pool(enc1))
        #print(f"enc2: {enc2.shape}")

        enc3 = self.enc_conv3(self.pool(enc2))
        #print(f"enc3: {enc3.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        #print(f"bottleneck: {bottleneck.shape}")

        # Decoder path
        dec3 = self.upconv3(bottleneck)
        #print(f"dec3 upconv: {dec3.shape}")
        dec3 = self.crop(dec3, enc3)
        #print(f"dec3 after crop: {dec3.shape}")
        dec3 = self.dec_conv3(dec3)
        #print(f"dec3 final: {dec3.shape}")

        dec2 = self.upconv2(dec3)
        #print(f"dec2 upconv: {dec2.shape}")
        dec2 = self.crop(dec2, enc2)
        #print(f"dec2 after crop: {dec2.shape}")
        dec2 = self.dec_conv2(dec2)
        #print(f"dec2 final: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        #print(f"dec1 upconv: {dec1.shape}")
        dec1 = self.crop(dec1, enc1)
        #print(f"dec1 after crop: {dec1.shape}")
        dec1 = self.dec_conv1(dec1)
        #print(f"dec1 final: {dec1.shape}")

        # Final output
        out = self.final_conv(dec1)
        #print(f"Final output: {out.shape}")

        return out


class CropConcat(nn.Module):
    def forward(self, x, enc):
        # Crop the encoder feature map to match decoder dimensions
        diffY = enc.size()[2] - x.size()[2]
        diffX = enc.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        return torch.cat([x, enc], dim=1)