import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetAutoencoder(nn.Module):
    def __init__(self, input_channels=5, output_channels=1, dropout_rate=0.2, init_scale=1.0):
        super(UNetAutoencoder, self).__init__()

        self.init_scale = init_scale  # Controle da intensidade da inicialização

        # Encoder mais raso para imagens pequenas
        self.enc_conv1 = self.conv_block(input_channels, 16, dropout_rate)
        self.enc_conv2 = self.conv_block(16, 32, dropout_rate)
        self.enc_conv3 = self.conv_block(32, 64, dropout_rate)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck menor
        self.bottleneck = self.conv_block(64, 128, dropout_rate)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(128, 64, dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(64, 32, dropout_rate)

        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv0 = self.conv_block(32, 16, dropout_rate)

        # Final output layer - SEM dropout na última camada
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Removido dropout da última camada para não limitar a saída
            nn.Conv2d(16, output_channels, kernel_size=1),
        )

        # Inicialização de pesos com controle de escala
        self._initialize_weights()

    def conv_block(self, in_channels, out_channels, dropout_rate):
        """Bloco convolucional com dropout aplicado CORRETAMENTE"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),  # ✅ CORRETO: dropout espacial 2D

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)  # ✅ CORRETO: dropout espacial 2D
        )

    def _initialize_weights(self):
        """Inicialização de pesos com controle de intensidade"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming normal com escala controlável
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Aplica escala na inicialização
                m.weight.data *= self.init_scale
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: [B, 5, 1, 24, 21] → precisamos remover a dimensão extra de canal
        # Se x tem 5 dimensões, assumimos que é [B, seq_len, C, H, W]
        if x.dim() == 5:
            # Combinar sequência e canais: [B, seq_len * C, H, W]
            batch_size, seq_len, channels, height, width = x.shape
            x = x.view(batch_size, seq_len * channels, height, width)

        # print(f"Input shape após ajuste: {x.shape}")

        # Encoder path
        enc1 = self.enc_conv1(x)  # [B, 16, 24, 21]
        # print(f"enc1 shape: {enc1.shape}")

        enc2_input = self.pool(enc1)  # [B, 16, 12, 10]
        # print(f"enc2_input shape: {enc2_input.shape}")
        enc2 = self.enc_conv2(enc2_input)  # [B, 32, 12, 10]
        # print(f"enc2 shape: {enc2.shape}")

        enc3_input = self.pool(enc2)  # [B, 32, 6, 5]
        # print(f"enc3_input shape: {enc3_input.shape}")
        enc3 = self.enc_conv3(enc3_input)  # [B, 64, 6, 5]
        # print(f"enc3 shape: {enc3.shape}")

        # Bottleneck
        bottleneck_input = self.pool(enc3)  # [B, 64, 3, 2]
        # print(f"bottleneck_input shape: {bottleneck_input.shape}")
        bottleneck = self.bottleneck(bottleneck_input)  # [B, 128, 3, 2]
        # print(f"bottleneck shape: {bottleneck.shape}")

        # Decoder path
        dec2 = self.upconv2(bottleneck)  # [B, 64, 6, 4]
        # print(f"dec2 upconv shape: {dec2.shape}")

        # Ajustar dimensões para match com enc3
        if dec2.shape[2:] != enc3.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc3.shape[2:], mode='bilinear', align_corners=False)
            # print(f"dec2 after interpolation: {dec2.shape}")

        dec2 = torch.cat([dec2, enc3], dim=1)  # [B, 128, 6, 5]
        dec2 = self.dec_conv2(dec2)  # [B, 64, 6, 5]
        # print(f"dec2 final shape: {dec2.shape}")

        dec1 = self.upconv1(dec2)  # [B, 32, 12, 10]
        # print(f"dec1 upconv shape: {dec1.shape}")

        if dec1.shape[2:] != enc2.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc2.shape[2:], mode='bilinear', align_corners=False)
            # print(f"dec1 after interpolation: {dec1.shape}")

        dec1 = torch.cat([dec1, enc2], dim=1)  # [B, 64, 12, 10]
        dec1 = self.dec_conv1(dec1)  # [B, 32, 12, 10]
        # print(f"dec1 final shape: {dec1.shape}")

        dec0 = self.upconv0(dec1)  # [B, 16, 24, 20]
        # print(f"dec0 upconv shape: {dec0.shape}")

        if dec0.shape[2:] != enc1.shape[2:]:
            dec0 = F.interpolate(dec0, size=enc1.shape[2:], mode='bilinear', align_corners=False)
            # print(f"dec0 after interpolation: {dec0.shape}")

        dec0 = torch.cat([dec0, enc1], dim=1)  # [B, 32, 24, 21]
        dec0 = self.dec_conv0(dec0)  # [B, 16, 24, 21]
        # print(f"dec0 final shape: {dec0.shape}")

        # Final output - SEM dropout na última camada
        out = self.final_conv(dec0)  # [B, 1, 24, 21]
        # print(f"Final output shape: {out.shape}")

        return out
