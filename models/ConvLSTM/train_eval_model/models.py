import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, dropout=0, init_scale=0.1):
        super(ConvLSTMCell, self).__init__()
        self.dropout_rate = dropout  # ✅ CORRIGIDO: estava faltando esta linha
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Camadas para processar a entrada (W) COM DROPOUT
        self.conv_xi = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_xf = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_xc = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_xo = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)

        # Dropout para as convoluções de entrada
        self.dropout_xi = nn.Dropout2d(self.dropout_rate)
        self.dropout_xf = nn.Dropout2d(self.dropout_rate)
        self.dropout_xc = nn.Dropout2d(self.dropout_rate)
        self.dropout_xo = nn.Dropout2d(self.dropout_rate)

        # Camadas para processar o hidden state (R) COM DROPOUT
        self.conv_hi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_hf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_hc = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_ho = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)

        # Dropout para as convoluções de hidden state
        self.dropout_hi = nn.Dropout2d(self.dropout_rate)
        self.dropout_hf = nn.Dropout2d(self.dropout_rate)
        self.dropout_hc = nn.Dropout2d(self.dropout_rate)
        self.dropout_ho = nn.Dropout2d(self.dropout_rate)

        # Parâmetros peephole
        self.init_scale = init_scale
        self.W_ci = nn.Parameter(torch.randn(1, hidden_dim, 1, 1) * init_scale)
        self.W_cf = nn.Parameter(torch.randn(1, hidden_dim, 1, 1) * init_scale)
        self.W_co = nn.Parameter(torch.randn(1, hidden_dim, 1, 1) * init_scale)

    def forward(self, x, h, c):
        """
        x: [batch, input_dim, height, width] (input atual)
        h: [batch, hidden_dim, height, width] (hidden state anterior)
        c: [batch, hidden_dim, height, width] (cell state anterior)
        """
        # Portão de input COM DROPOUT
        i_t = torch.sigmoid(
            self.dropout_xi(self.conv_xi(x)) +
            self.dropout_hi(self.conv_hi(h)) +
            self.W_ci * c
        )

        # Portão de forget COM DROPOUT
        f_t = torch.sigmoid(
            self.dropout_xf(self.conv_xf(x)) +
            self.dropout_hf(self.conv_hf(h)) +
            self.W_cf * c
        )

        # Candidato para cell state COM DROPOUT
        c_tilde = torch.tanh(
            self.dropout_xc(self.conv_xc(x)) +
            self.dropout_hc(self.conv_hc(h))
        )

        # Novo cell state (sem dropout - informação crítica)
        new_c = f_t * c + i_t * c_tilde

        # Portão de output COM DROPOUT
        o_t = torch.sigmoid(
            self.dropout_xo(self.conv_xo(x)) +
            self.dropout_ho(self.conv_ho(h)) +
            self.W_co * new_c
        )

        # Novo hidden state (sem dropout - informação crítica)
        new_h = o_t * torch.tanh(new_c)

        return new_h, new_c


class ConvLSTM_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, dropout=0, init_scale=1.0):
        super(ConvLSTM_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, padding, dropout, init_scale)

    def forward(self, x, return_sequences=True):
        """
        x shape: [batch, timesteps, input_dim, height, width]
        return_sequences: se True, retorna todos hidden states temporais
        """
        batch_size, timesteps, channels, height, width = x.shape

        # Inicializar estados
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)

        hidden_states = []

        # Processar cada timestep sequencialmente
        for t in range(timesteps):
            x_t = x[:, t, :, :, :]
            h, c = self.cell(x_t, h, c)
            hidden_states.append(h)

        if return_sequences:
            # Stack todos os hidden states temporais
            # [batch, timesteps, hidden_dim, height, width]
            return torch.stack(hidden_states, dim=1), (h, c)
        else:
            # Apenas último hidden state
            return h, (h, c)


class MultiLayerConvLSTM(nn.Module):
    def __init__(self, layer_configs, n_input_frames=5, dropout_rate=0.2, init_scale=1.0):
        super(MultiLayerConvLSTM, self).__init__()
        self.layer_configs = layer_configs
        self.n_input_frames = n_input_frames
        self.init_scale = init_scale
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()

        for input_dim, hidden_dim, kernel_size, padding in layer_configs:
            self.layers.append(
                ConvLSTM_Layer(input_dim, hidden_dim, kernel_size, padding,
                               self.dropout_rate, self.init_scale)
            )

        # ✅ Adicionar LayerNorm para estabilidade
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([hidden_dim, 24, 21]) for _, hidden_dim, _, _ in layer_configs
        ])

        last_hidden_dim = layer_configs[-1][1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(last_hidden_dim, last_hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(last_hidden_dim),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(last_hidden_dim, last_hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(last_hidden_dim // 2, 1, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= self.init_scale
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Stacked LSTM CORRETO:
        - Camada 1: processa sequência temporal t=0..T-1
        - Camada 2: processa hidden states temporais da camada 1
        - Cada camada tem SEUS PRÓPRIOS estados (h, c) iniciais
        """
        batch_size, timesteps, channels, height, width = x.shape

        current_sequence = x

        for layer_idx, (layer, layer_norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Processar sequência temporal completa
            # Retorna todos hidden states: [batch, timesteps, hidden_dim, H, W]
            hidden_sequence, _ = layer(current_sequence, return_sequences=True)

            # ✅ Aplicar LayerNorm
            hidden_sequence = layer_norm(hidden_sequence)

            # Preparar input para próxima camada
            current_sequence = hidden_sequence

        # Usar último hidden state da última camada
        final_hidden = hidden_sequence[:, -1, :, :, :]  # Último timestep
        output = self.output_conv(final_hidden)
        return output
def create_conv_lstm_model(n_input_frames=5, use_multi_layer=False, dropout_rate=0.2, init_scale=0.01):
    if use_multi_layer:
        layer_configs = [
            # (input_dim, hidden_dim, kernel_size, padding)
            (1, 16, 3, 1),
            (16, 32, 3, 1),
            (32, 64, 3, 1),
            (64, 128, 3, 1)
        ]
        return MultiLayerConvLSTM(
            layer_configs,
            n_input_frames,
            dropout_rate=dropout_rate,
            init_scale=init_scale
        )
    else:
        # Versão simples com 1 camada
        class SimpleConvLSTM(nn.Module):
            def __init__(self, input_channels=1, hidden_channels=64, n_input_frames=5):
                super(SimpleConvLSTM, self).__init__()
                self.n_input_frames = n_input_frames
                self.dropout_rate = dropout_rate
                self.init_scale = init_scale
                self.cell = ConvLSTMCell(input_channels, hidden_channels, 3, 1,dropout=self.dropout_rate, init_scale= self.init_scale)

                self.dropout_rate = dropout_rate
                self.output_conv = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(self.dropout_rate),
                    nn.Conv2d(hidden_channels//2, 1, kernel_size=1),
                )

            def forward(self, x):
                batch_size, timesteps, channels, height, width = x.shape

                # Inicializar estados
                h = torch.zeros(batch_size, self.cell.hidden_dim, height, width, device=x.device)
                c = torch.zeros(batch_size, self.cell.hidden_dim, height, width, device=x.device)

                # Processar cada timestep
                for t in range(timesteps):
                    x_t = x[:, t, :, :, :]  # [batch, channels, height, width]
                    h, c = self.cell(x_t, h, c)

                # Saída final
                output = self.output_conv(h)

                return output

        return SimpleConvLSTM(input_channels=1, hidden_channels=64, n_input_frames=n_input_frames)