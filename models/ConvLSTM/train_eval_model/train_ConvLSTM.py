import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm #para barras de progresso das epocas
import csv
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import pandas as pd
"""MODELOS"""
from models import create_conv_lstm_model
"""DATASETS"""
from dataset import MultiFrameNextTimestampDataset, NormalizedMultiFrameDataset
from torch.utils.data import DataLoader,ConcatDataset
from norm_denorm import normalize_3d_tensor
from performance_mesure import evaluate_reconstruction, plot_metrics_comparison,plot_comprehensive_comparison
from norm_denorm import normalize_3d_tensor, denormalize_3d_tensor


#Função para criar csv com registro de losses de treinamento e validação
def atualizar_coluna_losses(lista_losses, arquivo_csv="Untitled_losses"):
    """
    Sobrescreve a primeira coluna de um arquivo CSV com os valores da lista,
    definindo o cabeçalho como 'hist_losses'.

    Args:
        lista_losses (list): Lista com os valores de loss a serem escritos
        arquivo_csv (str): Nome do arquivo CSV (opcional, usa o padrão se não fornecido)
    """
    # Abre o arquivo em modo de escrita
    with open(f'{arquivo_csv}', 'w', newline='') as arquivo:
        writer = csv.writer(arquivo)
        # Escreve o cabeçalho
        writer.writerow(['hist_losses'])
        # Escreve cada valor da lista em uma linha
        for register in lista_losses:
            writer.writerow([register])

# Função para exibir cabeçalho das épocas
def print_epoch_header(current_epoch, total_num_epochs):
    header = f" Epoch {current_epoch + 1}/{total_num_epochs} "
    print("\n" + "=" * 60)
    print(header.center(60, '='))
    print("=" * 60)

#Função para carregar sequencias de com o indice dos timestamps
def load_sequences_from_tensor(csv_path, tensor_path,timestamps_path, top_n,n_input_frames=5, times_aread=1, batch_size=32, seed=42):
    """
    Carrega as N primeiras sequências do tensor com base no CSV e cria datasets combinados.

    Args:
        csv_path: Caminho para o arquivo CSV com os índices
        tensor_path: Caminho para o tensor de precipitação
        top_n: Número de sequências a serem carregadas
        times_aread: número de registros a frente que será previsto
        batch_size: Tamanho do batch para os DataLoaders
        seed: Semente para reprodutibilidade (default: 42)

    Returns:
        Tuple: (train_loader, val_loader, test_loader, test_dataset, normalization_max_va)

    """
    # Carregar os dados
    df = pd.read_csv(csv_path)

    #Carrega tensor e anexa indice do timestamp correspondente
    full_tensor = torch.load(tensor_path, weights_only=True)
    with open(timestamps_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        lista.pop(0)
        print(len(lista))
        print(len(full_tensor))
        #novo formato dos dados (idx, full_tensor[idx])
        tensor_with_timestamp = [(k, full_tensor[k].float()) for k in range(len(lista))]
        #print(tensor_with_timestamp[0])

    min_required_length = n_input_frames + times_aread
    #df = df[df['n_registros'] >= min_required_length]
    # Ordenar por precipitação máxima (se não estiver ordenado)
    #df = df.sort_values('max_precipitacao', ascending=False)

    if top_n == "all": #Seleciona todas as sequencias válidas
        top_sequences = df
    else:
        # Pegar as top_n sequências
        top_sequences = df.head(top_n)

    # Lista para armazenar os datasets
    sequences = []
    discarded_seqs = 0
    for _, row in top_sequences.iterrows():
        start_idx = row['indice_inicial']
        end_idx = row['indice_final']

        # Extrair a sequência do tensor
        sequence = tensor_with_timestamp[start_idx:end_idx+1]
        #possivel lugar para vincular os timestamps
        #print(sequence[0])
        if len(sequence) < min_required_length:
            discarded_seqs += 1
            continue  # força a descartar sequencias de tamanho inapropriado


        # Criar dataset para esta sequência
            # Adicionar à lista de datasets individuais
        sequences.append(sequence)

    print(f"{discarded_seqs} sequência(s) descartada(s) por tamanho insuficiente")
    print(f"{len(sequences)} sequência(s) válida(s)")

        # Criar dataset normalizado combinado
    normalized_dataset = NormalizedMultiFrameDataset(
        sequences,
        transform=normalize_3d_tensor,
        n_input_frames=n_input_frames,
        n_timestamps_ahead=times_aread
    )

    # Dividir em train, val, test (70%, 15%, 15%)
    total_len = len(normalized_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # Criar generator com seed fixa
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = torch.utils.data.random_split(
        normalized_dataset,
        [train_len, val_len, test_len],
        generator=generator
    )

    # Criar DataLoaders
    train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_load = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_load = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_load, val_load, test_load, test_set, normalized_dataset.max_val

#Carrega CSV com timestamps
def load_timestamps(csv_path):
    with open(csv_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        # Remove cabeçalho e pega apenas a coluna do timestamp (assumindo que é a primeira coluna)
        timestamps = [row[0] for row in lista[1:]]
    return timestamps

#analisa primeiro batch para avaliar quais as dimensões adequadas.
def first_batch_analysis(loader):
    # Mostrar shape do primeiro batch
    print("=" * 50)
    print("ANÁLISE DO PRIMEIRO BATCH:")
    print("=" * 50)

    # Pegar primeiro batch
    first_batch = next(iter(loader))
    print(f"Tipo do batch: {type(first_batch)}")

    if isinstance(first_batch, (list, tuple)) and len(first_batch) == 2:
        # Descompacta batch
        input_data, target_data = first_batch
        input_indices, input_images = input_data
        target_index, target_image = target_data
        # print(input_data)
        # print(target_data)
        print(f"Input indices shape: {len(input_indices)} índices)")
        print(f"Input data shape: {input_images.shape}")  # [batch, timesteps, channels, height, width]
        print(f"Target index shape: {len(target_index)} indices")
        print(f"Target data shape: {target_image.shape}")  # [batch, channels, height, width]

        # Verificar se precisa adicionar dimensão de canal
        if input_images.dim() == 4:
            print("Input images tem 4 dimensões, adicionando dimensão de canal...")
            input_images = input_images.unsqueeze(2)  # [batch, timesteps, 1, height, width]
            print(f"Input images shape após ajuste: {input_images.shape}")

        if target_image.dim() == 3:
            print("Target image tem 3 dimensões, adicionando dimensão de canal...")
            target_image = target_image.unsqueeze(1)  # [batch, 1, height, width]
            target_image = target_image.unsqueeze(2)  # [batch, 1, 1, height, width]
            print(f"Target image shape após ajuste: {target_image.shape}")

    print("=" * 50)

#file_paths
path_tensor_precip = "../../../data/tensores/filtrados_any_5mm/tensor_radar_RJ_precipitacao_filtrado_any5.0mm.pt"
path_csv_seq = "../../../data/tensores/filtrados_any_5mm/sequencias_sem_gaps_filtradas_any5mm.csv"
path_csv_timestamps = "../../../data/tensores/filtrados_any_5mm/timestamps_filtrado_any5.0mm.csv"


# Parâmetros de treinamento (MODELO SERÁ NOMEADO A PARTIR DESTAS INFOS)
n_sequencias_datasets = 'all'
n_inputs = 5
n_stamps_aread = 2
batchs_size = 8
num_epochs = 200
patience = 15
learning_rate = 0.0005
drop_rate= 0.2
init_scale = 0.2 #escala de iniciação de parametros peephole(bias in ConvLSTM Cell)

# Configurações iniciais
model_name = f"correctedMultiLayer_ConvLSTM_any5mm_predict{n_stamps_aread}aread_{n_sequencias_datasets}seq_batch{batchs_size}_ep{num_epochs}_pat{patience}_lr{str(learning_rate).replace('.', '_')}_drop{str(drop_rate).replace('.', '_')}_initScale{str(init_scale).replace('.', '_')}"
os.makedirs(f'../saved_models/{model_name}', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicialização do modelo
model = create_conv_lstm_model(n_input_frames=n_inputs,dropout_rate=drop_rate,init_scale=init_scale, use_multi_layer=True).to(device)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Carregamento dos dados
train_loader, val_loader, test_loader, test_dataset, normalization_max_val = load_sequences_from_tensor(
    path_csv_seq,
    path_tensor_precip,
    timestamps_path=path_csv_timestamps,
    top_n=n_sequencias_datasets,
    times_aread=n_stamps_aread,
    n_input_frames=n_inputs,
    batch_size=batchs_size)

first_batch_analysis(train_loader)

best_loss = float('inf')
patience_counter = 0
early_stop = False

# Listas para armazenamento
train_losses = []
val_losses = []


# LOOP PRINCIPAL DE TREINAMENTO
for epoch in range(num_epochs):
    if early_stop:
        break

    print_epoch_header(epoch, num_epochs)

    model.train()
    train_loss = 0.0
    train_progress = tqdm(train_loader, desc='Training', leave=False)

    # Executa modelo com dataloader de treinamento, atualiza gradientes, e gera as losses
    for batch in train_progress:
        # Descompacta batch
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_data, target_data = batch
            input_indices, input_images = input_data
            target_index, target_image = target_data
        else:
            print("Problemas com recebimento do dataset (Formato inválido)")
            continue

        # Ajustar dimensionalidade dos inputs
        if input_images.dim() == 4:  # [batch, timesteps, height, width]
            input_images = input_images.unsqueeze(2)  # Adiciona dimensão de canal -> [batch, timesteps, 1, height, width]

        if target_image.dim() == 3:  # [batch, height, width]
            target_image = target_image.unsqueeze(1)  # Adiciona dimensão de canal -> [batch, 1, height, width]
        elif target_image.dim() == 4:  # [batch, 1, height, width] - já está correto
            pass
        else:
            raise ValueError(f"Dimensionalidade inválida do target: {target_image.dim()}")

        # Converter para float e enviar para dispositivo
        input_images = input_images.float().to(device)
        target_image = target_image.float().to(device)

        # Zerar gradientes e forward pass
        optimizer.zero_grad()
        outputs = model(input_images)

        # Calcular loss e backpropagation
        loss = criterion(outputs, target_image)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * target_image.size(0)
        train_progress.set_postfix({'train_batch_loss': f'{loss.item():.6f}'})

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    #salva csv de historico de losses
    atualizar_coluna_losses(train_losses, arquivo_csv=f'../saved_models/{model_name}/train_losses.csv')

####FASE DE VALIDAÇÃO#######
    model.eval()
    val_loss = 0.0
    val_progress = tqdm(val_loader, desc='Validating', leave=False)
    #Executa modelo com dataloader de validação e gera as losses
    with torch.no_grad():
        n_batch=0
        for batch in val_progress:
            # Descompacta batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                val_input, val_target = batch
                val_input_indices, val_input_images = val_input
                val_target_index, val_target_image = val_target
            else:
                print("Problemas com recebimento do dataset (Formato inválido)")
                continue

            # Ajustar dimensionalidade dos inputs
            if val_input_images.dim() == 4:  # [batch, timesteps, height, width]
                val_input_images = val_input_images.unsqueeze(2)  # Adiciona dimensão de canal -> [batch, timesteps, 1, height, width]

            if val_target_image.dim() == 3:  # [batch, height, width]
                val_target_image = val_target_image.unsqueeze(1)  # Adiciona dimensão de canal -> [batch, 1, height, width]
            elif val_target_image.dim() == 4:  # [batch, 1, height, width] - já está correto
                pass
            else:
                raise ValueError(f"Dimensionalidade inválida do target: {target_image.dim()}")

            # Converter para float e enviar para dispositivo
            val_input_images = val_input_images.float().to(device)
            val_target_image = val_target_image.float().to(device)

            val_outputs = model(val_input_images)
            loss = criterion(val_outputs, val_target_image)
            val_loss += loss.item() * val_target_image.size(0)
            val_progress.set_postfix({'val_batch_loss': f'{loss.item()}'})

    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    #salva csv de historico de losses
    atualizar_coluna_losses(val_losses, arquivo_csv=f'../saved_models/{model_name}/val_losses.csv')


    # Resumo da época
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Training Loss:   {train_loss}")
    print(f"  Validation Loss: {val_loss}")
    print(f"  Best Val Loss:   {best_loss}")

##### LÓGICA DO EARLY-STOPPING
    if val_loss < best_loss:
        print(f"\n  Validation loss improved from {best_loss} to {val_loss}")
        print("  Saving model...")
        best_loss = val_loss
        patience_counter = 0

        os.makedirs(f'../saved_models/{model_name}', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f'../saved_models/{model_name}/best_model.pth')
    else:
        patience_counter += 1
        print(f"\n  No improvement in validation loss ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("  Early stopping triggered!")
            early_stop = True
    if train_loss < 0.8 * val_loss:
        print("  Warning: Potential overfitting detected")
    print("-" * 60)


#DEPOIS DE TODAS AS ÉPOCAS
# Plotagem de Train e Validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)

filename = f"../saved_models/{model_name}/train_val_losses.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
# AVALIAÇÃO DESNORMALIZADA ###########################
model.eval()


def evaluate_model(loader, model, denormalize=False, max_val=None):
    all_inputs_last_only = []  # ⭐ NOVO: apenas últimos frames
    all_inputs_complete = []  # ⭐ NOVO: todos os 5 frames
    all_outputs = []
    all_targets = []

    for test_batch in loader:
        # Descompactar batch
        input_data, target_data = test_batch
        input_indices, input_images = input_data
        target_index, target_image = target_data

        # Ajustar dimensionalidade
        if input_images.dim() == 4:
            input_images = input_images.unsqueeze(2)
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(1)

        # Converter para dispositivo
        input_images = input_images.float().to(device)
        target_image = target_image.float().to(device)

        # Forward pass
        with torch.no_grad():
            test_outputs = model(input_images)

        if denormalize and max_val is not None:
            input_images = denormalize_3d_tensor(input_images, max_val)
            test_outputs = denormalize_3d_tensor(test_outputs, max_val)
            target_image = denormalize_3d_tensor(target_image, max_val)

        # Converter para CPU
        input_images = input_images.cpu()
        test_outputs = test_outputs.cpu()
        target_image = target_image.cpu()

        # CORREÇÃO: Processar índices corretamente
        batch_size = input_images.shape[0]

        for i in range(batch_size):
            # ⭐⭐ VERSÃO 1: Apenas o último frame (para comparação simples)
            last_frame_idx = input_indices[4][i].item()  # Índice do último frame (t)
            last_frame_image = input_images[i, -1]  # Última imagem da sequência (t)
            all_inputs_last_only.append((last_frame_idx, last_frame_image))

            # ⭐⭐ VERSÃO 2: Todos os 5 frames (para visualização completa)
            complete_frames = []
            for j in range(5):  # Para cada um dos 5 frames
                frame_idx = input_indices[j][i].item()
                frame_image = input_images[i, j]
                complete_frames.append((frame_idx, frame_image))
            all_inputs_complete.append(complete_frames)

            # Target e output correspondentes
            target_idx = target_index[i].item()
            all_outputs.append((target_idx, test_outputs[i]))
            all_targets.append((target_idx, target_image[i]))

    return all_inputs_last_only, all_inputs_complete, all_outputs, all_targets
# Avaliação
x_denorm_last, x_denorm_complete, y_pred_denorm, y_true_denorm = evaluate_model(
    test_loader, model, denormalize=True, max_val=normalization_max_val
)

# Calcular métricas
predict_metrics_denorm = evaluate_reconstruction(y_true_denorm, y_pred_denorm)
persist_metrics_denorm = evaluate_reconstruction(y_true_denorm, x_denorm_last)

print("\nMÉTRICAS DO MODELO CONVLSTM:")
for key, value in predict_metrics_denorm.items():
    if '_mean' in key:
        print(f"{key}: {value:.4f}")

timestamps = load_timestamps(path_csv_timestamps)


print(f"Métricas modelo: {len(predict_metrics_denorm.get('mse', []))} amostras")
print(f"Métricas persistência: {len(persist_metrics_denorm.get('mse', []))} amostras")

# Verificar se há valores válidos
valid_model = [x for x in predict_metrics_denorm.get('mse', []) if not np.isnan(x)]
valid_persist = [x for x in persist_metrics_denorm.get('mse', []) if not np.isnan(x)]

print(f"Modelo - amostras válidas: {len(valid_model)}")
print(f"Persistência - amostras válidas: {len(valid_persist)}")

# Plot comparativo
plot_metrics_comparison(predict_metrics_denorm, persist_metrics_denorm, "ConvLSTM", "Persistência",model_name_dir=f'../saved_models/{model_name}')

# Plot abrangente.
plot_comprehensive_comparison(y_true_denorm, y_pred_denorm, x_denorm_complete, timestamps, model_name_dir=f'../saved_models/{model_name}',top_n=3)

