import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

"""MODELOS"""
from models import create_conv_lstm_model

"""DATASETS"""
from dataset import NormalizedMultiFrameDataset
from norm_denorm import log_norm, log_denorm, test_normalization_roundtrip
from performance_mesure import evaluate_predictions, plot_metrics_comparison, plot_comprehensive_comparison


# Função para criar csv com registro de losses de treinamento e validação
def atualizar_coluna_losses(lista_losses, arquivo_csv="Untitled_losses"):
    with open(f'{arquivo_csv}', 'w', newline='') as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(['hist_losses'])
        for register in lista_losses:
            writer.writerow([register])


# Função para exibir cabeçalho das épocas
def print_epoch_header(current_epoch, total_num_epochs, model_name):
    header = f" {model_name} - Epoch {current_epoch + 1}/{total_num_epochs} "
    print("\n" + "=" * 80)
    print(header.center(80, '='))
    print("=" * 80)


# Função para carregar sequências
def load_normalized_sequences_loaders(sequences_csv_path, tensor_path, timestamps_path, top_n, n_input_frames=5, times_aread=1,
                               batch_size=32, seed=42):
    df = pd.read_csv(sequences_csv_path)
    full_tensor = torch.load(tensor_path, weights_only=True)

    with open(timestamps_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        lista.pop(0)
        tensor_with_timestamp = [(k, full_tensor[k].float()) for k in range(len(lista))]

    min_required_length = n_input_frames + times_aread
    if top_n == "all":
        top_sequences = df
    else:
        top_sequences = df.head(top_n)

    sequences = []
    discarded_seqs = 0
    for _, row in top_sequences.iterrows():
        start_idx = row['indice_inicial']
        end_idx = row['indice_final']
        sequence = tensor_with_timestamp[start_idx:end_idx + 1]

        if len(sequence) < min_required_length:
            discarded_seqs += 1
            continue
        sequences.append(sequence)

    print(f"{discarded_seqs} sequência(s) descartada(s) por tamanho insuficiente")
    print(f"{len(sequences)} sequência(s) válida(s)")

    normalized_dataset = NormalizedMultiFrameDataset(
        sequences,
        transform=log_norm,
        n_input_frames=n_input_frames,
        n_timestamps_ahead=times_aread
    )

    total_len = len(normalized_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        normalized_dataset,
        [train_len, val_len, test_len],
        generator=generator
    )

    train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_load = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_load = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_load, val_load, test_load, test_set


# Carrega CSV com timestamps
def load_timestamps(csv_path):
    with open(csv_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        timestamps = [row[0] for row in lista[1:]]
    return timestamps


# Função de análise do primeiro batch
def first_batch_analysis(loader):
    print("=" * 50)
    print("ANÁLISE DO PRIMEIRO BATCH:")
    print("=" * 50)

    first_batch = next(iter(loader))
    print(f"Tipo do batch: {type(first_batch)}")

    if isinstance(first_batch, (list, tuple)) and len(first_batch) == 2:
        input_data, target_data = first_batch
        input_indices, input_images = input_data
        target_index, target_image = target_data

        print(f"Input indices shape: {len(input_indices)} índices)")
        print(f"Input data shape: {input_images.shape}")
        print(f"Target index shape: {len(target_index)} indices")
        print(f"Target data shape: {target_image.shape}")

        if input_images.dim() == 4:
            print("Input images tem 4 dimensões, adicionando dimensão de canal...")
            input_images = input_images.unsqueeze(2)
            print(f"Input images shape após ajuste: {input_images.shape}")

        if target_image.dim() == 3:
            print("Target image tem 3 dimensões, adicionando dimensão de canal...")
            target_image = target_image.unsqueeze(1)
            target_image = target_image.unsqueeze(2)
            print(f"Target image shape após ajuste: {target_image.shape}")

    print("=" * 50)


# Executa modelo para test loader e desnormaliza
def executeTest_Denormalize(loader, model, denormalize=False, max_val=None, n_inputs=5):
    all_inputs_last_only = []
    all_inputs_complete = []
    all_outputs = []
    all_targets = []

    for test_batch in loader:
        input_data, target_data = test_batch
        input_indices, input_images = input_data
        target_index, target_image = target_data

        if input_images.dim() == 4:
            input_images = input_images.unsqueeze(2)
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(1)

        input_images = input_images.float().to(device)
        target_image = target_image.float().to(device)

        with torch.no_grad():
            test_outputs = model(input_images)

        if denormalize:
            input_images = log_denorm(input_images)
            test_outputs = log_denorm(test_outputs)
            target_image = log_denorm(target_image)

        input_images = input_images.cpu()
        test_outputs = test_outputs.cpu()
        target_image = target_image.cpu()

        batch_size = input_images.shape[0]

        for i in range(batch_size):
            last_frame_idx = input_indices[n_inputs - 1][i].item()
            last_frame_image = input_images[i, -1]
            all_inputs_last_only.append((last_frame_idx, last_frame_image))

            complete_frames = []
            for j in range(n_inputs):
                frame_idx = input_indices[j][i].item()
                frame_image = input_images[i, j]
                complete_frames.append((frame_idx, frame_image))
            all_inputs_complete.append(complete_frames)

            target_idx = target_index[i].item()
            all_outputs.append((target_idx, test_outputs[i]))
            all_targets.append((target_idx, target_image[i]))

    return all_inputs_last_only, all_inputs_complete, all_outputs, all_targets

# Plotar comparação de múltiplos modelos
def plot_multiple_models_comparison(all_models_metrics, all_models_persist_metrics,
                                    output_dir='../saved_models/comparison'):
    """
    Plota comparação de métricas entre múltiplos modelos e suas persistências individuais
    """
    import matplotlib
    matplotlib_version = matplotlib.__version__

    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = ['mse', 'mae', 'rmse', 'r2', 'nash_sutcliffe_efficiency', 'ssim', 'correlation_pearsonr']

    for metric in metrics_to_plot:
        model_names = []
        all_data = []
        means = []

        # Adicionar persistências individuais para cada modelo
        for model_name, metrics in all_models_metrics.items():
            # Persistência para este modelo específico
            persist_metrics = all_models_persist_metrics.get(model_name, {})
            persist_data = persist_metrics.get(metric, [])

            if persist_data:
                persist_data_clean = [x for x in persist_data if not np.isnan(x)]
                if persist_data_clean:
                    minutes = model_name.split('_')[1].replace('min', '') + 'min'
                    model_names.append(f'Persist {minutes}')
                    all_data.append(persist_data_clean)
                    means.append(np.mean(persist_data_clean))

            # Modelo preditivo
            model_data = metrics.get(metric, [])
            if model_data:
                model_data_clean = [x for x in model_data if not np.isnan(x)]
                if model_data_clean:
                    # Encurtar nome do modelo se necessário
                    short_name = model_name.split('_')[0] + '_' + model_name.split('_')[1]
                    model_names.append(short_name)
                    all_data.append(model_data_clean)
                    means.append(np.mean(model_data_clean))

        if not all_data:
            continue

        plt.figure(figsize=(max(10, len(model_names) * 1.2), 8))

        # Criar boxplot
        if tuple(map(int, matplotlib_version.split('.')[:2])) >= (3, 9):
            box = plt.boxplot(all_data, tick_labels=model_names, patch_artist=True)
        else:
            box = plt.boxplot(all_data, labels=model_names, patch_artist=True)

        # Colorir boxes
        colors = ['lightcoral' if 'Persist' in name else 'lightblue' for name in model_names]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Adicionar médias de forma discreta
        for i, mean_val in enumerate(means):
            # Posicionar na linha superior do boxplot
            y_pos = np.percentile(all_data[i], 75) * 1.05

            plt.text(i + 1, y_pos, f'{mean_val:.3f}',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9,
                               edgecolor='gray', linewidth=0.5))

        plt.title(f'{metric.upper()} - Modelos vs Persistências', fontsize=14, fontweight='bold')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Salvar também um CSV com as médias para referência
        summary_df = pd.DataFrame({
            'Modelo': model_names,
            f'{metric}_mean': means,
            f'{metric}_std': [np.std(data) for data in all_data]
        })
        summary_df.to_csv(f'{output_dir}/summary_{metric}.csv', index=False)
# Plotar losses de treinamento de múltiplos modelos
def plot_multiple_models_losses(all_models_losses, output_dir='../saved_models/comparison'):
    """
    Plota comparação de losses de treinamento e validação entre múltiplos modelos em um único gráfico
    """
    os.makedirs(output_dir, exist_ok=True)

    # Configurações visuais
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models_losses)))
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # Criar figura única
    plt.figure(figsize=(14, 8))

    # Plotar todas as losses
    for i, (model_name, losses) in enumerate(all_models_losses.items()):
        epochs = range(1, len(losses['train']) + 1)

        # Training loss
        plt.plot(epochs, losses['train'],
                 color=colors[i],
                 linestyle=line_styles[0],
                 linewidth=2,
                 marker=markers[i % len(markers)],
                 markevery=max(1, len(epochs) // 10),
                 markersize=6,
                 label=f'{model_name} - Train')

        # Validation loss
        plt.plot(epochs, losses['val'],
                 color=colors[i],
                 linestyle=line_styles[1],
                 linewidth=2,
                 marker=markers[i % len(markers)],
                 markevery=max(1, len(epochs) // 10),
                 markersize=6,
                 label=f'{model_name} - Val',
                 alpha=0.7)

    # Configurações do gráfico
    plt.xlabel('Épocas', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training e Validation Loss - Todos os Modelos', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Salvar
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_all_losses_single_plot.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Gráfico único de comparação salvo em: {output_dir}")


# CONFIGURAÇÃO DE MÚLTIPLOS MODELOS


models_config = [
    {
        'name': 'predict_10min',
        'n_sequencias_datasets': 'all',
        'n_inputs': 5,
        'n_stamps_aread': 1,  # 10 minutos (1 × 10min)
        'batchs_size': 8,
        'num_epochs': 200,
        'patience': 15,
        'learning_rate': 0.0001,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True
    },
    {
        'name': 'predict_20min',
        'n_sequencias_datasets': 'all',
        'n_inputs': 5,
        'n_stamps_aread': 2,  # 30 minutos (3 × 10min)
        'batchs_size': 8,
        'num_epochs': 200,
        'patience': 15,
        'learning_rate': 0.0001,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True
    },
    {
        'name': 'predict_30min',
        'n_sequencias_datasets': 'all',
        'n_inputs': 5,
        'n_stamps_aread': 3,  # 50 minutos (5 × 10min)
        'batchs_size': 8,
        'num_epochs': 200,
        'patience': 15,
        'learning_rate': 0.0001,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True
    },
    {
        'name': 'predict_40min',
        'n_sequencias_datasets': 'all',
        'n_inputs': 5,
        'n_stamps_aread': 4,  # 70 minutos (7 × 10min)
        'batchs_size': 8,
        'num_epochs': 200,
        'patience': 15,
        'learning_rate': 0.0001,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True
    },
    {
        'name': 'predict_50min',
        'n_sequencias_datasets': 'all',
        'n_inputs': 5,
        'n_stamps_aread': 5,  # 90 minutos (9 × 10min)
        'batchs_size': 8,
        'num_epochs': 200,
        'patience': 15,
        'learning_rate': 0.0001,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True
    },
    {
        'name': 'predict_60min',
        'n_sequencias_datasets': 'all',
        'n_inputs': 5,
        'n_stamps_aread': 6  ,  # 110 minutos (11 × 10min)
        'batchs_size': 8,
        'num_epochs': 200,
        'patience': 15,
        'learning_rate': 0.0001,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True
    }
]
# File paths (comuns a todos os modelos)
path_tensor_precip = "../../../data/tensores/complete/tensor_radar_RJ_precipitacao.pt"
path_csv_seq = "../../../data/csvs/indices_sequencias_RJ.csv"
path_csv_timestamps = "../../../data/csvs/timestampsRJ.csv"

multiTrainName = 'LogNorm_multitrain_10to600min'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dicionários para armazenar resultados de todos os modelos
all_models_metrics = {}
all_models_losses = {}
all_models_predictions = {}
all_models_persist_metrics = {}

n_inputs = 5
drop_rate = 0.2
init_scale = 0.1
use_multi_layer = True

# Inicialização do modelo
model = create_conv_lstm_model(
        n_input_frames=n_inputs,
        dropout_rate=drop_rate,
        init_scale=init_scale,
        use_multi_layer=use_multi_layer
    ).to(device)


# LOOP PRINCIPAL PARA TREINAR MÚLTIPLOS MODELOS
for model_config in models_config:
    print(f"\n{'#' * 80}")
    print(f"INICIANDO TREINAMENTO DO MODELO: {model_config['name']}")
    print(f"{'#' * 80}")

    # Extrair configurações
    name = model_config['name']
    n_sequencias_datasets = model_config['n_sequencias_datasets']
    n_inputs = model_config['n_inputs']
    n_stamps_aread = model_config['n_stamps_aread']
    batchs_size = model_config['batchs_size']
    num_epochs = model_config['num_epochs']
    patience = model_config['patience']
    learning_rate = model_config['learning_rate']
    drop_rate = model_config['drop_rate']
    init_scale = model_config['init_scale']
    use_multi_layer = model_config['use_multi_layer']

    # Criar nome do modelo e diretório
    model_name = f"{name}_predict{n_stamps_aread}aread_{n_sequencias_datasets}seq_batch{batchs_size}_ep{num_epochs}_lr{str(learning_rate).replace('.', '_')}_drop{str(drop_rate).replace('.', '_')}_initScale{str(init_scale).replace('.', '_')}"
    model_dir = f'../saved_models/{multiTrainName}/{model_name}'
    os.makedirs(model_dir, exist_ok=True)



    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Carregamento dos dados 
    train_loader, val_loader, test_loader, test_dataset = load_normalized_sequences_loaders(
        sequences_csv_path = path_csv_seq,
        tensor_path= path_tensor_precip,
        timestamps_path=path_csv_timestamps,
        top_n=n_sequencias_datasets,
        times_aread=n_stamps_aread,
        n_input_frames=n_inputs,
        batch_size=batchs_size
    )



    # Análise do primeiro batch 
    if model_config == models_config[0]:
        test_normalization_roundtrip(log_norm, log_denorm, tensor=path_tensor_precip)
        first_batch_analysis(train_loader)

    # Treinamento do modelo individual
    best_loss = float('inf')
    patience_counter = 0
    early_stop = False

    train_losses = []
    val_losses = []

    #LOOP PRINCIPAL DE tREINAMENTO
    for epoch in range(num_epochs):
        if early_stop:
            break

        print_epoch_header(epoch, num_epochs, name)

        # Fase de treinamento
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc='Training', leave=False)

        for batch in train_progress:
            input_data, target_data = batch
            input_indices, input_images = input_data
            target_index, target_image = target_data

            if input_images.dim() == 4:
                input_images = input_images.unsqueeze(2)
            if target_image.dim() == 3:
                target_image = target_image.unsqueeze(1)

            input_images = input_images.float().to(device)
            target_image = target_image.float().to(device)

            optimizer.zero_grad()
            outputs = model(input_images)
            loss = criterion(outputs, target_image)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target_image.size(0)
            train_progress.set_postfix({'train_batch_loss': f'{loss.item():.6f}'})

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        atualizar_coluna_losses(train_losses, arquivo_csv=f'{model_dir}/train_losses.csv')

        # Fase de validação
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc='Validating', leave=False)

        with torch.no_grad():
            for batch in val_progress:
                val_input, val_target = batch
                val_input_indices, val_input_images = val_input
                val_target_index, val_target_image = val_target

                if val_input_images.dim() == 4:
                    val_input_images = val_input_images.unsqueeze(2)
                if val_target_image.dim() == 3:
                    val_target_image = val_target_image.unsqueeze(1)

                val_input_images = val_input_images.float().to(device)
                val_target_image = val_target_image.float().to(device)

                val_outputs = model(val_input_images)
                loss = criterion(val_outputs, val_target_image)
                val_loss += loss.item() * val_target_image.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        atualizar_coluna_losses(val_losses, arquivo_csv=f'{model_dir}/val_losses.csv')

        # Resumo da época
        print(f"\n{name} - Epoch {epoch + 1} Summary:")
        print(f"  Training Loss:   {train_loss:.6f}")
        print(f"  Validation Loss: {val_loss:.6f}")
        print(f"  Best Val Loss:   {best_loss:.6f}")

        # Early stopping
        if val_loss < best_loss:
            print(f"\n  Validation loss improved from {best_loss:.6f} to {val_loss:.6f}")
            best_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'{model_dir}/best_model.pth')
        else:
            patience_counter += 1
            print(f"\n  No improvement in validation loss ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("  Early stopping triggered!")
                early_stop = True

        print("-" * 60)

    # Armazenar losses para comparação posterior
    all_models_losses[name] = {
        'train': train_losses,
        'val': val_losses
    }

    # Plot losses individuais
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{name} - Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_dir}/train_val_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Avaliação do modelo
    model.eval()
    x_denorm_last, x_denorm_complete, y_pred_denorm, y_true_denorm = executeTest_Denormalize(
        test_loader, model, denormalize=True, n_inputs= n_inputs)

    # Calcular métricas
    predict_metrics_denorm = evaluate_predictions(y_true_denorm, y_pred_denorm)
    persist_metrics_denorm = evaluate_predictions(y_true_denorm, x_denorm_last)

    # Armazenar métricas para comparação geral
    all_models_metrics[name] = predict_metrics_denorm
    all_models_persist_metrics[name] = persist_metrics_denorm

    all_models_predictions[name] = {
        'y_true': y_true_denorm,
        'y_pred': y_pred_denorm,
        'x_inputs': x_denorm_complete
    }

    print(f"\n{name} - MÉTRICAS DO MODELO CONVLSTM:")
    for key, value in predict_metrics_denorm.items():
        if '_mean' in key:
            print(f"{key}: {value:.4f}")

    print(f"{name} - MÉTRICAS DA PERSISTÊNCIA ({n_stamps_aread * 10}min):")
    for key, value in persist_metrics_denorm.items():
        if '_mean' in key:
            print(f"{key}: {value:.4f}")

    # Plot comparativo individual
    timestamps = load_timestamps(path_csv_timestamps)
    plot_metrics_comparison(predict_metrics_denorm, persist_metrics_denorm, name,
                            f"Persistência {n_stamps_aread * 10}min",
                            model_name_dir=model_dir)

    # Plot abrangente individual
    plot_comprehensive_comparison(
        y_true_denorm, y_pred_denorm, x_denorm_complete, timestamps,
        model_name_dir=model_dir, top_n=3, n_inputs= n_inputs
    )

# APÓS TREINAR TODOS OS MODELOS - COMPARAÇÃO GERAL (MODIFICADO)
print(f"\n{'#' * 80}")
print("COMPARAÇÃO GERAL ENTRE TODOS OS MODELOS")
print(f"{'#' * 80}")

# Gerar gráficos de comparação entre todos os modelos
comparison_dir = f'../saved_models/{multiTrainName}/comparison_geral'
plot_multiple_models_comparison(all_models_metrics, all_models_persist_metrics, output_dir=comparison_dir)
plot_multiple_models_losses(all_models_losses, output_dir=comparison_dir)

# Salvar resumo das métricas em CSV (MODIFICADO)
summary_data = []
for model_name, metrics in all_models_metrics.items():
    # Modelo preditivo
    row = {'Modelo': model_name, 'Tipo': 'Predição'}
    for key, value in metrics.items():
        if '_mean' in key:
            row[key] = value
    summary_data.append(row)

    # Persistência correspondente
    persist_metrics = all_models_persist_metrics[model_name]
    persist_row = {'Modelo': model_name, 'Tipo': 'Persistência'}
    for key, value in persist_metrics.items():
        if '_mean' in key:
            persist_row[key] = value
    summary_data.append(persist_row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{comparison_dir}/resumo_metricas.csv', index=False)

print("\nTreinamento de todos os modelos concluído!")
print(f"Gráficos de comparação salvos em: {comparison_dir}")
print(f"Resumo das métricas salvo em: {comparison_dir}/resumo_metricas.csv")

# ADICIONAR: Exibir resumo comparativo
print("\nRESUMO COMPARATIVO:")
print("=" * 100)
for model_name in all_models_metrics.keys():
    pred_metrics = all_models_metrics[model_name]
    persist_metrics = all_models_persist_metrics[model_name]

    # Extrair minutos do nome do modelo
    minutes = model_name.split('_')[1].replace('min', '') + 'min'

    print(f"\n{model_name} ({minutes} no futuro):")
    print(
        f"  MSE - Predição: {pred_metrics.get('mse_mean', 'N/A'):.4f} | Persistência: {persist_metrics.get('mse_mean', 'N/A'):.4f}")
    print(
        f"  MAE - Predição: {pred_metrics.get('mae_mean', 'N/A'):.4f} | Persistência: {persist_metrics.get('mae_mean', 'N/A'):.4f}")
    print(
        f"  R²  - Predição: {pred_metrics.get('r2_mean', 'N/A'):.4f} | Persistência: {persist_metrics.get('r2_mean', 'N/A'):.4f}")