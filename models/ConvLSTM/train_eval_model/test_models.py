import torch
import torch.nn as nn
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

"""MODELOS E DATASETS (Seus imports customizados)"""
from models import create_conv_lstm_model
from dataset import NormalizedMultiFrameDataset
from norm_denorm import log_norm, log_denorm
from performance_mesure import evaluate_predictions, plot_metrics_comparison, plot_comprehensive_comparison

# ==============================================================================
# 1. CONFIGURAÇÃO GERAL E CAMINHOS
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando em: {device}")

# Caminhos dos dados (Tensores e CSVs)
PATH_TENSOR_PRECIP = "../../../data/tensores/filtrados_any_5mm/tensor_radar_RJ_precipitacao_filtrado_any5.0mm.pt"
PATH_CSV_SEQ = "../../../data/tensores/filtrados_any_5mm/sequencias_sem_gaps_filtradas_any5mm.csv"
PATH_CSV_TIMESTAMPS = "../../../data/tensores/filtrados_any_5mm/timestamps_filtrado_any5.0mm.csv"

# Diretório onde os resultados da avaliação serão salvos
OUTPUT_DIR_ROOT = './test_results/comparison_batch_1'

# ==============================================================================
# 2. CONFIGURAÇÃO DOS MODELOS PARA TESTE
# ==============================================================================
# Adicione aqui os dicionários para cada modelo que deseja carregar e testar.
# IMPORTANTE: Os hiperparâmetros (hidden size, n_inputs, etc.) devem ser iguais aos do treino.

models_to_test = [
    {
        'alias': 'Modelo_10min',
        'path_pth': '../saved_models/5inputs_LogNorm_multitrain_10to60min/predict_10min_predict1aread_allseq_batch8_ep200_lr0_0001_drop0_3_initScale0_15/best_model.pth',
        'n_inputs': 5,
        'n_stamps_aread': 1,  # 1 step = 10 min
        'n_sequencias': 'all',
        'batch_size': 8,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True,
        'seed': 42
    },
    {
        'alias': 'Modelo_20min',
        'path_pth': '../saved_models/5inputs_LogNorm_multitrain_10to60min/predict_20min_predict2aread_allseq_batch8_ep200_lr0_0001_drop0_3_initScale0_15/best_model.pth',
        'n_inputs': 5,
        'n_stamps_aread': 2,  # 2 steps = 20 min
        'n_sequencias': 'all',
        'batch_size': 8,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True,
        'seed': 42
    },
    {
        'alias': 'Modelo_30min',
        'path_pth': '../saved_models/5inputs_LogNorm_multitrain_10to60min/predict_30min_predict3aread_allseq_batch8_ep200_lr0_0001_drop0_3_initScale0_15/best_model.pth',
        'n_inputs': 5,
        'n_stamps_aread': 3,  # 3 steps = 30 min
        'n_sequencias': 'all',
        'batch_size': 8,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True,
        'seed': 42
    },
    {
        'alias': 'Modelo_40min',
        'path_pth': '../saved_models/5inputs_LogNorm_multitrain_10to60min/predict_40min_predict4aread_allseq_batch8_ep200_lr0_0001_drop0_3_initScale0_15/best_model.pth',
        'n_inputs': 5,
        'n_stamps_aread': 4,  # 4 steps = 40 min
        'n_sequencias': 'all',
        'batch_size': 8,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True,
        'seed': 42
    },
    {
        'alias': 'Modelo_50min',
        'path_pth': '../saved_models/5inputs_LogNorm_multitrain_10to60min/predict_50min_predict5aread_allseq_batch8_ep200_lr0_0001_drop0_3_initScale0_15/best_model.pth',
        'n_inputs': 5,
        'n_stamps_aread': 5,  # 5 steps = 50 min
        'n_sequencias': 'all',
        'batch_size': 8,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True,
        'seed': 42
    },
    {
        'alias': 'Modelo_60min',
        'path_pth': '../saved_models/5inputs_LogNorm_multitrain_10to60min/predict_60min_predict6aread_allseq_batch8_ep200_lr0_0001_drop0_3_initScale0_15/best_model.pth',
        'n_inputs': 5,
        'n_stamps_aread': 6,  # 6 steps = 60 min
        'n_sequencias': 'all',
        'batch_size': 8,
        'drop_rate': 0.3,
        'init_scale': 0.15,
        'use_multi_layer': True,
        'seed': 42
    }
]

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS (Reaproveitadas do seu código)
# ==============================================================================

def load_timestamps(csv_path):
    with open(csv_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        timestamps = [row[0] for row in lista[1:]]
    return timestamps


def load_test_loader_only(sequences_csv_path, tensor_path, timestamps_path, top_n,
                          n_input_frames=5, times_aread=1, batch_size=32, seed=42):
    """
    Versão modificada para retornar apenas o test_loader, mantendo a consistência do split.
    """
    df = pd.read_csv(sequences_csv_path)
    full_tensor = torch.load(tensor_path, map_location='cpu', weights_only=True)  # Carrega na CPU primeiro para evitar OOM

    with open(timestamps_path, 'r') as arquivo:
        reader = csv.reader(arquivo)
        lista = list(reader)
        lista.pop(0)
        # Otimização: manter tensor na CPU até o DataLoader pegar o batch
        tensor_with_timestamp = [(k, full_tensor[k].float()) for k in range(len(lista))]

    min_required_length = n_input_frames + times_aread
    if top_n == "all":
        top_sequences = df
    else:
        top_sequences = df.head(top_n)

    sequences = []
    for _, row in top_sequences.iterrows():
        start_idx = row['indice_inicial']
        end_idx = row['indice_final']
        sequence = tensor_with_timestamp[start_idx:end_idx + 1]

        if len(sequence) >= min_required_length:
            sequences.append(sequence)

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
    # Precisamos fazer o split para garantir que o test_set seja o mesmo do treino
    _, _, test_set = torch.utils.data.random_split(
        normalized_dataset,
        [train_len, val_len, test_len],
        generator=generator
    )

    test_load = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_load


def executeTest_Denormalize(loader, model, n_inputs=5):
    all_inputs_last_only = []
    all_inputs_complete = []
    all_outputs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for test_batch in loader:
            input_data, target_data = test_batch
            input_indices, input_images = input_data
            target_index, target_image = target_data

            if input_images.dim() == 4:
                input_images = input_images.unsqueeze(2)
            if target_image.dim() == 3:
                target_image = target_image.unsqueeze(1).unsqueeze(2)

            input_images = input_images.float().to(device)
            # O modelo retorna normalizado, vamos denormalizar depois
            test_outputs = model(input_images)

            # Denormalização (Log -> Real)
            input_images_denorm = log_denorm(input_images.cpu())
            test_outputs_denorm = log_denorm(test_outputs.cpu())
            target_image_denorm = log_denorm(
                target_image.float().cpu())  # Target original já vem normalizado do dataset? Assumindo que sim.

            batch_size = input_images.shape[0]

            for i in range(batch_size):
                # Dados para persistência (último frame de input)
                last_frame_idx = input_indices[n_inputs - 1][i].item()
                last_frame_image = input_images_denorm[i, -1]
                all_inputs_last_only.append((last_frame_idx, last_frame_image))

                # Dados completos de input para plot
                complete_frames = []
                for j in range(n_inputs):
                    frame_idx = input_indices[j][i].item()
                    frame_image = input_images_denorm[i, j]
                    complete_frames.append((frame_idx, frame_image))
                all_inputs_complete.append(complete_frames)

                target_idx = target_index[i].item()
                all_outputs.append((target_idx, test_outputs_denorm[i]))
                all_targets.append((target_idx, target_image_denorm[i]))

    return all_inputs_last_only, all_inputs_complete, all_outputs, all_targets


def plot_multiple_models_comparison(all_models_metrics, all_models_persist_metrics, output_dir):
    """
    Versão simplificada da sua função de plotagem de boxplots
    """
    import matplotlib
    matplotlib_version = matplotlib.__version__
    os.makedirs(output_dir, exist_ok=True)
    metrics_to_plot = ['mse', 'mae', 'rmse', 'r2', 'ssim']

    for metric in metrics_to_plot:
        model_names = []
        all_data = []
        means = []

        for model_name, metrics in all_models_metrics.items():
            # Persistência
            persist_metrics = all_models_persist_metrics.get(model_name, {})
            persist_data = persist_metrics.get(metric, [])
            if persist_data:
                persist_data_clean = [x for x in persist_data if not np.isnan(x)]
                if persist_data_clean:
                    model_names.append(f'Persist {model_name}')
                    all_data.append(persist_data_clean)
                    means.append(np.mean(persist_data_clean))

            # Modelo
            model_data = metrics.get(metric, [])
            if model_data:
                model_data_clean = [x for x in model_data if not np.isnan(x)]
                if model_data_clean:
                    model_names.append(model_name)
                    all_data.append(model_data_clean)
                    means.append(np.mean(model_data_clean))

        if not all_data: continue

        plt.figure(figsize=(max(10, len(model_names) * 1.5), 8))

        # Compatibilidade Matplotlib
        if tuple(map(int, matplotlib_version.split('.')[:2])) >= (3, 9):
            box = plt.boxplot(all_data, tick_labels=model_names, patch_artist=True)
        else:
            box = plt.boxplot(all_data, labels=model_names, patch_artist=True)

        colors = ['lightcoral' if 'Persist' in name else 'lightblue' for name in model_names]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Adicionar médias
        for i, mean_val in enumerate(means):
            y_pos = np.percentile(all_data[i], 75) * 1.05
            plt.text(i + 1, y_pos, f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.title(f'Comparação {metric.upper()}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_{metric}.png', dpi=300)
        plt.close()


# ==============================================================================
# 4. EXECUÇÃO DO TESTE
# ==============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)

    # Dicionários para armazenar resultados globais
    all_models_metrics = {}
    all_models_persist_metrics = {}

    for config in models_to_test:
        alias = config['alias']
        print(f"\n{'#' * 60}")
        print(f"AVALIANDO: {alias}")
        print(f"Carregando: {config['path_pth']}")
        print(f"{'#' * 60}")

        # 1. Recriar Arquitetura do Modelo
        # Nota: Precisamos instanciar o modelo vazio com os mesmos parâmetros de construção
        model = create_conv_lstm_model(
            n_input_frames=config['n_inputs'],
            dropout_rate=config['drop_rate'],
            init_scale=config['init_scale'],
            use_multi_layer=config['use_multi_layer']
        ).to(device)

        # 2. Carregar Pesos (State Dict)
        try:
            checkpoint = torch.load(config['path_pth'], map_location=device, weights_only=False)
            # Verifica se o checkpoint salvou o dict inteiro ou só o state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Pesos carregados com sucesso.")
        except FileNotFoundError:
            print(f"ERRO: Arquivo não encontrado: {config['path_pth']}")
            continue
        except Exception as e:
            print(f"ERRO ao carregar pesos: {e}")
            continue

        # 3. Carregar DataLoader de Teste
        print("Gerando DataLoader de teste...")
        test_loader = load_test_loader_only(
            sequences_csv_path=PATH_CSV_SEQ,
            tensor_path=PATH_TENSOR_PRECIP,
            timestamps_path=PATH_CSV_TIMESTAMPS,
            top_n=config['n_sequencias'],
            n_input_frames=config['n_inputs'],
            times_aread=config['n_stamps_aread'],
            batch_size=config['batch_size'],
            seed=config['seed']
        )

        # 4. Executar Inferência e Denormalização
        print("Executando inferência...")
        x_last, x_complete, y_pred, y_true = executeTest_Denormalize(
            test_loader, model, n_inputs=config['n_inputs']
        )

        # 5. Calcular Métricas
        print("Calculando métricas...")
        predict_metrics = evaluate_predictions(y_true, y_pred)
        persist_metrics = evaluate_predictions(y_true, x_last)

        # Salvar nos dicionários globais
        all_models_metrics[alias] = predict_metrics
        all_models_persist_metrics[alias] = persist_metrics

        # 6. Salvar Resultados Individuais
        model_out_dir = os.path.join(OUTPUT_DIR_ROOT, alias)
        os.makedirs(model_out_dir, exist_ok=True)

        # Print rápido
        print(f"MSE ({alias}): {predict_metrics['mse_mean']:.4f}")
        print(f"MSE (Persistência): {persist_metrics['mse_mean']:.4f}")

        # Plots individuais
        timestamps = load_timestamps(PATH_CSV_TIMESTAMPS)

        plot_metrics_comparison(
            predict_metrics, persist_metrics, alias,
            f"Persistência {config['n_stamps_aread'] * 10}min",
            model_name_dir=model_out_dir
        )

        plot_comprehensive_comparison(
            y_true, y_pred, x_complete, timestamps,
            model_name_dir=model_out_dir, top_n=3, n_inputs=config['n_inputs']
        )

    # ==============================================================================
    # 5. CONSOLIDAÇÃO DOS RESULTADOS
    # ==============================================================================
    print(f"\n{'#' * 60}")
    print("GERANDO COMPARAÇÃO FINAL ENTRE MODELOS")
    print(f"{'#' * 60}")

    if all_models_metrics:
        # Gráficos comparativos (Boxplots)
        plot_multiple_models_comparison(all_models_metrics, all_models_persist_metrics, OUTPUT_DIR_ROOT)

        # CSV Resumo
        summary_data = []
        for name, metrics in all_models_metrics.items():
            # Linha Modelo
            row = {'Modelo': name, 'Tipo': 'Predição'}
            for key, val in metrics.items():
                if '_mean' in key: row[key] = val
            summary_data.append(row)

            # Linha Persistência
            p_metrics = all_models_persist_metrics.get(name, {})
            p_row = {'Modelo': name, 'Tipo': 'Persistência'}
            for key, val in p_metrics.items():
                if '_mean' in key: p_row[key] = val
            summary_data.append(p_row)

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(os.path.join(OUTPUT_DIR_ROOT, 'resumo_final_comparativo.csv'), index=False)
        print(f"Avaliação concluída! Verifique a pasta: {OUTPUT_DIR_ROOT}")
    else:
        print("Nenhum modelo foi avaliado com sucesso.")