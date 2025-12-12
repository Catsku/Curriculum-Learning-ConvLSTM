from torch.utils.data import Dataset
import torch
import csv


class MultiFrameNextTimestampDataset(Dataset):
    def __init__(self, tensor_data, transform=None, n_input_frames=5, n_timestamps_ahead=1):
        """
        Dataset para previsão temporal com múltiplos frames de entrada

        Args:
            tensor_data: Lista de (índice, tensor) com formato [(idx, tensor), ...]
            transform: Função de transformação/normalização
            n_input_frames: Número de frames consecutivos para input
            n_timestamps_ahead: Quantos timestamps à frente prever (apenas 1 frame de saída)
        """
        self.data = tensor_data  # Lista de (índice, tensor)
        self.n_input_frames = n_input_frames
        self.n_timestamps_ahead = n_timestamps_ahead
        self.transform = transform

        # Aplicar transformação se fornecida
        if transform is not None:
            # Extrair apenas os tensores para normalização
            tensors_only = [tensor for idx, tensor in self.data]
            stacked_tensors = torch.stack(tensors_only)
            self.normalized_tensors, self.max_val = transform(stacked_tensors)

            # Reconstruir dados com tensores normalizados
            self.transformed_data = []
            for i, (idx, _) in enumerate(self.data):
                self.transformed_data.append((idx, self.normalized_tensors[i]))
        else:
            self.transformed_data = self.data
            self.max_val = None

        # Verificação de segurança
        min_required_length = n_input_frames + n_timestamps_ahead
        if len(self.transformed_data) < min_required_length:
            raise ValueError(
                f"Tensor de entrada muito pequeno. Tempo={len(self.transformed_data)}, precisa de pelo menos {min_required_length} frames")

    def __len__(self):
        return len(self.transformed_data) - (self.n_input_frames + self.n_timestamps_ahead - 1)

    def __getitem__(self, idx):
        # Input: n_input_frames consecutivos
        input_frames = []
        input_indices = []

        for i in range(self.n_input_frames):
            frame_idx = idx + i
            current_idx, current_tensor = self.transformed_data[frame_idx]
            input_indices.append(current_idx)
            input_frames.append(current_tensor)

        # Output: 1 frame n_timestamps_ahead à frente
        output_idx = idx + self.n_input_frames + self.n_timestamps_ahead - 1
        output_current_idx, output_current_tensor = self.transformed_data[output_idx]

        # Stack dos frames de input
        input_stack = torch.stack(input_frames)  # [n_input_frames, channels, height, width]

        return (input_indices, input_stack), (output_current_idx, output_current_tensor)

    def extract_all_data(self):
        all_data = []
        for idx in range(len(self)):
            current_data, next_data = self.__getitem__(idx)
            all_data.append((current_data, next_data))
        return all_data


class NormalizedMultiFrameDataset(Dataset):
    def __init__(self, dataset_list, transform, n_input_frames=5, n_timestamps_ahead=1):
        """
        Dataset normalizado que combina múltiplos MultiFrameNextTimestampDataset

        Args:
            dataset_list: Lista de datasets individuais
            transform: Função de transformação/normalização
            n_input_frames: Número de frames de entrada
            n_timestamps_ahead: Quantos timestamps à frente prever
        """
        self.n_input_frames = n_input_frames
        self.n_timestamps_ahead = n_timestamps_ahead

        # Coletar todos os tensores para normalização global
        all_tensors = []
        for dataset in dataset_list:
            for idx, tensor in dataset:
                all_tensors.append(tensor)

        # Aplicar normalização global
        stacked_tensors = torch.stack(all_tensors)
        self.normalized_tensors, self.global_max_val = transform(stacked_tensors)

        # Reconstruir datasets normalizados
        self.normalized_samples = []
        tensor_ptr = 0

        for dataset in dataset_list:
            #Para cada sequencia de frames normalizados executa MultiFrameNextTimestampDataset, separando os 5 inputs e o timestamp futuro.
            dataset_length = len(dataset)
            normalized_dataset = []

            for i in range(dataset_length):
                idx, original_tensor = dataset[i]
                normalized_tensor = self.normalized_tensors[tensor_ptr]
                normalized_dataset.append((idx, normalized_tensor))
                tensor_ptr += 1

            # Criar MultiFrameDataset com dados normalizados
            multi_frame_dataset = MultiFrameNextTimestampDataset(
                normalized_dataset,
                transform=None,  # Já está normalizado
                n_input_frames=n_input_frames,
                n_timestamps_ahead=n_timestamps_ahead
            )

            # Extrair todas as amostras
            for sample in multi_frame_dataset.extract_all_data():
                self.normalized_samples.append(sample)

    def __len__(self):
        return len(self.normalized_samples)

    def __getitem__(self, idx):
        return self.normalized_samples[idx]

    @property
    def max_val(self):
        return self.global_max_val