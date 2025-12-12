from torch.utils.data import Dataset
import torch
import csv

class NextTimestampDataset(Dataset):
    def __init__(self, tensor_data, transform=None, n_inputs=5, n_temps=1):
        self.data = tensor_data #(tempo, largura, altura)
        self.n_temps = n_temps
        self.n_inputs = n_inputs
        if transform is not None:
            self.transformed_tensor, self.max_val = transform(self.data)
        else:
            self.transformed_tensor = self.data

        # Verificação de segurança
        if len(self.data) <= self.n_temps:
            raise ValueError(f"Tensor de entrada muito pequeno. Tempo={len(self.data)}, mas n_temps={n_temps}")

    def __len__(self):
        return  len(self.transformed_tensor) - self.n_temps  # Tamanho da dimensão temporal (eixo 0)
        #Evita os ultimos n_temps timestamp para que seja o output esperado para o penultimo timestamp

    def __getitem__(self, idx):
        idx1, frame = self.transformed_tensor[idx]  # Pega o frame no índice idx (formato [height, width])
        idx2, next_frame = self.transformed_tensor[idx + self.n_temps]
        return (idx1,frame), (idx2,next_frame)

    def extract_all_data(self):
        all_data = []
        for idx in range(len(self.data) - self.n_temps):
            current_data, next_data = self.__getitem__(idx)

            # Extrai índices e tensores
            current_idx, current_tensor = current_data
            next_idx, next_tensor = next_data

            all_data.append(((current_idx, current_tensor), (next_idx, next_tensor)))
        return all_data


class NormalizedConcatDataset(Dataset):
    def __init__(self, concat_dataset, transform):
        self.concat_dataset = concat_dataset
        self.transform = transform

        #print(f"concat_dataset:\n\n {self.concat_dataset}")
        # Passo 1: Coletar apenas os tensores para normalização
        tensors_to_normalize = []
        for sample in concat_dataset:
            # Assumindo que cada sample é ((idx_x, tensor_x), (idx_y, tensor_y))
            tensors_to_normalize.append(sample[0][1])  # tensor_x
            tensors_to_normalize.append(sample[1][1])  # tensor_y


        # Passo 2: Empilhar e normalizar apenas os tensores
        stacked_tensors = torch.stack(tensors_to_normalize)
        self.normalized_tensors, self.global_max_val = transform(stacked_tensors)

        # Passo 3: Reconstruir a estrutura com índices
        self.normalized_samples = []
        ptr = 0
        for original_sample in concat_dataset:
            (x_idx, _), (y_idx, _) = original_sample
            x_norm = self.normalized_tensors[ptr]
            y_norm = self.normalized_tensors[ptr + 1]
            self.normalized_samples.append(((x_idx, x_norm), (y_idx, y_norm)))
            ptr += 2


    def __len__(self):
        return len(self.normalized_samples)

    def __getitem__(self, idx):
        return self.normalized_samples[idx]  # Retorna ((idx, x_norm), (idx, y_norm)

    @property
    def max_val(self):
        return self.global_max_val