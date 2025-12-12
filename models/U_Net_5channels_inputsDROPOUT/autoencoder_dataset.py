import pandas as pd
from torch.utils.data import Dataset

class Tensor3dDataset(Dataset):
    def __init__(self, tensor_data, transform=None):

        self.data = tensor_data #(tempo, largura, altura)
        self.transformed_tensor, self.max_val = transform(self.data)

    def __len__(self):
        return len(self.data)  # Tamanho da dimensão temporal (eixo 0)

    def __getitem__(self, idx):
        frame = self.transformed_tensor[idx]  # Pega o frame no índice idx (formato [height, width])
        return frame