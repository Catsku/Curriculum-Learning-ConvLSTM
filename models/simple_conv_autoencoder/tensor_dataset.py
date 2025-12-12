import pandas as pd
from torch.utils.data import Dataset

class Tensor3dDataset(Dataset):
    def __init__(self, tensor_data, transform=None):

        self.data = tensor_data #(tempo, largura, altura)
        self.transform = transform

    def __len__(self):
        return len(self.data)  # Tamanho da dimensão temporal (eixo 0)

    def __getitem__(self, idx):

        frame = self.data[idx]  # Pega o frame no índice idx (formato [height, width])

        #Aplica transformação
        if self.transform:
            frame = self.transform(frame)

        return frame, frame # (input, label)