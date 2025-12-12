import torch
import pandas as pd

csv_path= '../../data/csvs/precipitacao_bacia_tamanduatei_sp_2019_2023_10min.csv'
tensor= torch.load('../../data/tensores/complete/tensor_radar_SP_precipitacao.pt')

try:
    df = pd.read_csv(csv_path, usecols=[0])
    timestamps = pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    valid_timestamps = timestamps[~timestamps.isna()].reset_index(drop=True)

    if len(valid_timestamps) != tensor.size(0):
        raise ValueError("Número de timestamps não corresponde ao tamanho temporal do tensor")
    else:
        valid_timestamps.to_csv('./csvs/timestamps.csv', index=False)

except Exception as e:
    raise ValueError(f"Erro ao carregar timestamps: {str(e)}")

