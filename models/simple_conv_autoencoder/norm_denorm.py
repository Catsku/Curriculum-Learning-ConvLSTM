import torch

def normalize_3d_tensor(input_tensor):
    # Achata o tensor para cálculo do máximo (funciona para 3D ou 4D)
    if input_tensor.is_contiguous():
        flattened_tensor = input_tensor.view(-1)
    else:
        flattened_tensor = input_tensor.flatten()

    # Encontra o valor máximo absoluto
    max_val = torch.max(torch.abs(flattened_tensor)).item()

    # Normaliza o tensor
    normalized_tensor = input_tensor / max_val

    # Garante que os valores estejam no intervalo [0, 1]
    normalized_tensor = torch.clamp(normalized_tensor, 0.0, 1.0)

    return normalized_tensor, max_val


# Função para desnormalizar
def denormalize_3d_tensor(normalized_tensor, max_val):
    return normalized_tensor * max_val