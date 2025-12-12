import torch


def log_norm(tensor):
    return torch.log1p(tensor)
def log_denorm(tensor):
    return torch.expm1(tensor)


def test_normalization_roundtrip(norm_func, denorm_func, tensor, tolerance=1e-6, test_name="Normalização"):
    """
    Testa o processo completo de normalização e desnormalização

    Args:
        norm_func: Função de normalização
        denorm_func: Função de desnormalização
        tensor: Tensor original para testar
        tolerance: Tolerância para considerar valores iguais
        test_name: Nome do teste para identificação
    """
    full_tensor = torch.load(tensor, weights_only=True)
    print(f"\n{'=' * 60}")
    print(f"TESTE: {test_name}")
    print(f"{'=' * 60}")

    # Salvar tensor original
    original = full_tensor.clone()
    print(f"Tensor original shape: {original.shape}")
    print(f"Tensor original - Min: {original.min():.6f}, Max: {original.max():.6f}")
    print(f"Tensor original - Média: {original.mean():.6f}, Std: {original.std():.6f}")

    # 1. Normalização
    normalized = norm_func(full_tensor)
    print(f"\n--- APÓS NORMALIZAÇÃO ---")
    print(f"Normalizado - Min: {normalized.min():.6f}, Max: {normalized.max():.6f}")
    print(f"Normalizado - Média: {normalized.mean():.6f}, Std: {normalized.std():.6f}")

    # 2. Desnormalização
    denormalized = denorm_func(normalized)
    print(f"\n--- APÓS DESNORMALIZAÇÃO ---")
    print(f"Desnormalizado - Min: {denormalized.min():.6f}, Max: {denormalized.max():.6f}")
    print(f"Desnormalizado - Média: {denormalized.mean():.6f}, Std: {denormalized.std():.6f}")

    # 3. Análise de diferenças
    differences = torch.abs(original - denormalized)

    print(f"\n--- ANÁLISE DE DIFERENÇAS ---")
    print(f"Maior diferença absoluta: {differences.max():.10f}")
    print(f"Média das diferenças: {differences.mean():.10f}")
    print(f"Std das diferenças: {differences.std():.10f}")

    # 4. Verificação de precisão
    max_diff = differences.max().item()
    mean_diff = differences.mean().item()

    print(f"\n--- RESULTADO DO TESTE ---")
    if max_diff <= tolerance and mean_diff <= tolerance:
        print(f"SUCESSO! Round-trip perfeito.")
        print(f"   Máxima diferença ({max_diff:.2e}) ≤ tolerância ({tolerance:.2e})")
        print(f"   Média diferenças ({mean_diff:.2e}) ≤ tolerância ({tolerance:.2e})")
    else:
        print(f"ATENÇÃO! Possíveis distorções detectadas.")
        print(f"   Máxima diferença: {max_diff:.2e} (tolerância: {tolerance:.2e})")
        print(f"   Média diferenças: {mean_diff:.2e} (tolerância: {tolerance:.2e})")

    # 5. Estatísticas detalhadas
    zero_diff_mask = differences == 0
    small_diff_mask = (differences > 0) & (differences <= 1e-10)
    significant_diff_mask = differences > 1e-10

    print(f"\n--- ESTATÍSTICAS DETALHADAS ---")
    print(f"Valores com diferença zero: {zero_diff_mask.sum().item()} / {original.numel()}")
    print(f"Valores com diferença ≤ 1e-10: {small_diff_mask.sum().item()} / {original.numel()}")
    print(f"Valores com diferença > 1e-10: {significant_diff_mask.sum().item()} / {original.numel()}")

    if significant_diff_mask.sum().item() > 0:
        print(f"Valores problemáticos:")
        problematic_indices = torch.nonzero(significant_diff_mask, as_tuple=True)
        for i in range(min(5, problematic_indices[0].shape[0])):
            idx = tuple(problematic_indices[j][i] for j in range(len(problematic_indices)))
            print(
                f"  Posição {idx}: Original={original[idx]:.6f}, Final={denormalized[idx]:.6f}, Diff={differences[idx]:.2e}")

    return {
        'success': max_diff <= tolerance and mean_diff <= tolerance,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'original': original,
        'normalized': normalized,
        'denormalized': denormalized,
        'differences': differences
    }



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

def denormalize_batch(batch, max_val):
    """Desnormaliza um batch de dados"""
    x, y = batch
    return x * max_val, y * max_val