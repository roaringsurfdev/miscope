import torch


def get_discrete_fourier_bases(period: int, device: str | None = None) -> torch.Tensor:
    n = period // 2
    bases = torch.ones(n, 2, period)

    for frequency in range(1, n):
        sampling_vector = torch.arange(period) * 2 * torch.pi * frequency / period

        bases[frequency, 0, :] = torch.sin(sampling_vector)
        bases[frequency, 1, :] = torch.cos(sampling_vector)

    return bases


def get_fourier_bases(size: int, device: str | None = None) -> tuple[torch.Tensor, list[str]]:
    # TODO:
    # 1. Move to device after normalizing.
    # Currently, normalization is calculated on device in order to reproduce the
    #   original experiment.
    # 2. For p >> 1000, consider using linspace-based computation for numerical stability
    # EX:
    #   theta_positions = torch.linspace(0, 2*torch.pi*(size-1)/size, size)
    #   torch.sin(frequency * theta_positions)
    # 3. Extend basis names to provide more information other than frequency index
    bases = []
    basis_names = []
    n = (size // 2) + 1

    bases.append(torch.ones(size))
    basis_names.append("Constant")

    for frequency in range(1, n):
        # theta = 2 * torch.pi * frequency / size
        # theta_range = torch.arange(size) * theta
        theta_range = torch.arange(size) * 2 * torch.pi * frequency / size

        bases.append(torch.sin(theta_range))
        # basis_names.append(f"sin({theta}), frequency index:{frequency}")
        basis_names.append(f"sin k={frequency}")

        bases.append(torch.cos(theta_range))
        # basis_names.append(f"cos({theta}), frequency index:{frequency}")
        basis_names.append(f"cos k={frequency}")

    bases = torch.stack(bases, dim=0)

    if device:
        bases = bases.to(device)

    bases = bases / bases.norm(dim=-1, keepdim=True)

    return bases, basis_names


def get_basis_coefficients(bases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    coefficients = bases @ weights
    return coefficients.norm(dim=-1)


def get_dominant_bases_indices(bases: torch.Tensor, weights: torch.Tensor) -> list[int]:
    # TODO: explore appropriate algorithms for determining dominance.
    # For now, a threshold of 1 will be used.
    threshold = 1.0
    coefficients = (bases @ weights).norm(dim=-1)
    is_dominant = coefficients > threshold
    dominant_bases = torch.argwhere(is_dominant)

    return dominant_bases.T[0].tolist()


def get_dominant_bases(bases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    dominant_indices = get_dominant_bases_indices(bases, weights)

    coefficients = bases @ weights
    dominant_bases = coefficients[dominant_indices]

    return dominant_bases
