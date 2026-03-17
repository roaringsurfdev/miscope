# Sections of this baseline is taken from the Grokking_demo.ipynb colab at
# https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportOperatorIssue=false

# %% imports
import os

import einops
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn.functional as F
import transformer_lens.utils as utils

import FourierEvaluation
from miscope.visualization import line
from ModuloAdditionSpecification import ModuloAdditionSpecification

# %% initialize environment and model
p = 113
seed = 999
data_seed = 999
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the location to save the model, using a relative path
MODEL_PATH = os.path.join(
    os.getcwd(), "results", "modulo_addition_source", f"p{p}_seed{seed}_dseed{data_seed}"
)

model_spec = ModuloAdditionSpecification(MODEL_PATH, p, device, seed=seed, data_seed=data_seed)

TRAIN_MODEL = True

# %% define the data

train_data, train_labels, test_data, test_labels, train_indices, test_indices = (
    model_spec.generate_training_data()
)
model = model_spec.create_model()

# %% baseline (pre-training)
pre_train_logits = model(train_data)
pre_train_loss = model_spec.loss_function(pre_train_logits, train_labels)
print(f"Pre-training loss on training data: {pre_train_loss}")
pre_train_test_logits = model(test_data)
pre_train_test_loss = model_spec.loss_function(pre_train_test_logits, test_labels)
print(f"Pre-training loss on test data: {pre_train_test_loss}")
print(f"Uniform loss: {np.log(p)}")  # Uniform cross-entropy -log(1/p)


# %% model training
model = model_spec.create_model()

if TRAIN_MODEL:
    model = model_spec.train()
else:
    model = model_spec.load_from_file()

# %% Show model training

pio.renderers.default = "vscode"
pio.templates["plotly"].layout.xaxis.title.font.size = 20
pio.templates["plotly"].layout.yaxis.title.font.size = 20
pio.templates["plotly"].layout.title.font.size = 30


line(
    [model_spec.train_losses[::100], model_spec.test_losses[::100]],
    x=np.arange(0, len(model_spec.train_losses), 100),
    xaxis="Epoch",
    yaxis="Loss",
    log_y=True,
    title="Training Curve for Modular Addition",
    line_labels=["train", "test"],
    toggle_x=True,
    toggle_y=True,
)


# %% Analyis - helper functions
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs).show(renderer)


# %% Analysis - run model with dataset
probe_logits, probe_cache = model.run_with_cache(model_spec.dataset)
print(probe_logits.numel())

# %% Analysis - get key weight matrices
W_E = model.embed.W_E[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)

# %% Analysis - Looking at activations
neuron_acts = probe_cache["post", 0, "mlp"][:, -1, :]

# %% Analysis - Show parameter shapes
for param_name, param in probe_cache.items():
    print(param_name, param.shape)

# %% Analysis - Average Attention Pattern per Head
imshow(
    probe_cache["pattern", 0].mean(dim=0)[:, -1, :],
    title="Average Attention Pattern per Head",
    xaxis="Source",
    yaxis="Head",
    x=["a", "b", "="],
)
# %% Analysis - Attention Pattern per Head (single example, index 5)
imshow(
    probe_cache["pattern", 0][5][:, -1, :],
    title="Attention Pattern per Head (example index 5)",
    xaxis="Source",
    yaxis="Head",
    x=["a", "b", "="],
)
# %% Analysis - Attention for Head 0 from a b -> =
imshow(
    probe_cache["pattern", 0][:, 0, -1, 0].reshape(p, p),
    title="Attention for Head 0 from a -> =",
    xaxis="b",
    yaxis="a",
)
# %% Analysis - Attention for Head 0 from a -> =
imshow(
    einops.rearrange(probe_cache["pattern", 0][:, :, -1, 0], "(a b) head -> head a b", a=p, b=p),
    title="Attention for Head 0 from a b -> =",
    xaxis="b",
    yaxis="a",
    facet_col=0,
)

# %% Experiment - Downsampled attention patterns (bilinear interpolation)
# Testing whether attention heads also show multi-scale structure like neurons
attn_pattern_to_a = einops.rearrange(
    probe_cache["pattern", 0][:, :, -1, 0], "(a b) head -> head a b", a=p, b=p
)
attn_4d = attn_pattern_to_a.unsqueeze(1).float()  # (head, 1, a, b)

for target_size in [28, 56]:
    interpolated = F.interpolate(
        attn_4d, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    imshow(
        interpolated.squeeze(1),
        title=f"Attention to 'a' per head (bilinear to {target_size}x{target_size})",
        xaxis="b",
        yaxis="a",
        facet_col=0,
    )

# Also test attention to 'b' position
attn_pattern_to_b = einops.rearrange(
    probe_cache["pattern", 0][:, :, -1, 1], "(a b) head -> head a b", a=p, b=p
)
attn_4d_b = attn_pattern_to_b.unsqueeze(1).float()

for target_size in [28, 56]:
    interpolated = F.interpolate(
        attn_4d_b, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    imshow(
        interpolated.squeeze(1),
        title=f"Attention to 'b' per head (bilinear to {target_size}x{target_size})",
        xaxis="b",
        yaxis="a",
        facet_col=0,
    )

# %% Analysis - Single Value Decomposition
U, S, Vh = torch.svd(W_E)
line(S, title="Singular Values")
imshow(U, title="Principal Components on the Input")

# %% Analysis - Single Value Decomposition with Control: Random Gaussian matrix
U, S, Vh = torch.svd(torch.randn_like(W_E))
line(S, title="Singular Values - Random Control")
imshow(U, title="Principal Components - Random Control")

# %% Explaining the Algorithm - The Embedding is a lookup table
U, S, Vh = torch.svd(W_E)
line(U[:, :8].T, title="Principal Components of the embedding", xaxis="Input Vocabulary")

# %% Explaining the Algorithm - Fourier basis setup
fourier_basis, fourier_basis_names = FourierEvaluation.get_fourier_bases(p, device)
dominant_indices = FourierEvaluation.get_dominant_bases_indices(fourier_basis, W_E)
dominant_bases = FourierEvaluation.get_dominant_bases(fourier_basis, W_E)
print(f"Dominant Bases: {dominant_bases}")
print(f"Dominant Bases (indices): {dominant_indices}")

imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)

line(
    fourier_basis[:8],
    xaxis="Input",
    line_labels=fourier_basis_names[:8],
    title="First 8 Fourier Components",
)
line(
    fourier_basis[25:29],
    xaxis="Input",
    line_labels=fourier_basis_names[25:29],
    title="Middle Fourier Components",
)
imshow(fourier_basis @ fourier_basis.T, title="All Fourier Vectors are Orthogonal")

# %% Explaining the Algorithm - Norms of Embedding in Fourier Basis
line(
    (fourier_basis @ W_E).norm(dim=-1),
    xaxis="Fourier Component",
    x=fourier_basis_names,
    title="Norms of Embedding in Fourier Basis",
)

# %% Explaining the Algorithm - Key frequencies
key_freq_indices = dominant_indices
key_freqs = dominant_indices[1::2]
print(f"New key_freqs:\n{key_freqs}")

fourier_embed = fourier_basis @ W_E
key_fourier_embed = fourier_embed[key_freq_indices]
print("key_fourier_embed", key_fourier_embed.shape)
imshow(
    key_fourier_embed @ key_fourier_embed.T, title="Dot Product of embedding of key Fourier Terms"
)

# %% Explaining the Algorithm - Key frequencies - Cos
key_freq_indices_cos = dominant_indices[1::2]
print(f"Refactored cos frequency indices: {key_freq_indices_cos}")
line(
    fourier_basis[key_freq_indices_cos], title="Cos of key freqs", line_labels=key_freq_indices_cos
)

# %% Explaining the Algorithm - Constructive Interference
line(fourier_basis[key_freq_indices_cos].mean(0), title="Constructive Interference")

# %% Analyse Neuron Activations
# Visualization: First 5 neuron activations
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p),
    title="First 5 neuron activations",
    xaxis="b",
    yaxis="a",
    facet_col=0,
)
# Visualization: First Neuron Activation
imshow(
    einops.rearrange(neuron_acts[:, 0], "(a b) -> a b", a=p, b=p),
    title="First neuron act",
    xaxis="b",
    yaxis="a",
)

# %% Experiment - Downsampled neuron activations (average pooling with einops)
# Exploring whether lower resolution makes periodic structure more visible
downsample_factor = 4  # Adjust to taste: 2, 4, or 8

neuron_acts_2d = einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p)

# Trim to make dimensions divisible by downsample_factor
trim_size = (p // downsample_factor) * downsample_factor
neuron_acts_trimmed = neuron_acts_2d[:, :trim_size, :trim_size]

# Average pooling using einops reduce
neuron_acts_downsampled = einops.reduce(
    neuron_acts_trimmed,
    "neuron (h h2) (w w2) -> neuron h w",
    "mean",
    h2=downsample_factor,
    w2=downsample_factor,
)

imshow(
    neuron_acts_downsampled,
    title=f"First 5 neuron activations (downsampled {downsample_factor}x)",
    xaxis="b",
    yaxis="a",
    facet_col=0,
)

# %% Experiment - Downsampled neuron activations (torch avg_pool2d)
neuron_acts_2d = einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p)

# avg_pool2d expects (N, C, H, W) - treat neurons as batch dimension
neuron_acts_4d = neuron_acts_2d.unsqueeze(1)  # (neuron, 1, a, b)

downsample_factor = 4
neuron_acts_pooled = F.avg_pool2d(neuron_acts_4d, kernel_size=downsample_factor).squeeze(1)

imshow(
    neuron_acts_pooled,
    title=f"First 5 neuron activations (avg_pool2d {downsample_factor}x)",
    xaxis="b",
    yaxis="a",
    facet_col=0,
)

# %% Experiment - Compare different downsample factors
for factor in [2, 4, 8]:
    neuron_acts_4d = einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron 1 a b", a=p, b=p)
    pooled = F.avg_pool2d(neuron_acts_4d, kernel_size=factor).squeeze(1)
    imshow(
        pooled,
        title=f"First 5 neuron activations ({factor}x downsample)",
        xaxis="b",
        yaxis="a",
        facet_col=0,
    )

# %% Experiment - Bilinear interpolation (closer to browser squish behavior)
# This uses continuous interpolation rather than block averaging
neuron_acts_2d = einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p)
neuron_acts_4d = neuron_acts_2d.unsqueeze(1).float()  # (neuron, 1, a, b)

for target_size in [28, 56]:
    interpolated = F.interpolate(
        neuron_acts_4d, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    imshow(
        interpolated.squeeze(1),
        title=f"First 5 neuron activations (bilinear to {target_size}x{target_size})",
        xaxis="b",
        yaxis="a",
        facet_col=0,
    )

# %% Experiment - Blur then sample (anti-aliased downsampling)
# Apply smoothing before downsampling to reduce aliasing artifacts
# Using a box blur via avg_pool2d with stride=1, then downsampling


def blur_then_sample(tensor_4d: torch.Tensor, blur_kernel: int, sample_factor: int) -> torch.Tensor:
    """Apply blur (avg pool with stride 1) then downsample."""
    # Blur: average pool with padding to maintain size
    padding = blur_kernel // 2
    blurred = F.avg_pool2d(tensor_4d, kernel_size=blur_kernel, stride=1, padding=padding)
    # Sample: take every nth pixel
    sampled = blurred[:, :, ::sample_factor, ::sample_factor]
    return sampled


neuron_acts_4d = einops.rearrange(
    neuron_acts[:, :5], "(a b) neuron -> neuron 1 a b", a=p, b=p
).float()

for blur_k, sample_f in [(3, 4), (5, 4), (7, 4), (5, 2)]:
    result = blur_then_sample(neuron_acts_4d, blur_k, sample_f).squeeze(1)
    imshow(
        result,
        title=f"First 5 neurons (blur={blur_k}, sample={sample_f}x)",
        xaxis="b",
        yaxis="a",
        facet_col=0,
    )

# %% Experiment - Coarseness Metric: Frequency Energy Ratio
# Measures what fraction of total energy is in low frequencies (center of 2D FFT)


def compute_low_freq_energy_ratio(
    activation_map_2d: torch.Tensor, radius_fraction: float = 0.125
) -> float:
    """Compute ratio of energy in low frequencies vs total energy.

    Args:
        activation_map_2d: 2D tensor (h, w)
        radius_fraction: Fraction of spectrum to consider "low frequency" (default 12.5%)

    Returns:
        Ratio from 0-1. Higher = more coarse structure.
    """
    h, w = activation_map_2d.shape
    fft_2d = torch.fft.fft2(activation_map_2d)
    fft_shifted = torch.fft.fftshift(fft_2d)  # Center the low frequencies
    power = torch.abs(fft_shifted) ** 2

    # Create circular mask for low frequencies (center of shifted FFT)
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_fraction)
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y_grid, x_grid = y_grid.to(activation_map_2d.device), x_grid.to(activation_map_2d.device)
    mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= radius**2

    low_freq_energy = power[mask].sum()
    total_energy = power.sum()

    return (low_freq_energy / total_energy).item()


# Compute for first 10 neurons
num_neurons_to_analyze = min(10, neuron_acts.shape[1])
neuron_coarseness_fft = []
for i in range(num_neurons_to_analyze):
    act_map = einops.rearrange(neuron_acts[:, i], "(a b) -> a b", a=p, b=p)
    ratio = compute_low_freq_energy_ratio(act_map)
    neuron_coarseness_fft.append(ratio)

line(
    torch.tensor(neuron_coarseness_fft),
    title="Neuron Coarseness (Low-Freq Energy Ratio)",
    xaxis="Neuron Index",
)

# %% Experiment - Coarseness Metric: Variance Preservation
# Measures how much variance survives downsampling and reconstruction


def compute_variance_preservation(
    activation_map_2d: torch.Tensor, downsample_size: int = 14
) -> float:
    """Compute ratio of variance preserved after downsample/upsample cycle.

    Args:
        activation_map_2d: 2D tensor (h, w)
        downsample_size: Target size for downsampling

    Returns:
        Ratio from 0-1. Higher = more coarse structure (survives downsampling).
    """
    original_var = activation_map_2d.var().item()
    if original_var == 0:
        return 0.0

    h, w = activation_map_2d.shape
    map_4d = activation_map_2d.unsqueeze(0).unsqueeze(0).float()

    downsampled = F.interpolate(
        map_4d, size=(downsample_size, downsample_size), mode="bilinear", align_corners=False
    )
    upsampled = F.interpolate(downsampled, size=(h, w), mode="bilinear", align_corners=False)

    preserved_var = upsampled.var().item()
    return preserved_var / original_var


# Compute for first 10 neurons
neuron_coarseness_var = []
for i in range(num_neurons_to_analyze):
    act_map = einops.rearrange(neuron_acts[:, i], "(a b) -> a b", a=p, b=p)
    ratio = compute_variance_preservation(act_map)
    neuron_coarseness_var.append(ratio)

line(
    torch.tensor(neuron_coarseness_var),
    title="Neuron Coarseness (Variance Preservation)",
    xaxis="Neuron Index",
)

# %% Experiment - Compare both coarseness metrics
# Scatter plot to see if they correlate
scatter(
    torch.tensor(neuron_coarseness_fft),
    torch.tensor(neuron_coarseness_var),
    xaxis="Low-Freq Energy Ratio",
    yaxis="Variance Preservation",
    title="Coarseness Metrics Comparison (first 10 neurons)",
)

# %% Experiment - Coarseness metrics for attention heads
# Apply same metrics to attention patterns

attn_to_a = einops.rearrange(
    probe_cache["pattern", 0][:, :, -1, 0], "(a b) head -> head a b", a=p, b=p
)

head_coarseness_fft = []
head_coarseness_var = []
for head_idx in range(attn_to_a.shape[0]):
    attn_map = attn_to_a[head_idx]
    head_coarseness_fft.append(compute_low_freq_energy_ratio(attn_map))
    head_coarseness_var.append(compute_variance_preservation(attn_map))

print("Attention Head Coarseness (to 'a'):")
for i, (fft_score, var_score) in enumerate(zip(head_coarseness_fft, head_coarseness_var)):
    print(f"  Head {i}: FFT={fft_score:.3f}, Var={var_score:.3f}")

# %% Experiment - Coarseness without DC component
# The DC component (mean) can dominate the low-freq energy ratio
# Try excluding it to see if that better captures blob vs grid distinction


def compute_low_freq_energy_ratio_no_dc(
    activation_map_2d: torch.Tensor, radius_fraction: float = 0.125
) -> float:
    """Compute low-freq energy ratio excluding DC (zero frequency) component."""
    h, w = activation_map_2d.shape
    fft_2d = torch.fft.fft2(activation_map_2d)
    fft_shifted = torch.fft.fftshift(fft_2d)
    power = torch.abs(fft_shifted) ** 2

    cy, cx = h // 2, w // 2

    # Zero out DC component
    power[cy, cx] = 0

    radius = int(min(h, w) * radius_fraction)
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y_grid, x_grid = y_grid.to(activation_map_2d.device), x_grid.to(activation_map_2d.device)
    mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= radius**2

    low_freq_energy = power[mask].sum()
    total_energy = power.sum()

    if total_energy == 0:
        return 0.0
    return (low_freq_energy / total_energy).item()


# Compare with and without DC for neurons
print("Neuron Coarseness - FFT with vs without DC:")
for i in range(num_neurons_to_analyze):
    act_map = einops.rearrange(neuron_acts[:, i], "(a b) -> a b", a=p, b=p)
    with_dc = compute_low_freq_energy_ratio(act_map)
    without_dc = compute_low_freq_energy_ratio_no_dc(act_map)
    print(f"  Neuron {i}: with_DC={with_dc:.3f}, no_DC={without_dc:.3f}")

# Compare with and without DC for attention heads
print("\nAttention Head Coarseness - FFT with vs without DC (to 'a'):")
for head_idx in range(attn_to_a.shape[0]):
    attn_map = attn_to_a[head_idx]
    with_dc = compute_low_freq_energy_ratio(attn_map)
    without_dc = compute_low_freq_energy_ratio_no_dc(attn_map)
    print(f"  Head {head_idx}: with_DC={with_dc:.3f}, no_DC={without_dc:.3f}")

# %% Experiment - Gradient magnitude metric
# Alternative approach: measure edge density (high gradient = fine structure)


def compute_gradient_magnitude(activation_map_2d: torch.Tensor) -> float:
    """Compute average gradient magnitude (edge density).

    Returns:
        Average absolute gradient. Higher = more edges = finer structure.
    """
    dx = activation_map_2d[:, 1:] - activation_map_2d[:, :-1]
    dy = activation_map_2d[1:, :] - activation_map_2d[:-1, :]
    return ((dx.abs().mean() + dy.abs().mean()) / 2).item()


print("\nGradient Magnitude (higher = finer structure):")
print("Neurons:")
for i in range(num_neurons_to_analyze):
    act_map = einops.rearrange(neuron_acts[:, i], "(a b) -> a b", a=p, b=p)
    grad = compute_gradient_magnitude(act_map)
    print(f"  Neuron {i}: grad={grad:.4f}")

print("\nAttention Heads (to 'a'):")
for head_idx in range(attn_to_a.shape[0]):
    attn_map = attn_to_a[head_idx]
    grad = compute_gradient_magnitude(attn_map)
    print(f"  Head {head_idx}: grad={grad:.4f}")

# %% Analyse plain frequency heat maps
# Visualization: Cos a * Cos b
for i in range(len(key_freqs)):
    fourier_basis_index = key_freq_indices_cos[i]
    fourer_basis_frequency = key_freqs[i] // 2
    imshow(
        fourier_basis[fourier_basis_index][None, :] * fourier_basis[fourier_basis_index][:, None],
        title=f"Cos {fourer_basis_frequency}a * cos {fourer_basis_frequency}b",
    )
# Visualization: Cos a * Constant
for i in range(len(key_freqs)):
    fourier_basis_index = key_freq_indices_cos[i]
    fourer_basis_frequency = key_freqs[i] // 2
    imshow(
        fourier_basis[fourier_basis_index][None, :] * fourier_basis[0][:, None],
        title=f"Cos {fourer_basis_frequency}a * const",
    )

# %% Analysis - 2D Fourier Transform of neurons 0-4
for i in range(5):
    imshow(
        fourier_basis @ neuron_acts[:, i].reshape(p, p) @ fourier_basis.T,
        title=f"2D Fourier Transformer of neuron {i}",
        xaxis="b",
        yaxis="a",
        x=fourier_basis_names,
        y=fourier_basis_names,
    )
# %% Analysis - 2D Fourier Transform of random control
imshow(
    fourier_basis @ torch.randn_like(neuron_acts[:, 0]).reshape(p, p) @ fourier_basis.T,
    title="2D Fourier Transformer of RANDOM",
    xaxis="b",
    yaxis="a",
    x=fourier_basis_names,
    y=fourier_basis_names,
)

# %% Analysis - Neuron clusters
fourier_neuron_acts = (
    fourier_basis
    @ einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=p, b=p)
    @ fourier_basis.T
)
# Center these by removing the mean - doesn't matter!
fourier_neuron_acts[:, 0, 0] = 0.0
print("fourier_neuron_acts", fourier_neuron_acts.shape)
neuron_freq_norm = torch.zeros(p // 2, model.cfg.d_mlp).to(device)
for freq in range(0, p // 2):
    for x in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
        for y in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
            neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y] ** 2

neuron_freq_norm = neuron_freq_norm / fourier_neuron_acts.pow(2).sum(dim=[-1, -2])[None, :]
# Visualization: Neuron Frac Explained by Freq
imshow(
    neuron_freq_norm,
    xaxis="Neuron",
    yaxis="Freq",
    y=torch.arange(1, p // 2 + 1),
    title="Neuron Frac Explained by Freq",
)

# Visualization: Max Neuron Frac Explained over Freqs
line(
    neuron_freq_norm.max(dim=0).values.sort().values,
    xaxis="Neuron",
    title="Max Neuron Frac Explained over Freqs",
)
# %% Analysis - Read Off the Neuron-Logit Weights to Interpret
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
line(
    (W_logit @ fourier_basis.T).norm(dim=0),
    x=fourier_basis_names,
    title="W_logit in the Fourier Basis",
)
# %%
for i in range(len(key_freqs)):
    frequency = key_freqs[i] // 2
    neurons = neuron_freq_norm[frequency - 1] > 0.85
    line(
        (W_logit[neurons] @ fourier_basis.T).norm(dim=0),
        x=fourier_basis_names,
        title=f"W_logit for freq {frequency} neurons in the Fourier Basis",
    )
# %%
for i in range(len(key_freqs)):
    frequency = key_freqs[i]
    sin_frequency = frequency - 1
    W_logit_fourier = W_logit @ fourier_basis
    neurons_sin_i = W_logit_fourier[:, sin_frequency]
    line(neurons_sin_i, title=f"W_logit @ fourier_basis for Frequency {frequency // 2}")
# %% Analysis - Fourier heatmap over inputs for sin9c
freq = 55
neurons_sin_9 = W_logit_fourier[:, 2 * freq - 1]
inputs_sin_9c = neuron_acts @ neurons_sin_9
imshow(
    fourier_basis @ inputs_sin_9c.reshape(p, p) @ fourier_basis.T,
    title="Fourier Heatmap over inputs for sin17c",
    x=fourier_basis_names,
    y=fourier_basis_names,
)


# %% Experiment - All 512 neurons visualization (composite image)
# Display all MLP neurons as a single composite image for artifact-free rendering
# To identify a neuron: neuron_idx = row * n_cols + col (0-indexed)

n_neurons = 512
n_cols = 32
n_rows = (n_neurons + n_cols - 1) // n_cols  # 16 rows
cell_size = p  # 113 for full resolution

# Reshape all neuron activations to 2D maps
all_neuron_maps = utils.to_numpy(
    einops.rearrange(neuron_acts[:, :n_neurons], "(a b) neuron -> neuron a b", a=p, b=p)
)

# Build composite image
composite_raw = np.zeros((n_rows * cell_size, n_cols * cell_size))
for idx in range(n_neurons):
    row, col = idx // n_cols, idx % n_cols
    composite_raw[
        row * cell_size : (row + 1) * cell_size,
        col * cell_size : (col + 1) * cell_size,
    ] = all_neuron_maps[idx]

# Display as single heatmap
fig = go.Figure(
    go.Heatmap(
        z=composite_raw,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
    )
)
fig.update_layout(
    title=f"All {n_neurons} MLP Neuron Activations ({n_rows}x{n_cols} grid, {cell_size}x{cell_size} per neuron)",
    height=800,
    width=1600,
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# %% Experiment - All 512 neurons visualization (downsampled composite)
# Same composite approach with bilinear downsampling to reveal coarse structure
# To identify a neuron: neuron_idx = row * n_cols + col (0-indexed)

target_size = 28  # Downsample from 113x113 to 28x28
n_neurons = 512
n_cols = 32
n_rows = (n_neurons + n_cols - 1) // n_cols

# Reshape and apply bilinear interpolation
all_neuron_maps_tensor = einops.rearrange(
    neuron_acts[:, :n_neurons], "(a b) neuron -> neuron a b", a=p, b=p
)
all_neuron_maps_4d = all_neuron_maps_tensor.unsqueeze(1).float()
interpolated = utils.to_numpy(
    F.interpolate(
        all_neuron_maps_4d, size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze(1)
)

# Build composite image
composite_downsampled = np.zeros((n_rows * target_size, n_cols * target_size))
for idx in range(n_neurons):
    row, col = idx // n_cols, idx % n_cols
    composite_downsampled[
        row * target_size : (row + 1) * target_size,
        col * target_size : (col + 1) * target_size,
    ] = interpolated[idx]

# Display as single heatmap
fig = go.Figure(
    go.Heatmap(
        z=composite_downsampled,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
    )
)
fig.update_layout(
    title=f"All {n_neurons} MLP Neuron Activations - Downsampled ({n_rows}x{n_cols} grid, {target_size}x{target_size} per neuron)",
    height=800,
    width=1600,
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()
# %%
