# %% imports
import sys
import os

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family

# %% helper functions
def compute_ideal_fourier_spectrum(p):
    """
    Compute the theoretical Fourier spectrum for f(a,b) = (a+b) mod p
    """
    # Create the addition table
    addition_table = np.zeros((p, p))
    for a in range(p):
        for b in range(p):
            addition_table[a, b] = (a + b) % p

    # Compute 2D FFT
    fourier = np.fft.fft2(addition_table)
    power_spectrum = np.abs(fourier) ** 2

    # Normalize
    power_spectrum = power_spectrum / power_spectrum.sum()

    return power_spectrum

def print_ideal_fourier_spectrum_list(p_list):
    for p in p_list:
        spectrum = compute_ideal_fourier_spectrum(p)
        # Find dominant components
        threshold = 0.01  # 1% of total power
        dominant_components = np.argwhere(spectrum > threshold * spectrum.max())
        print(f"Prime {p}: Dominant (k,ℓ) components:")
        for k, l in dominant_components[:10]:  # Show top 10  # noqa: E741
            print(f"  ({k}, {l}): {spectrum[k, l]:.4f}")    

# %% load model and list variants
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)
model = variant.at(9000)

# %% analyze probe
probe = variant.make_probe([[3, 29]])
logits, cache = model.run_with_cache(probe)
for param_name, param in cache.items():
    print(param_name, param.shape)
# %%
W_E = model.embed.W_E[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)

# %% artifacts
artifacts = variant.artifacts

# %% show ideal frequencies
p_list = [97, 101, 103]
print_ideal_fourier_spectrum_list(p_list)

p_list = [107, 109, 113, 127]
print_ideal_fourier_spectrum_list(p_list)

# %%
