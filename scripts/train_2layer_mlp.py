"""Train a 2-layer MLP on modular addition.

Standalone training script for TwoLayerMLPFamily variants.
Produces the same checkpoint format (safetensors) and directory structure
as the 1-layer transformer family, but writes an MLP-specific config.json.

Usage:
    python scripts/train_2layer_mlp.py --prime 113 --seed 999 --data-seed 598
    python scripts/train_2layer_mlp.py --prime 113 --seed 999 --data-seed 598 --d-hidden 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import tqdm.auto as tqdm
from safetensors.torch import save_file

# Ensure project root is on the path when run as a script
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from miscope.config import get_config
from miscope.families.implementations.two_layer_mlp import load_two_layer_mlp_family
from miscope.families.variant import Variant


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for classification (no sequence dimension)."""
    log_probs = logits.to(torch.float64).log_softmax(dim=-1)
    return -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean()


def _save_config(
    variant_dir: Path,
    params: dict,
    d_hidden: int,
    training_fraction: float,
) -> None:
    """Write MLP-specific config.json (no TransformerLens fields)."""
    config = {
        "architecture": "two_layer_mlp",
        "vocab_size": params["prime"],
        "d_hidden": d_hidden,
        "act_fn": "relu",
        **params,
        "training_fraction": training_fraction,
    }
    with open(variant_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def train(
    prime: int,
    seed: int,
    data_seed: int,
    d_hidden: int = 512,
    training_fraction: float = 0.3,
    device: str | None = None,
) -> None:
    """Train a 2-layer MLP variant and save checkpoints.

    Args:
        prime: Modulus for the addition task
        seed: Random seed for model initialization
        data_seed: Random seed for train/test split
        d_hidden: Hidden layer width (default 512)
        training_fraction: Fraction of data for training (default 0.3)
        device: Device string ('cpu', 'cuda', or None for auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_config()
    family = load_two_layer_mlp_family(cfg.model_families_dir)

    params = {"prime": prime, "seed": seed, "data_seed": data_seed}
    # Override d_hidden from CLI if different from family.json default
    family._config["architecture"]["d_hidden"] = d_hidden

    variant = Variant(family, params, cfg.results_dir)
    variant.ensure_directories()

    model = family.create_model(params, device=device)
    train_data, train_labels, test_data, test_labels, train_indices, test_indices = (
        family.generate_training_dataset(params, training_fraction=training_fraction, data_seed=data_seed, device=device)
    )

    training_config = family.get_training_config()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=training_config["betas"],
    )

    num_epochs: int = training_config["num_epochs"]
    checkpoint_epochs = set(training_config["default_checkpoint_epochs"])

    train_losses: list[float] = []
    test_losses: list[float] = []
    saved_epochs: list[int] = []

    print(f"Training 2L MLP: p={prime}, seed={seed}, data_seed={data_seed}, d_hidden={d_hidden}")
    print(f"Variant dir: {variant.variant_dir}")
    print(f"Device: {device}")

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
        train_logits = model(train_data)
        loss = _loss(train_logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        with torch.inference_mode():
            test_logits = model(test_data)
            test_loss = _loss(test_logits, test_labels)
            test_losses.append(test_loss.item())

        if epoch in checkpoint_epochs:
            checkpoint_path = variant.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"
            save_file(model.state_dict(), str(checkpoint_path))
            saved_epochs.append(epoch)

        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d} — train {loss.item():.4f}, test {test_loss.item():.4f}")

    # Save final checkpoint
    final_epoch = num_epochs - 1
    if final_epoch not in checkpoint_epochs:
        checkpoint_path = variant.checkpoints_dir / f"checkpoint_epoch_{final_epoch:05d}.safetensors"
        save_file(model.state_dict(), str(checkpoint_path))
        saved_epochs.append(final_epoch)

    _save_config(variant.variant_dir, params, d_hidden, training_fraction)

    metadata = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "checkpoint_epochs": sorted(saved_epochs),
        "num_epochs": num_epochs,
    }
    with open(variant.variant_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    final_train = train_losses[-1]
    final_test = test_losses[-1]
    print(f"\nDone. Final train loss: {final_train:.4f}, final test loss: {final_test:.4f}")
    print(f"Checkpoints saved: {len(saved_epochs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2-layer MLP on modular addition")
    parser.add_argument("--prime", type=int, required=True, help="Modulus for addition task")
    parser.add_argument("--seed", type=int, default=999, help="Model initialization seed (default: 999)")
    parser.add_argument("--data-seed", type=int, default=598, help="Train/test split seed (default: 598)")
    parser.add_argument("--d-hidden", type=int, default=512, help="Hidden layer width (default: 512)")
    parser.add_argument("--training-fraction", type=float, default=0.3, help="Fraction for training (default: 0.3)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda, default: auto)")
    args = parser.parse_args()

    train(
        prime=args.prime,
        seed=args.seed,
        data_seed=args.data_seed,
        d_hidden=args.d_hidden,
        training_fraction=args.training_fraction,
        device=args.device,
    )


if __name__ == "__main__":
    main()
