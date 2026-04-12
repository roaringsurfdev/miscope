"""Train a learned-embedding MLP on modular addition.

Standalone training script for ModuloAdditionEmbedMLPFamily variants.
Produces the same checkpoint format (safetensors) and directory structure
as the other MLP families.

Usage:
    python scripts/train_learned_emb_mlp.py --prime 113 --seed 999 --data-seed 598
    python scripts/train_learned_emb_mlp.py --prime 113 --seed 999 --data-seed 598 --d-embed 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import tqdm.auto as tqdm
from safetensors.torch import save_file

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from miscope.config import get_config  # noqa: E402
from miscope.families.implementations.modulo_addition_embed_mlp import (  # noqa: E402
    load_modulo_addition_embed_mlp_family,  # noqa: E402
)
from miscope.families.variant import Variant  # noqa: E402


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = logits.to(torch.float64).log_softmax(dim=-1)
    return -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean()


def train(
    prime: int,
    seed: int,
    data_seed: int,
    d_embed: int = 16,
    d_hidden: int = 512,
    training_fraction: float = 0.75,
    device: str | None = None,
) -> None:
    """Train a learned-embedding MLP variant and save checkpoints.

    Args:
        prime: Modulus for the addition task
        seed: Random seed for model initialization
        data_seed: Random seed for train/test split
        d_embed: Embedding dimension (default 16, per viability certificate analysis)
        d_hidden: Hidden layer width (default 512)
        training_fraction: Fraction of data for training (default 0.75)
        device: Device string ('cpu', 'cuda', or None for auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_config()
    family = load_modulo_addition_embed_mlp_family(cfg.model_families_dir)

    family._config["architecture"]["d_embed"] = d_embed
    family._config["architecture"]["d_hidden"] = d_hidden

    params = {"prime": prime, "seed": seed, "data_seed": data_seed}
    variant = Variant(family, params, cfg.results_dir)  # type: ignore
    variant.ensure_directories()

    model = family.create_model(params, device=device)
    train_data, train_labels, test_data, test_labels, train_indices, test_indices = (
        family.generate_training_dataset(
            params, training_fraction=training_fraction, data_seed=data_seed, device=device
        )
    )
    train_a, train_b = train_data.unbind(1)
    test_a, test_b = test_data.unbind(1)

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

    print(
        f"Training Learned-Emb MLP: p={prime}, seed={seed}, data_seed={data_seed}, "
        f"d_embed={d_embed}, d_hidden={d_hidden}"
    )
    print(f"Variant dir: {variant.variant_dir}")
    print(f"Device: {device}")

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
        train_logits = model(train_a, train_b)
        loss = _loss(train_logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        with torch.inference_mode():
            test_logits = model(test_a, test_b)
            test_loss = _loss(test_logits, test_labels)
            test_losses.append(test_loss.item())

        if epoch in checkpoint_epochs:
            checkpoint_path = variant.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"
            save_file(model.state_dict(), str(checkpoint_path))
            saved_epochs.append(epoch)

        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d} — train {loss.item():.4f}, test {test_loss.item():.4f}")

    final_epoch = num_epochs - 1
    if final_epoch not in checkpoint_epochs:
        checkpoint_path = (
            variant.checkpoints_dir / f"checkpoint_epoch_{final_epoch:05d}.safetensors"
        )
        save_file(model.state_dict(), str(checkpoint_path))
        saved_epochs.append(final_epoch)

    config = {
        "architecture": "learned_emb_mlp",
        "vocab_size": prime,
        "d_embed": d_embed,
        "d_hidden": d_hidden,
        "act_fn": "relu",
        **params,
        "training_fraction": training_fraction,
    }
    with open(variant.variant_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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

    print(f"\nDone. Final train loss: {train_losses[-1]:.6f}, test loss: {test_losses[-1]:.6f}")
    print(f"Checkpoints saved: {len(saved_epochs)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a learned-embedding MLP on modular addition"
    )
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--data-seed", type=int, default=598)
    parser.add_argument("--d-embed", type=int, default=16, help="Embedding dimension (default: 16)")
    parser.add_argument("--d-hidden", type=int, default=512)
    parser.add_argument("--training-fraction", type=float, default=0.75)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        prime=args.prime,
        seed=args.seed,
        data_seed=args.data_seed,
        d_embed=args.d_embed,
        d_hidden=args.d_hidden,
        training_fraction=args.training_fraction,
        device=args.device,
    )


if __name__ == "__main__":
    main()
