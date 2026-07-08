import json
import os

import einops
import numpy as np
import torch
import tqdm.auto as tqdm
from safetensors.torch import load_file, save_file
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Default checkpoint schedule optimized for grokking analysis
# Dense checkpoints during grokking phase (5000-6000), sparse elsewhere
DEFAULT_CHECKPOINT_EPOCHS = sorted(
    list(
        set(
            [
                *range(0, 1500, 100),  # Early training - sparse (~15 checkpoints)
                *range(1500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 13000, 100),  # Grokking region - dense (~100 checkpoints)
                *range(13000, 25000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)


class ModuloAdditionSpecification:
    """Baseline modulo addition model"""

    def __init__(self, model_dir, prime, device, seed=999, data_seed=598, training_fraction=0.3):
        self.model = None
        self.name = "modulo_addition"
        self.model_dir = model_dir
        self.prime = prime
        self.device = device
        self.seed = seed
        self.data_seed = data_seed
        self.training_fraction = training_fraction
        self.train_losses = []
        self.test_losses = []

        # Directory structure for REQ_002
        # results/{model_name}/{model_name}_p{prime}_seed{seed}/
        self.run_name = f"{self.name}_p{prime}_seed{seed}"
        self.full_dir = os.path.join(model_dir, self.name, self.run_name)
        self.checkpoints_dir = os.path.join(self.full_dir, "checkpoints")
        self.artifacts_dir = os.path.join(self.full_dir, "artifacts")

        # File paths
        self.model_path = os.path.join(self.full_dir, f"{self.run_name}.safetensors")
        self.metadata_path = os.path.join(self.full_dir, "metadata.json")
        self.config_path = os.path.join(self.full_dir, "config.json")

        # Legacy path for backward compatibility
        self.legacy_path = os.path.join(model_dir, self.name, f"{self.name}.pth")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def create_model(self):
        # %% model definition
        cfg = HookedTransformerConfig(
            n_layers=1,
            n_heads=4,
            d_model=128,
            d_head=32,
            d_mlp=512,
            act_fn="relu",
            normalization_type=None,
            d_vocab=self.prime + 1,
            d_vocab_out=self.prime,
            n_ctx=3,
            init_weights=True,
            device=self.device,
            seed=self.seed,
        )
        model = HookedTransformer(cfg)

        # Disable the biases, as we don't need them for this task and it makes things easier to interpret.
        for name, param in model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        self.model = model

        return self.model

    def load_from_file(self) -> HookedTransformer:
        """
        Load a trained model from file.

        Automatically detects format:
        - New format: safetensors + JSON metadata
        - Legacy format: pickle-based .pth file

        Returns:
            HookedTransformer: The loaded model
        """
        model = self.create_model()

        # Try new safetensors format first
        if os.path.exists(self.model_path):
            state_dict = load_file(self.model_path)
            model.load_state_dict(state_dict)

            # Load metadata
            with open(self.metadata_path) as f:
                metadata = json.load(f)

            self.checkpoint_epochs = metadata["checkpoint_epochs"]
            self.test_losses = metadata["test_losses"]
            self.train_losses = metadata["train_losses"]
            self.train_indices = torch.tensor(metadata["train_indices"])
            self.test_indices = torch.tensor(metadata["test_indices"])

        # Fall back to legacy pickle format for backward compatibility
        elif os.path.exists(self.legacy_path):
            cached_data = torch.load(self.legacy_path, weights_only=False)
            model.load_state_dict(cached_data["model"])
            self.model_checkpoints = cached_data["checkpoints"]
            self.checkpoint_epochs = cached_data["checkpoint_epochs"]
            self.test_losses = cached_data["test_losses"]
            self.train_losses = cached_data["train_losses"]
            self.train_indices = cached_data["train_indices"]
            self.test_indices = cached_data["test_indices"]
        else:
            raise FileNotFoundError(
                f"No model found. Checked:\n  - {self.model_path}\n  - {self.legacy_path}"
            )

        self.model = model
        return self.model

    def load_checkpoint(self, epoch: int) -> dict:
        """
        Load a specific checkpoint by epoch number.

        Args:
            epoch: The epoch number of the checkpoint to load

        Returns:
            dict: The model state_dict at that epoch

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = os.path.join(
            self.checkpoints_dir, f"checkpoint_epoch_{epoch:05d}.safetensors"
        )

        if os.path.exists(checkpoint_path):
            return load_file(checkpoint_path)

        # Try legacy format
        if os.path.exists(self.legacy_path):
            cached_data = torch.load(self.legacy_path, weights_only=False)
            if hasattr(self, "model_checkpoints") or "checkpoints" in cached_data:
                checkpoints = cached_data.get("checkpoints", self.model_checkpoints)
                epochs = cached_data.get("checkpoint_epochs", self.checkpoint_epochs)
                if epoch in epochs:
                    idx = epochs.index(epoch)
                    return checkpoints[idx]

        raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")

    def get_available_checkpoints(self) -> list[int]:
        """
        Get list of available checkpoint epochs.

        Returns:
            list[int]: Sorted list of epoch numbers with available checkpoints
        """
        epochs = []

        # Check new format checkpoints
        if os.path.exists(self.checkpoints_dir):
            for filename in os.listdir(self.checkpoints_dir):
                if filename.startswith("checkpoint_epoch_") and filename.endswith(".safetensors"):
                    epoch_str = filename.replace("checkpoint_epoch_", "").replace(
                        ".safetensors", ""
                    )
                    epochs.append(int(epoch_str))

        # Check legacy format
        if not epochs and os.path.exists(self.legacy_path):
            cached_data = torch.load(self.legacy_path, weights_only=False)
            epochs = cached_data.get("checkpoint_epochs", [])

        return sorted(epochs)

    def loss_function(self, logits, labels):
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()

    def get_optimizer(self, model):
        # optimizer config
        lr = 1e-3
        wd = 1.0
        betas = (0.9, 0.98)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        return optimizer

    def train(
        self, num_epochs: int = 25000, checkpoint_epochs: list[int] | None = None
    ) -> HookedTransformer:
        """
        Train the model with configurable checkpoint epochs.

        Args:
            num_epochs: Total number of training epochs (default: 25000)
            checkpoint_epochs: List of epoch numbers at which to save checkpoints.
                             If None, uses DEFAULT_CHECKPOINT_EPOCHS.
                             Epochs >= num_epochs are automatically filtered out.
        """
        train_data, train_labels, test_data, test_labels, train_indices, test_indices = (
            self.generate_training_data()
        )

        # Use default checkpoint schedule if none provided (REQ_001)
        if checkpoint_epochs is None:
            checkpoint_epochs = DEFAULT_CHECKPOINT_EPOCHS

        # Filter and sort checkpoint epochs, convert to set for O(1) lookup
        checkpoint_epochs = sorted([e for e in checkpoint_epochs if e < num_epochs])
        checkpoint_epochs_set = set(checkpoint_epochs)

        train_losses = []
        test_losses = []
        saved_checkpoint_epochs = []

        model = self.create_model()
        optimizer = self.get_optimizer(model)

        for epoch in tqdm.tqdm(range(num_epochs)):
            train_logits = model(train_data)
            train_loss = self.loss_function(train_logits, train_labels)
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.inference_mode():
                test_logits = model(test_data)
                test_loss = self.loss_function(test_logits, test_labels)
                test_losses.append(test_loss.item())

            # Save checkpoint immediately to disk if this epoch is in the checkpoint list (REQ_001 + REQ_002)
            if epoch in checkpoint_epochs_set:
                self._save_checkpoint(model.state_dict(), epoch)
                saved_checkpoint_epochs.append(epoch)
                print(
                    f"Epoch {epoch} Train Loss {train_loss.item():.6f} Test Loss {test_loss.item():.6f}"
                )

        # Save final model as safetensors (REQ_002)
        save_file(model.state_dict(), self.model_path)

        # Save model config as JSON (REQ_002)
        self._save_config(model.cfg)

        # Save training metadata as JSON (REQ_002)
        self._save_metadata(
            train_losses=train_losses,
            test_losses=test_losses,
            train_indices=train_indices.tolist(),
            test_indices=test_indices.tolist(),
            checkpoint_epochs=saved_checkpoint_epochs,
            num_epochs=num_epochs,
        )

        self.test_losses = test_losses
        self.train_losses = train_losses
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.checkpoint_epochs = saved_checkpoint_epochs

        self.model = model

        return self.model

    def _save_checkpoint(self, state_dict: dict, epoch: int) -> None:
        """Save a single checkpoint to disk as safetensors."""
        checkpoint_path = os.path.join(
            self.checkpoints_dir, f"checkpoint_epoch_{epoch:05d}.safetensors"
        )
        save_file(state_dict, checkpoint_path)

    def _save_config(self, cfg: HookedTransformerConfig) -> None:
        """Save model configuration as JSON."""
        config_dict = {
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "d_head": cfg.d_head,
            "d_mlp": cfg.d_mlp,
            "act_fn": cfg.act_fn,
            "normalization_type": cfg.normalization_type,
            "d_vocab": cfg.d_vocab,
            "d_vocab_out": cfg.d_vocab_out,
            "n_ctx": cfg.n_ctx,
            "seed": cfg.seed,
            "prime": self.prime,
            "model_seed": self.seed,
            "data_seed": self.data_seed,
            "training_fraction": self.training_fraction,
        }
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_metadata(self, **kwargs) -> None:
        """Save training metadata as JSON."""
        with open(self.metadata_path, "w") as f:
            json.dump(kwargs, f)

    def generate_training_data(self):
        a_vector = einops.repeat(torch.arange(self.prime), "i -> (i j)", j=self.prime)
        b_vector = einops.repeat(torch.arange(self.prime), "j -> (i j)", i=self.prime)
        equals_vector = einops.repeat(
            torch.tensor(self.prime), " -> (i j)", i=self.prime, j=self.prime
        )

        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(self.device)
        self.dataset = dataset
        labels = (dataset[:, 0] + dataset[:, 1]) % self.prime

        torch.manual_seed(self.data_seed)
        indices = torch.randperm(self.prime * self.prime)
        cutoff = int(self.prime * self.prime * self.training_fraction)
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]

        train_data = dataset[train_indices]
        train_labels = labels[train_indices]
        test_data = dataset[test_indices]
        test_labels = labels[test_indices]

        return train_data, train_labels, test_data, test_labels, train_indices, test_indices

    def compute_probe_activations(self, probe):
        assert self.model is not None, "Model must be trained or loaded first"
        probe_logits, probe_cache = self.model.run_with_cache(probe)
        return probe_logits, probe_cache

    def compute_uniform_loss(self):
        return np.log(self.prime)
