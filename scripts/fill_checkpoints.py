"""
Fill in dense checkpoints for a variant around a target epoch window.

Resumes from an existing checkpoint (fresh optimizer state, deterministic
data split) and saves at 100-epoch intervals through the window. Existing
checkpoints are skipped. Variant metadata is not modified.

After running this script, re-run the analysis pipeline on the variant
with force=False — only the new epochs will be processed.

Usage:
    python scripts/fill_checkpoints.py
    (Edit the VARIANT_PARAMS and WINDOW constants below before running)

Optimizer note: Adam momentum is not saved, so the first ~100-200 epochs
after resume may differ slightly from the original trajectory. Resume
from 200+ epochs before the window of interest to allow momentum to settle.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from safetensors.torch import load_file, save_file

from miscope import load_family

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

FAMILY_NAME = "modulo_addition_1layer"

VARIANT_PARAMS = {"prime": 103, "seed": 999, "data_seed": 598}

# Resume from this existing checkpoint (must already exist on disk)
RESUME_FROM_EPOCH = 4500

# Fill 100-epoch intervals through this window
FILL_WINDOW_START = RESUME_FROM_EPOCH + 1  # exclusive of resume point
FILL_WINDOW_END = 9000

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load family and variant
# ---------------------------------------------------------------------------

loaded_family = load_family(FAMILY_NAME)
family = loaded_family.family  # underlying ModelFamily
variant = loaded_family.get_variant(**VARIANT_PARAMS)

print(f"Variant: {variant.name}")
print(f"Resume from: epoch {RESUME_FROM_EPOCH}")
print(f"Fill window: {FILL_WINDOW_START} – {FILL_WINDOW_END} (100-epoch intervals)")

# ---------------------------------------------------------------------------
# Determine which checkpoint epochs are missing
# ---------------------------------------------------------------------------

existing_epochs = set(variant.get_available_checkpoints())
target_epochs = set(range(FILL_WINDOW_START, FILL_WINDOW_END + 1, 100))
new_epochs = sorted(target_epochs - existing_epochs)

print(f"Existing checkpoints in window: {len(target_epochs) - len(new_epochs)}")
print(f"New checkpoints to generate: {len(new_epochs)}")
if new_epochs:
    print(f"  First: {new_epochs[0]}, Last: {new_epochs[-1]}")
else:
    print("Nothing to do — all target epochs already exist.")
    sys.exit(0)

new_epoch_set = set(new_epochs)

# ---------------------------------------------------------------------------
# Load model from resume checkpoint
# ---------------------------------------------------------------------------

resume_path = variant.checkpoints_dir / f"checkpoint_epoch_{RESUME_FROM_EPOCH:05d}.safetensors"
if not resume_path.exists():
    raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

model = family.create_model(VARIANT_PARAMS, device=DEVICE)
state_dict = load_file(str(resume_path))
state_dict = {k: v.to(DEVICE) for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.train()

print(f"Loaded model from {resume_path.name}")

# ---------------------------------------------------------------------------
# Re-create training data (same data_seed → identical split)
# ---------------------------------------------------------------------------

(train_data, train_labels,
 test_data, test_labels,
 _train_idx, _test_idx) = family.generate_training_dataset(
    VARIANT_PARAMS,
    training_fraction=0.3,
    data_seed=VARIANT_PARAMS["data_seed"],
    device=DEVICE,
)

# ---------------------------------------------------------------------------
# Fresh optimizer (no saved momentum state)
# ---------------------------------------------------------------------------

optimizer = family.create_optimizer(model)

# ---------------------------------------------------------------------------
# Training loop — only save at new target epochs
# ---------------------------------------------------------------------------

saved = []
total_epochs = FILL_WINDOW_END - RESUME_FROM_EPOCH

for step, epoch in enumerate(range(RESUME_FROM_EPOCH + 1, FILL_WINDOW_END + 1)):
    # Standard forward/backward
    train_logits = model(train_data)
    train_loss = family.compute_loss(train_logits, train_labels)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch in new_epoch_set:
        ckpt_path = variant.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"
        save_file(model.state_dict(), str(ckpt_path))
        saved.append(epoch)

    if step % 500 == 0:
        print(f"  epoch {epoch}/{FILL_WINDOW_END}  train_loss={train_loss.item():.6f}")

print(f"\nDone. Saved {len(saved)} new checkpoints: {saved[:5]}...{saved[-5:] if len(saved) > 5 else ''}")
print()
print("Next step: re-run the analysis pipeline on this variant with force=False.")
print("  Example:")
print("    from miscope import load_family")
print("    from miscope.analysis import AnalysisPipeline")
print("    from miscope.analysis.analyzers import RepresentationalGeometryAnalyzer")
print("    family = load_family('modulo_addition_1layer')")
print("    variant = family.get_variant(prime=103, seed=999, data_seed=598)")
print("    pipeline = AnalysisPipeline(variant)")
print("    pipeline.register(RepresentationalGeometryAnalyzer())")
print("    pipeline.run(force=False)")
