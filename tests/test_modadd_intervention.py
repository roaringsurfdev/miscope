"""Tests for REQ_067: Intervention Model Family Spec (POC).

Covers:
- Family registration and discoverability
- Storage isolation from baseline results
- Variant identity: naming, intervention ID determinism
- Config schema includes all required fields
- No-op intervention variant construction
- training_hook parameter in Variant.train()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miscope.families.implementations.modadd_intervention import (
    ModAddInterventionFamily,
    compute_intervention_id,
)
from miscope.families.registry import FamilyRegistry
from miscope.families.variant import Variant

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_FAMILIES_DIR = Path(__file__).parent.parent / "model_families"
RESULTS_DIR = Path(__file__).parent.parent / "results"

NOOP_INTERVENTION = {
    "type": "frequency_gain",
    "target_heads": "all",
    "target_frequencies": [],
    "gain": {},
    "epoch_start": 1500,
    "epoch_end": 6500,
    "ramp_epochs": 200,
}

COMPETITOR_DAMPENING = {
    "type": "frequency_gain",
    "target_heads": "all",
    "target_frequencies": [4, 10, 12, 17, 22],
    "gain": {4: 0.3, 10: 0.3, 12: 0.3, 17: 0.3, 22: 0.3},
    "epoch_start": 1500,
    "epoch_end": 6500,
    "ramp_epochs": 200,
}


@pytest.fixture
def registry() -> FamilyRegistry:
    return FamilyRegistry(MODEL_FAMILIES_DIR, RESULTS_DIR)


@pytest.fixture
def family(registry: FamilyRegistry) -> ModAddInterventionFamily:
    f = registry.get_family("modadd_intervention")
    assert isinstance(f, ModAddInterventionFamily)
    return f


@pytest.fixture
def noop_variant(family: ModAddInterventionFamily) -> Variant:
    return family.create_intervention_variant(
        prime=59,
        seed=485,
        data_seed=598,
        intervention_config=NOOP_INTERVENTION,
        results_dir=RESULTS_DIR,
    )


# ---------------------------------------------------------------------------
# Family registration
# ---------------------------------------------------------------------------


def test_family_is_registered(registry: FamilyRegistry) -> None:
    assert "modadd_intervention" in registry
    assert "modadd_intervention" in registry.get_family_names()


def test_family_is_correct_type(registry: FamilyRegistry) -> None:
    family = registry.get_family("modadd_intervention")
    assert isinstance(family, ModAddInterventionFamily)


def test_family_is_distinct_from_baseline(registry: FamilyRegistry) -> None:
    baseline = registry.get_family("modulo_addition_1layer")
    intervention = registry.get_family("modadd_intervention")
    assert type(baseline) is not type(intervention)
    assert baseline.name != intervention.name


# ---------------------------------------------------------------------------
# Storage isolation
# ---------------------------------------------------------------------------


def test_results_dir_is_isolated(noop_variant: Variant) -> None:
    baseline_root = RESULTS_DIR / "modulo_addition_1layer"
    intervention_root = RESULTS_DIR / "modadd_intervention"
    variant_dir = noop_variant.variant_dir
    assert not str(variant_dir).startswith(str(baseline_root)), (
        f"Intervention variant dir {variant_dir} is inside baseline results root"
    )
    assert str(variant_dir).startswith(str(intervention_root)), (
        f"Intervention variant dir {variant_dir} is not under modadd_intervention/"
    )


def test_variant_dir_does_not_overlap_baseline(noop_variant: Variant) -> None:
    # The family name in the results path must differ from baseline
    assert "modulo_addition_1layer" not in str(noop_variant.variant_dir)
    assert "modadd_intervention" in str(noop_variant.variant_dir)


# ---------------------------------------------------------------------------
# Variant identity
# ---------------------------------------------------------------------------


def test_variant_name_includes_all_domain_params(noop_variant: Variant) -> None:
    name = noop_variant.name
    assert "p59" in name
    assert "seed485" in name
    assert "dseed598" in name
    assert "_iv_" in name


def test_intervention_id_is_in_variant_name(family: ModAddInterventionFamily) -> None:
    expected_id = compute_intervention_id(NOOP_INTERVENTION)
    variant = family.create_intervention_variant(
        prime=59, seed=485, data_seed=598,
        intervention_config=NOOP_INTERVENTION,
        results_dir=RESULTS_DIR,
    )
    assert expected_id in variant.name


def test_intervention_id_is_deterministic() -> None:
    id1 = compute_intervention_id(NOOP_INTERVENTION)
    id2 = compute_intervention_id(NOOP_INTERVENTION)
    assert id1 == id2
    assert len(id1) == 8


def test_different_configs_produce_different_ids() -> None:
    id_noop = compute_intervention_id(NOOP_INTERVENTION)
    id_damp = compute_intervention_id(COMPETITOR_DAMPENING)
    assert id_noop != id_damp


def test_id_is_order_independent() -> None:
    config_a = {"type": "frequency_gain", "epoch_start": 1500, "gain": {}}
    config_b = {"gain": {}, "epoch_start": 1500, "type": "frequency_gain"}
    assert compute_intervention_id(config_a) == compute_intervention_id(config_b)


def test_two_variants_with_same_config_have_same_name(
    family: ModAddInterventionFamily,
) -> None:
    v1 = family.create_intervention_variant(
        prime=59, seed=485, data_seed=598,
        intervention_config=NOOP_INTERVENTION,
        results_dir=RESULTS_DIR,
    )
    v2 = family.create_intervention_variant(
        prime=59, seed=485, data_seed=598,
        intervention_config=NOOP_INTERVENTION,
        results_dir=RESULTS_DIR,
    )
    assert v1.name == v2.name


def test_different_intervention_configs_produce_different_names(
    family: ModAddInterventionFamily,
) -> None:
    v_noop = family.create_intervention_variant(
        prime=59, seed=485, data_seed=598,
        intervention_config=NOOP_INTERVENTION,
        results_dir=RESULTS_DIR,
    )
    v_damp = family.create_intervention_variant(
        prime=59, seed=485, data_seed=598,
        intervention_config=COMPETITOR_DAMPENING,
        results_dir=RESULTS_DIR,
    )
    assert v_noop.name != v_damp.name


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_variant_params_include_intervention_block(noop_variant: Variant) -> None:
    params = noop_variant.params
    assert "prime" in params
    assert "seed" in params
    assert "data_seed" in params
    assert "intervention_id" in params
    assert "intervention" in params


def test_intervention_block_has_required_fields(noop_variant: Variant) -> None:
    spec: dict[str, Any] = noop_variant.params["intervention"]
    for field in ("type", "target_heads", "target_frequencies", "gain",
                  "epoch_start", "epoch_end", "ramp_epochs"):
        assert field in spec, f"Missing required field '{field}' in intervention spec"


# ---------------------------------------------------------------------------
# training_hook parameter
# ---------------------------------------------------------------------------


def test_train_accepts_training_hook_none() -> None:
    """Variant.train() signature accepts training_hook=None without error."""
    import inspect
    sig = inspect.signature(Variant.train)
    assert "training_hook" in sig.parameters
    param = sig.parameters["training_hook"]
    assert param.default is None


def test_train_uses_run_with_hooks_when_hook_returns_hooks(
    noop_variant: Variant,
) -> None:
    """When training_hook returns non-empty list, run_with_hooks is called."""
    hook_called_epochs: list[int] = []

    def fake_hook(epoch: int) -> list[tuple[str, Any]]:
        hook_called_epochs.append(epoch)
        if epoch == 0:
            return [("hook.0.attn_out", lambda value, hook: value)]
        return []

    mock_model = MagicMock()
    mock_model.run_with_hooks.return_value = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.named_parameters.return_value = []
    mock_model.parameters.return_value = iter([])
    mock_model.cfg.n_layers = 1
    mock_model.cfg.n_heads = 4
    mock_model.cfg.d_model = 128
    mock_model.cfg.d_head = 32
    mock_model.cfg.d_mlp = 512
    mock_model.cfg.act_fn = "relu"
    mock_model.cfg.normalization_type = None
    mock_model.cfg.d_vocab = 60
    mock_model.cfg.d_vocab_out = 59
    mock_model.cfg.n_ctx = 3
    mock_model.cfg.seed = 485

    import torch

    dummy_data = torch.zeros(3, 3, dtype=torch.long)
    dummy_labels = torch.zeros(3, dtype=torch.long)

    def make_logits() -> torch.Tensor:
        return torch.zeros(3, 59, requires_grad=True)

    mock_model.return_value = make_logits()
    mock_model.side_effect = lambda *a, **kw: make_logits()
    mock_model.run_with_hooks.side_effect = lambda *a, **kw: make_logits()

    with (
        patch.object(noop_variant._family, "create_model", return_value=mock_model),
        patch.object(
            noop_variant._family,
            "generate_training_dataset",
            return_value=(dummy_data, dummy_labels, dummy_data, dummy_labels,
                          torch.arange(3), torch.arange(3)),
        ),
        patch.object(noop_variant, "_save_checkpoint"),
        patch.object(noop_variant, "_save_config"),
        patch.object(noop_variant, "_save_metadata"),
        patch.object(noop_variant, "ensure_directories"),
        patch("torch.optim.AdamW"),
    ):
        noop_variant.train(num_epochs=3, training_hook=fake_hook)

    # Hook should have been called once per epoch
    assert len(hook_called_epochs) == 3
    # run_with_hooks used only for epoch 0 (only epoch with non-empty hooks)
    assert mock_model.run_with_hooks.call_count == 1
    # model() called at least for epochs 1 and 2 training passes (test eval adds more)
    assert mock_model.call_count >= 2
