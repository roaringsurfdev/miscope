"""Tests for ModuloAdditionEmbedMLPFamily and the underlying HookedEmbeddingMLP.

Migrated under REQ_113 / REQ_114: standalone ``ModuloAdditionEmbedMLP``
and per-family ``*ActivationBundle`` classes are retired in favor of
``HookedEmbeddingMLP`` (canonical-name surface). Bundles and
``family.run_forward_pass`` are gone — analyzers consume
``HookedModel.run_with_cache`` directly.
"""

from __future__ import annotations

import pytest
import torch

from miscope.architectures import (
    ActivationCache,
    HookedEmbeddingMLP,
    HookedEmbeddingMLPConfig,
    HookedModel,
)
from miscope.families.implementations.modulo_addition_embed_mlp import (
    ModuloAdditionEmbedMLPFamily,
    load_modulo_addition_embed_mlp_family,
)

P = 13
D_EMBED = 8
D_HIDDEN = 32


@pytest.fixture
def model() -> HookedEmbeddingMLP:
    return HookedEmbeddingMLP(
        HookedEmbeddingMLPConfig(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=42)
    )


@pytest.fixture
def probe() -> torch.Tensor:
    """(N, 2) long tensor of (a, b) index pairs for all P² inputs."""
    a_vals = torch.arange(P).repeat_interleave(P)
    b_vals = torch.arange(P).repeat(P)
    return torch.stack([a_vals, b_vals], dim=1)


@pytest.fixture
def family() -> ModuloAdditionEmbedMLPFamily:
    return load_modulo_addition_embed_mlp_family("model_families")


@pytest.fixture
def params() -> dict:
    return {"prime": P, "seed": 42, "data_seed": 598}


# ── HookedEmbeddingMLP ───────────────────────────────────────────────────────


class TestHookedEmbeddingMLP:
    def test_is_hooked_model(self, model):
        assert isinstance(model, HookedModel)

    def test_forward_output_shape(self, model, probe):
        with torch.inference_mode():
            out = model(probe)
        assert out.shape == (P * P, P)

    def test_seedable_initialization(self):
        m1 = HookedEmbeddingMLP(
            HookedEmbeddingMLPConfig(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=7)
        )
        m2 = HookedEmbeddingMLP(
            HookedEmbeddingMLPConfig(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=7)
        )
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = HookedEmbeddingMLP(
            HookedEmbeddingMLPConfig(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=1)
        )
        m2 = HookedEmbeddingMLP(
            HookedEmbeddingMLPConfig(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=2)
        )
        differs = any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
        assert differs

    def test_embedding_shapes(self, model):
        assert model.embed_a.weight.shape == (P, D_EMBED)
        assert model.embed_b.weight.shape == (P, D_EMBED)

    def test_weight_shapes(self, model):
        assert model.W_in.weight.shape == (D_HIDDEN, D_EMBED)
        assert model.W_out.weight.shape == (P, D_HIDDEN)

    def test_no_bias(self, model):
        assert model.W_in.bias is None
        assert model.W_out.bias is None

    def test_sum_embedding_symmetry(self, model):
        """embed_a(3) + embed_b(5) should differ from embed_a(5) + embed_b(3)."""
        a = torch.tensor([3])
        b = torch.tensor([5])
        with torch.inference_mode():
            sum_ab = model.embed_a(a) + model.embed_b(b)
            sum_ba = model.embed_a(b) + model.embed_b(a)
        assert not torch.equal(sum_ab, sum_ba)

    def test_canonical_weight_access(self, model):
        assert torch.equal(model.get_weight("embed.embed_a"), model.embed_a.weight)
        assert torch.equal(model.get_weight("embed.embed_b"), model.embed_b.weight)
        assert torch.equal(model.get_weight("blocks.0.mlp.in.W"), model.W_in.weight)
        assert torch.equal(model.get_weight("blocks.0.mlp.out.W"), model.W_out.weight)

    def test_embedding_identity_invariant_W_E_raises(self, model):
        """embed.W_E must raise KeyError — load-bearing for architecture identity.

        An analyzer expecting a transformer's shared embedding matrix must
        fail loudly here rather than be silently aliased to one of the
        per-input matrices.
        """
        with pytest.raises(KeyError):
            model.get_weight("embed.W_E")

    def test_run_with_cache_publishes_embed_hook(self, model, probe):
        """embed.hook_out captures the post-sum embed_a(a) + embed_b(b)."""
        logits, cache = model.run_with_cache(probe)
        assert isinstance(cache, ActivationCache)
        assert cache["embed.hook_out"].shape == (P * P, D_EMBED)

        # Validate post-sum semantics: cache value equals embed_a(a) + embed_b(b)
        a, b = probe.unbind(1)
        with torch.inference_mode():
            expected = model.embed_a(a) + model.embed_b(b)
        assert torch.equal(cache["embed.hook_out"], expected)

    def test_run_with_cache_canonical_hooks_present(self, model, probe):
        _, cache = model.run_with_cache(probe)
        for canonical in (
            "embed.hook_out",
            "blocks.0.hook_in",
            "blocks.0.mlp.hook_pre",
            "blocks.0.mlp.hook_out",
            "blocks.0.hook_out",
            "unembed.hook_out",
        ):
            assert canonical in cache, f"missing: {canonical}"


# ── ModuloAdditionEmbedMLPFamily ────────────────────────────────────────────


class TestModuloAdditionEmbedMLPFamily:
    def test_name(self, family):
        assert family.name == "modulo_addition_learned_emb_mlp"

    def test_create_model_returns_correct_type(self, family, params):
        model = family.create_model(params)
        assert isinstance(model, HookedEmbeddingMLP)
        assert isinstance(model, HookedModel)
        assert model.vocab_size == P

    def test_create_model_d_embed_from_architecture(self, family, params):
        model = family.create_model(params)
        assert model.d_embed == family.architecture.get("d_embed", 16)

    def test_generate_analysis_dataset_shape(self, family, params):
        probe = family.generate_analysis_dataset(params)
        assert probe.shape == (P * P, 2)
        assert probe.dtype == torch.long

    def test_generate_analysis_dataset_covers_all_pairs(self, family, params):
        probe = family.generate_analysis_dataset(params)
        pairs = set(map(tuple, probe.tolist()))
        assert len(pairs) == P * P

    def test_generate_training_dataset_returns_6_tuple(self, family, params):
        result = family.generate_training_dataset(params)
        assert len(result) == 6

    def test_generate_training_dataset_shapes(self, family, params):
        train_data, train_l, test_data, test_l, tr_idx, te_idx = family.generate_training_dataset(
            params
        )
        assert len(tr_idx) + len(te_idx) == P * P
        assert train_data.shape == (len(tr_idx), 2)
        assert test_data.shape == (len(te_idx), 2)
        assert train_l.shape == (len(tr_idx),)
        assert test_l.shape == (len(te_idx),)

    def test_generate_training_dataset_reproducible(self, family, params):
        r1 = family.generate_training_dataset(params)
        r2 = family.generate_training_dataset(params)
        assert torch.equal(r1[4], r2[4])  # train_indices

    def test_create_model_run_with_cache(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        with torch.inference_mode():
            logits, cache = model.run_with_cache(probe)
        d_hidden = family.architecture.get("d_hidden", 512)
        assert logits.shape == (P * P, P)
        assert cache["blocks.0.mlp.hook_out"].shape == (P * P, d_hidden)

    def test_prepare_analysis_context_keys(self, family, params):
        ctx = family.prepare_analysis_context(params, "cpu")
        assert "fourier_basis" in ctx
        assert "params" in ctx
        assert "loss_fn" in ctx

    def test_prepare_analysis_context_fourier_basis_shape(self, family, params):
        ctx = family.prepare_analysis_context(params, "cpu")
        assert ctx["fourier_basis"].shape == (P, P)

    def test_prepare_analysis_context_loss_fn(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        ctx = family.prepare_analysis_context(params, "cpu")
        loss = ctx["loss_fn"](model, probe)
        assert isinstance(loss, float)
        assert loss > 0

    def test_make_probe_shape(self, family, params):
        probe = family.make_probe(params, [[3, 5], [0, 12]])
        assert probe.shape == (2, 2)
        assert probe.dtype == torch.long
        assert probe[0, 0].item() == 3  # a
        assert probe[0, 1].item() == 5  # b

    def test_analyzers_includes_neuron_activations(self, family):
        assert "neuron_activations" in family.analyzers

    def test_analyzers_includes_parameter_snapshot(self, family):
        assert "parameter_snapshot" in family.analyzers

    def test_analyzers_excludes_transformer_specific(self, family):
        transformer_only = {
            "attention_freq",
            "attention_fourier",
            "attention_patterns",
            "fourier_nucleation",
        }
        assert not transformer_only.intersection(set(family.analyzers))

    def test_secondary_analyzers_excludes_neuron_fourier(self, family):
        # neuron_fourier assumes W_in has 2*p columns (one-hot encoding);
        # learned-emb MLP uses d_embed=16 — incompatible, excluded intentionally.
        assert "neuron_fourier" not in family.secondary_analyzers

    def test_cross_epoch_analyzers_includes_neuron_group_pca(self, family):
        assert "neuron_group_pca" in family.cross_epoch_analyzers

    def test_get_training_config_keys(self, family):
        cfg = family.get_training_config()
        for key in (
            "learning_rate",
            "weight_decay",
            "betas",
            "num_epochs",
            "default_checkpoint_epochs",
        ):
            assert key in cfg

    def test_get_training_config_he_et_al_settings(self, family):
        """Verify hyperparameters match He et al. (2602.16849)."""
        cfg = family.get_training_config()
        assert cfg["learning_rate"] == 1e-4
        assert cfg["weight_decay"] == 2.0
        assert cfg["num_epochs"] == 50000
