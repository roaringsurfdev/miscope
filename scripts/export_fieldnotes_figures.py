"""Export Plotly figures for fieldnotes posts.

Usage:
    python scripts/export_fieldnotes_figures.py

Outputs HTML files to fieldnotes/public/figures/.
All figures use Plotly CDN and responsive mode so they resize to fill
their iframe container.

Figures exported:
  variants-and-variables.mdx:
    loss_p113_s999_ds598.html
    loss_p113_s999_ds999.html

  reproducibility.mdx:
    multistream_p113_s999_ds598.html
    multistream_p109_s485_ds598.html
    multistream_p101_s999_ds598.html
    pca_3d_p113_s999_ds598.html
    pca_3d_p109_s485_ds598.html
    pca_3d_p101_s999_ds598.html
"""

from pathlib import Path

from miscope.config import get_config
from miscope.families import FamilyRegistry, Variant
from miscope.visualization.renderers.loss_curves import render_loss_curves_with_indicator

FIGURES_DIR = Path(__file__).parent.parent / "fieldnotes" / "public" / "figures"


def export_html(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Clear fixed width so the figure fills the iframe container via responsive mode.
    # Without this, the figure renders at its layout width (e.g. 900px) regardless
    # of the iframe size, leaving dead space on the right and causing scrollbars.
    fig.update_layout(width=None)
    path = FIGURES_DIR / f"{name}.html"
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        config={"responsive": True},
        # Suppress iframe scrollbars — the figure already fills the viewport.
        post_script="document.body.style.overflow='hidden';",
    )
    print(f"  {path.name}")


def find_variant(variants: list[Variant], prime: int, seed: int, data_seed: int) -> Variant:
    for v in variants:
        p = v.params
        if p["prime"] == prime and p["seed"] == seed and p["data_seed"] == data_seed:
            return v
    raise KeyError(f"Variant not found: p={prime} seed={seed} data_seed={data_seed}")


def export_loss_curve(variant: Variant, label: str) -> None:
    meta = variant.metadata
    n = len(meta["train_losses"])
    p = variant.params
    fig = render_loss_curves_with_indicator(
        meta["train_losses"],
        meta["test_losses"],
        current_epoch=n,  # out of range — suppresses cursor for clean blog figure
        title=f"Training Curves — p={p['prime']}, seed={p['seed']}, data_seed={p['data_seed']}",
    )
    fig.update_layout(height=350)
    export_html(fig, f"loss_{label}")


def export_multi_stream(variant: Variant, label: str) -> None:
    p = variant.params
    fig = (
        variant.at(None)
        .view("multi_stream_specialization")
        .figure(
            title=f"Multi-Stream Specialization — p={p['prime']}, seed={p['seed']}, data_seed={p['data_seed']}",
            width=900,
            height=1350,
        )
    )
    export_html(fig, f"multistream_{label}")


def export_pca_3d(variant: Variant, label: str) -> None:
    p = variant.params
    fig = variant.at(None).view("parameters.pca.scatter_3d").figure()
    fig.update_layout(
        title=f"Parameter Trajectory (PCA 3D) — p={p['prime']}, seed={p['seed']}, data_seed={p['data_seed']}",
        width=750,
        height=620,
    )
    export_html(fig, f"pca_3d_{label}")


def main() -> None:
    cfg = get_config()
    registry = FamilyRegistry(cfg.model_families_dir, cfg.results_dir)
    family = registry.get_family("modulo_addition_1layer")
    all_variants = registry.get_variants(family)

    # --- variants-and-variables.mdx: loss curves ---
    print("Loss curves:")
    export_loss_curve(find_variant(all_variants, 113, 999, 598), "p113_s999_ds598")
    export_loss_curve(find_variant(all_variants, 113, 999, 999), "p113_s999_ds999")

    # --- reproducibility.mdx: multi-stream and PCA 3D ---
    reproduct_variants = [
        (find_variant(all_variants, 113, 999, 598), "p113_s999_ds598"),
        (find_variant(all_variants, 109, 485, 598), "p109_s485_ds598"),
        (find_variant(all_variants, 101, 999, 598), "p101_s999_ds598"),
    ]

    print("Multi-stream specialization:")
    for v, label in reproduct_variants:
        export_multi_stream(v, label)

    print("PCA 3D scatter:")
    for v, label in reproduct_variants:
        export_pca_3d(v, label)

    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
