# Mechanistic Interpretability Data Domain Model

This document captures the data model for mechanistic interpretability research,
with a focus on learning dynamics — analysis of model checkpoints over training.

The domain model is organized into five layers:

- **Platform Objects** — infrastructure defined by the platform, not the model
- **Intrinsic Objects** — defined by the model architecture; uncontroversial
- **Performance Metrics** — well-established scalar training signals
- **Derived / Virtual Objects** — composed weight objects with mechanistic interpretability justification
- **Analysis Objects** — link domain objects to decomposition results

---

# Schema Definition Format

### Object Name
*Object description*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| field_name | data_type | analyzer_name or Key | Description | STATUS | Note |

**STATUS values:** NEW, EXISTS, DELTA, CHEAP

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Key_PK | Object | PRIMARY_KEY | field_list | Description | Note |

*Key types: PRIMARY_KEY, FOREIGN_KEY*

#### Indexes
| Index Name | Object Name | Fields | Type | Description | Note |
| --- | --- | --- | --- | --- | --- |
| IDX_Name | Object | field_list | UNIQUE or NA | Description | Note |

---

## Platform Objects
*This section is reserved for objects defined by the **platform**. They should not by controversial or require mathematical proof.* 

### Family
*List of model architectures + tasks that provide context for model variations (Variants)*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| family_id | int | Key | Surrogate key | - | - |
| name | string | - | Full family name | - | - |
| abbr | string | - | Abbreviated family name | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Family_PK | Family | PRIMARY_KEY | family_id | Surrogate key | - |

---

### Family_Parameter
*List of per-family parameters used to differentiate model Variants. Does not include model_seed or data_seed, because all Variants across all model architectures will use pre-defined seeds. This table is only to be used for parameters intended to capture different task parameters.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| family_id | int | Key | - | - | - |
| name | string | - | Parameter name | - | - |
| datatype | string | - | Parameter data type | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Family_Parameter_PK | Family_Parameter | PRIMARY_KEY | family_id, name | - | - |
| Family_Parameter_FK_Family | Family | FOREIGN_KEY | family_id | - | - |

---

### Variant
*A realized instance of a Family. Container for all checkpoints and analysis data associated with one training run.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | Surrogate key | - | - |
| family_id | int | Key | Parent family | - | - |
| name | string | - | Full variant name | - | - |
| abbr | string | - | Abbreviated variant name | - | - |
| model_seed | int | - | Seed for model weight initialization | - | - |
| data_seed | int | - | Seed for dataset creation and selection | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Variant_PK | Variant | PRIMARY_KEY | variant_id | Surrogate key | - |
| Variant_FK_Family | Family | FOREIGN_KEY | family_id | - | - |

---

### Variant_Parameter
*Resolved parameter values for a Variant instance (e.g., prime=113 for a modular
addition variant). Joins Family_Parameter definitions to their per-variant values.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| parameter_name | string | Key | Matches Family_Parameter.name | - | - |
| value | string | - | Serialized parameter value | - | Type-cast on read via Family_Parameter.datatype |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Variant_Parameter_PK | Variant_Parameter | PRIMARY_KEY | variant_id, parameter_name | - | - |
| Variant_Parameter_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

---

### Checkpoint
*A point in training time for a Variant. Anchors all weight snapshots and activation data generated at that step. The foreign key target for epoch-scoped analysis results.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| epoch | int | Key | Training step at which checkpoint was saved | - | - |
| loss_train | float | - | Training loss at this checkpoint | - | - |
| loss_test | float | - | Test loss at this checkpoint | - | - |
| safetensors_path | string | - | Path to checkpoint weights on disk | - | Relative to variant root |
| wall_clock_time | float | - | Wall clock seconds elapsed at this checkpoint | - | Optional |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Checkpoint_PK | Checkpoint | PRIMARY_KEY | variant_id, epoch | - | - |
| Checkpoint_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

---

### Probe
*Input data used in a forward pass to generate activations. A result is only meaningful relative to the probe that produced it.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| probe_id | int | Key | Surrogate key | - | - |
| family_id | int | Key | Family this probe was constructed for | - | - |
| name | string | - | Descriptive probe name | - | - |
| description | string | - | Purpose and construction method | - | - |
| n_samples | int | - | Number of input samples in the probe | - | - |
| probe_path | string | - | Path to serialized probe tensor on disk | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Probe_PK | Probe | PRIMARY_KEY | probe_id | Surrogate key | - |
| Probe_FK_Family | Family | FOREIGN_KEY | family_id | - | - |

---

### Training_Window
*A labeled region of a training trajectory. Per-Variant: two variants of the same
family may enter the same phase at different epochs. Boundary method is recorded
because window boundaries are interpretation-dependent and multiple methods
may be compared.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| window_name | string | Key | Phase label | - | See WINDOW_NAME values |
| boundary_method | string | Key | Method used to identify boundary | - | e.g., visual, loss_derivative, dmd |
| epoch_start | int | - | First epoch of the window | - | Inclusive |
| epoch_end | int | - | Last epoch of the window | - | Inclusive; NULL if terminal |
| notes | string | - | Analyst notes on this boundary | - | Optional |

**WINDOW_NAME values:** first_descent, plateau, second_descent, terminal_training

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Training_Window_PK | Training_Window | PRIMARY_KEY | variant_id, window_name, boundary_method | - | - |
| Training_Window_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

---

##  Intrinsic Objects
*This section is reserved for objects defined by the **model architecture**. They should not be controversial or require mathematical proof. Weight-bearing objects have weights; activation-point objects are snapshots in the forward pass.*


### Embedding
*Token embedding weights. Weight-bearing.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Embedding_PK | Embedding | PRIMARY_KEY | variant_id, epoch | - | - |
| Embedding_FK_Checkpoint | Checkpoint | FOREIGN_KEY | variant_id, epoch | - | - |

---

### Unembedding
*Unembedding (output projection) weights. Weight-bearing.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Unembedding_PK | Unembedding | PRIMARY_KEY | variant_id, epoch | - | - |
| Unembedding_FK_Checkpoint | Checkpoint | FOREIGN_KEY | variant_id, epoch | - | - |

---

### MLP
*MLP block at a given layer. Weight-bearing.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT(0) |
| epoch | int | Key | - | - | - |

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| MLP_PK | - | PRIMARY_KEY | variant_id, layer_index, epoch | Text | Text | Text |
| MLP_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

---

### MLP_Neuron
*A single neuron within an MLP block. Positional — corresponds to a row/column of
an MLP weight matrix. Behavior is an activation-space concept; this object captures
the structural identity.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT 0 |
| neuron_index | int | Key | Position in MLP weight matrix | - | - |
| epoch | int | Key | - | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| MLP_Neuron_PK | MLP_Neuron | PRIMARY_KEY | variant_id, layer_index, neuron_index, epoch | - | - |
| MLP_Neuron_FK_MLP | MLP | FOREIGN_KEY | variant_id, layer_index, epoch | - | - |

---

### Attention
*Attention block at a given layer and head. Weight-bearing.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT 0 |
| head_index | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Attention_PK | Attention | PRIMARY_KEY | variant_id, layer_index, head_index, epoch | - | - |
| Attention_FK_Checkpoint | Checkpoint | FOREIGN_KEY | variant_id, epoch | - | - |

---

### Residual Stream
*Activation snapshot at residual stream post-position (resid_post). An activation-point object — no dedicated weights. Meaningful only relative to the Probe that generated it.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| epoch | int | Key | - | - | - |
| probe_id | int | Key | Probe used in the forward pass | - | - |
| layer_index | int | Key | Layer at which resid_post was captured | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Residual_Stream_PK | Residual_Stream | PRIMARY_KEY | variant_id, epoch, probe_id, layer_index | - | - |
| Residual_Stream_FK_Checkpoint | Checkpoint | FOREIGN_KEY | variant_id, epoch | - | - |
| Residual_Stream_FK_Probe | Probe | FOREIGN_KEY | probe_id | - | - |

---

## Performance Metrics
*Well-established scalar training signals. Uncontroversial; closer to performance optimization than model internals.*

### Loss
*Stores Train and Test Loss values over training*
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| loss_type | int | Key | - | - | - |
| epoch | int | Key | - | - | - |
| value | float | - | - | - | - |

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Loss_PK | - | PRIMARY_KEY | variant_id, loss_type, epoch | Text | Text | Text |
| Loss_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

LOSS_TYPE: TRAIN=0, TEST=1

## Virtual Weights/Derived Objects
*Objects with mechanistic interpretability justification. Composed from intrinsic weights. Higher churn expected as the field develops — designed for extensibility.*

### QK_Circuit
*The composed virtual weight W_Q^T · W_K for one attention head. Captures the query-key interaction pattern. One circuit per (layer, head) pair.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT 0 |
| head_index | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| QK_Circuit_PK | QK_Circuit | PRIMARY_KEY | variant_id, layer_index, head_index, epoch | - | - |
| QK_Circuit_FK_Attention | Attention | FOREIGN_KEY | variant_id, layer_index, head_index, epoch | - | - |

#### Indexes
| Index Name | Object Name | Fields | Type | Description | Note |
| --- | --- | --- | --- | --- | --- |
| QK_Circuit_IDX01 | QK_Circuit | variant_id, layer_index, head_index | UNIQUE | One circuit per head per checkpoint | - |

---

### OV_Circuit
*The composed virtual weight W_O · W_V for one attention head. Captures the output-value projection pattern. One circuit per (layer, head) pair.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT 0 |
| head_index | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| OV_Circuit_PK | OV_Circuit | PRIMARY_KEY | variant_id, layer_index, head_index, epoch | - | - |
| OV_Circuit_FK_Attention | Attention | FOREIGN_KEY | variant_id, layer_index, head_index, epoch | - | - |

#### Indexes
| Index Name | Object Name | Fields | Type | Description | Note |
| --- | --- | --- | --- | --- | --- |
| OV_Circuit_IDX01 | OV_Circuit | variant_id, layer_index, head_index | UNIQUE | One circuit per head per checkpoint | - |

---
## Analysis Objects

*Links domain objects (Intrinsic or Derived) to decomposition results. Designed for
high churn: new decomposition types and new source object types are added by
introducing new values, not new tables.*

*The `object_type` discriminator makes the link table agnostic to the source object.
New theoretical objects (future circuits, composed weights) do not require schema
migrations — they introduce new `object_type` values.*

---

### Decomposition
*Link record connecting a domain object to a set of decomposition results. One record
per (object, decomposition type, recipe) combination. The join point for all
cross-variant and cross-checkpoint analysis queries.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| decomposition_id | int | Key | Surrogate key | - | - |
| variant_id | int | Key | - | - | - |
| epoch | int | Key | - | NULL for cross-epoch decompositions | - |
| object_type | string | - | Discriminator for the source domain object | - | See OBJECT_TYPE |
| object_key | string | - | Serialized composite key of the source object | - | e.g., "0:1" for layer 0, head 1 |
| decomposition_type | string | - | Analysis method applied | - | See DECOMPOSITION_TYPE |
| recipe_id | string | - | Recipe signature from parameterization | - | Empty string = default recipe |
| is_cross_epoch | bool | - | True if results span multiple checkpoints | - | - |
| epoch_start | int | - | First epoch of cross-epoch window | - | NULL for per-epoch |
| epoch_end | int | - | Last epoch of cross-epoch window | - | NULL for per-epoch |

**OBJECT_TYPE values:** mlp, mlp_neuron, attention, residual_stream, embedding,
unembedding, qk_circuit, ov_circuit

**DECOMPOSITION_TYPE values:** svd, dmd, fourier, pca, eigendecomposition

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Decomposition_PK | Decomposition | PRIMARY_KEY | decomposition_id | Surrogate key | - |
| Decomposition_FK_Checkpoint | Checkpoint | FOREIGN_KEY | variant_id, epoch | NULL for cross-epoch | - |

#### Indexes
| Index Name | Object Name | Fields | Type | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Decomposition_IDX01 | Decomposition | variant_id, object_type, object_key, decomposition_type, recipe_id | UNIQUE | One result per (object, method, recipe) per epoch | - |

---

### Decomposition_Scalar
*Columnar (queryable) results from a decomposition. Flattened to long format for
cross-variant and cross-checkpoint comparison. Each row is one named scalar metric
from one decomposition run.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| decomposition_id | int | Key | Parent decomposition record | - | - |
| field_name | string | Key | Metric name | - | e.g., explained_variance_ratio, frequency_sparsity |
| value | float | - | Scalar value | - | - |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Decomposition_Scalar_PK | Decomposition_Scalar | PRIMARY_KEY | decomposition_id, field_name | - | - |
| Decomposition_Scalar_FK | Decomposition | FOREIGN_KEY | decomposition_id | - | - |

---

### Decomposition_Tensor
*Blob references for dense array outputs from a decomposition. Each row points into
the tensor store (Zarr). Retrieved for numerical computation, not queried.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| --- | --- | --- | --- | --- | --- |
| decomposition_id | int | Key | Parent decomposition record | - | - |
| field_name | string | Key | Tensor name | - | e.g., U, S, Vt, modes, eigenvalues |
| zarr_path | string | - | Path within Zarr store | - | - |
| shape | string | - | Serialized tensor shape | - | e.g., "[512, 64]" |
| dtype | string | - | Element dtype | - | e.g., float32, complex128 |

#### Keys
| Key Name | Reference | Type | Fields | Description | Note |
| --- | --- | --- | --- | --- | --- |
| Decomposition_Tensor_PK | Decomposition_Tensor | PRIMARY_KEY | decomposition_id, field_name | - | - |
| Decomposition_Tensor_FK | Decomposition | FOREIGN_KEY | decomposition_id | - | - |

---

# Decomposition Reference

## Mode vs. Basis

A **mode** is one element of a decomposition. A **basis** is the complete set.

- A single Fourier frequency → a mode
- The complete set of Fourier frequencies → the Fourier basis

## Spectral Decomposition

Spectral decomposition reveals a *spectrum*: how much of something is present at each
component. All of the following are spectral decompositions:

- Eigendecomposition (spectrum of eigenvalues)
- PCA (spectrum of variances)
- DFT / Fourier (spectrum of frequencies)
- SVD (spectrum of singular values)
- DMD (spectrum of dynamic eigenvalues)

## Decomposition Output Taxonomy

| Decomposition | Shape component | Weight / Strength component | Behavior component |
| --- | --- | --- | --- |
| Eigendecomposition | Eigenvector | Eigenvalue | — |
| SVD | Left / right singular vectors | Singular value | — |
| PCA | Principal component (loading) | Explained variance | — |
| Fourier / DFT | Frequency (basis function) | Amplitude | Phase |
| DMD | DMD mode (spatial) | Amplitude | Eigenvalue (growth + frequency) |

## DMD

### Analysis Hierarchy

```
Koopman operator (theoretical, infinite-dimensional)
    ↓  approximated by
DMD matrix A (finite, numerical)
    ↓  decomposed into
DMD modes (vectors) + eigenvalues (scalars)
```

### Full DMD Output Inventory

| Artifact | Type | What it tells you |
| --- | --- | --- |
| A (or Ã, reduced) | Matrix | Linear operator approximating the dynamics |
| Φ (DMD modes) | Set of vectors | Spatial patterns in the data |
| λ (eigenvalues) | Set of complex scalars | Growth rate + frequency of each mode |
| ω (continuous eigenvalues) | Set of complex scalars | log(λ)/dt — more interpretable form |
| b (mode amplitudes) | Set of scalars | How much each mode contributes at t=0 |
| Φ·b (reconstruction) | Tensor | Optional: reconstructed / projected data |

### Stored Structure

```
DMDResult:
  ├── A_tilde        # reduced matrix (r × r)
  ├── modes          # matrix of column vectors (n × r)
  ├── eigenvalues    # complex vector (r,)
  ├── omega          # continuous eigenvalues, complex vector (r,)
  ├── amplitudes     # real vector (r,)
  └── metadata
        ├── dt           # timestep
        ├── r            # rank (truncation level)
        └── input_shape  # shape of original tensors
```

## Fourier Analysis Output Structure

```
FourierAnalysis:
  ├── dominant_frequencies[]     # which k values are active
  ├── amplitudes[]               # per frequency
  ├── phases[]                   # per frequency
  ├── explained_variance_ratio   # scalar — trackable over time
  ├── frequency_sparsity         # scalar — does it concentrate on a few k?
  └── reconstruction_error       # scalar
```

## Object → Decomposition Traversal Pattern

```
ModelVariant
  └── Checkpoint (epoch, loss)
        └── Decomposition (object_type, object_key, decomposition_type, recipe)
              ├── Decomposition_Scalar (field_name, value)     ← queryable
              └── Decomposition_Tensor (field_name, zarr_path) ← retrievable
```

## Group Structure Note

For tasks with algebraic structure (e.g., modular addition over Z/pZ):

```
System has a symmetry group G
    ↓
Decompose into irreducible representations of G
    ↓
Those representations ARE the natural modes
```

This motivates Fourier decomposition as a principled analysis strategy for modular
arithmetic tasks: the Fourier basis over Z/pZ is the natural basis for the group's
irreducible representations.
