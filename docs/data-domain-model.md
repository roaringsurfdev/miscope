This document captures the data model for mechanistic interpretability research. Not included in this document is any infrastructure-specific domain model language outside of Family, Variant, and Model.

# Schema Definition Format
### Object Name:
*Object description*
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Text | data_type |  Text | Text | STATUS_VALUE | Text |

STATUS_VALUE: NEW, EXISTS, DELTA, CHEAP

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Text | Text | KEY_TYPE | List | Text | Text | Text |

*KEY_TYPE: PRIMARY, FOREIGN*

#### Indexes:
| Index Name | Object Name | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Text | Text | List | INDEX_TYPE | Text | Text |

*INDEX_TYPE: UNIQUE, NA*
## Schema Definitions - Platform Objects
*This section is reserved for objects defined by the **platform**. They should not by controversial or require mathematical proof.* 

### Family
*List of model architectures + tasks that provide context for model variations (Variants)*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| family_id | int |  Key | - | - | - |
| name | string |  - | - | - | Full Family Name |
| abbr | string |  - | - | - | Abbreviated Family Name |

| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Family_PK | Family | PRIMARY_KEY | family_id | - | - | Surrogate Key |

### Family_Parameter
*List of per-family parameters used to differentiate model Variants. Does not include model_seed or data_seed, because all Variants across all model architectures will use pre-defined seeds. This table is only to be used for parameters intended to capture different task parameters.*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| family_id | int |  Key | - | - | - |
| name | string |  - | Parameter Name | - | - |
| datatype | string |  - | Parameter Data Type | - | - |

| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Family_Parameter_PK | Family_Parameter | PRIMARY_KEY | family_id, name | - | - | - |

### Variant
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int |  Key | - | - | - |
| name | string |  - | - | - | Full Variant Name |
| abbr | string |  - | - | - | Abbreviated Variant Name |
| model_seed | int |  - | - | - | seed value used to initialize the model |
| data_seed | int |  - | - | - | seed value used to randomize dataset creation/selection |

| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Variant_PK | Variant | PRIMARY_KEY | variant_id | - | - | Surrogate Key |

### Probe

##  Schema Definitions - Intrinsic Objects
*This section is reserved for objects defined by the **model architecture**. They should not be controversial or require mathematical proof.*


### MLP *(Block)*
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

### MLP_Neuron
*I think this might properly belong in the theoretical subset*
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT(0) |
| neuron_index | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| MLP_Neuron_PK | MLP_Neuron | PRIMARY_KEY | variant_id, layer_index, neuron_index, epoch | Text | Text | Text |
| MLP_Neuron_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

### Attention *(Block)*

| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT(0) |
| head_index | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Attention_PK | - | PRIMARY_KEY | variant_id, layer_index, head_index, epoch | Text | Text | Text |
| Attention_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

### Residual Stream
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Residual_Stream_PK | - | PRIMARY_KEY | variant_id, epoch | Text | Text | Text |
| Residual_Stream_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

## Schema Definitions - Performance Metrics
*This section is reserved for stores that capture performance metrics that are well-established, uncontroversial, and fall more squarely in the Performance Optimization space as opposed to data about model internals. A prime example is Loss values.*

### Loss
*Stores Train and Test Loss values over training*
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| loss_type | int | Key | - | - | - |
| epoch | int | Key | - | - | - |
| value | float | - | - | - | - |

## Schema Definitions - Virtual Weights/Derived Objects
*This section is reserved for objects that have proven useful in the study of **mechanistic interpretability**. These objects will require more rigor to justify.*

### QK_Circuit
*W_Q^T W_K — composed virtual weight*
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT(0) |
| head_index | int | Key | - | - | - |
| epoch | int | Key | - | - | - |

#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| QK_Circuit_PK | - | PRIMARY_KEY | variant_id, layer_index, head_index, epoch | NA | NA |
| QK_Circuit_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

#### Indexes:
| Index Name | Object Name | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| QK_Circuit_IDX01 | QK_Circuit | head_id, variant_id | UNIQUE | Text | Text |

#### Questions:
Is there one QK_Circuit per layer + head_index pair? Same question for the OV_Circuit.

### OV_Circuit
*W_O W_V — composed virtual weight*
| Field Name | Data Type | Source Analyzer | Description | Status | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| variant_id | int | Key | - | - | - |
| head_index | int | Key | - | - | - |
| layer_index | int | Key | - | - | DEFAULT(0) |
| epoch | int | Key | - | - | - |


#### Keys:
| Key Name | Reference | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| OV_Circuit_PK | - | PRIMARY_KEY | variant_id, layer_index, head_index, epoch | NA | NA |
| OV_Circuit_FK_Variant | Variant | FOREIGN_KEY | variant_id | - | - |

#### Indexes:
| Index Name | Object Name | Type | Fields | Description | Note |
| ---- | ---- | ---- | ---- | ---- | ---- |
| QK_Circuit_IDX01 | QK_Circuit | head_id, variant_id | UNIQUE | Text | Text |


## Notes

#### Decomposition Techniques

Mode vs. Basis — this is the key distinction
A mode is one element. A basis is the full set of them.

* A single Fourier frequency → a mode
* The complete set of Fourier frequencies → the Fourier basis
----
Spectral Decomposition, where decomposition reveals a *spectrum* that tells you how much of something is present

* Eigendecomposition (spectrum of eigenvalues)
* PCA (spectrum of variances)
* DFT/Fourier (spectrum of frequencies)
* SVD (spectrum of singular values)
* DMD (spectrum of dynamic eigenvalues)

| Decomposition | "Shape" component | "Weight/Strength" component | "Behavior" component |
| ---- | ---- | ---- | ---- |
| Eigendecomposition | Eigenvector | Eigenvalue | — | 
| SVD | Left/right singular vectors | Singular value | — | 
| PCA | Principal component (loading) | Explained variance | — | 
| Fourier/DFT | Frequency (basis function) | Amplitude | Phase | 
| DMD | DMD mode (spatial) | Amplitude | Eigenvalue (growth + frequency) | 

#### DMD

Analysis Hierarchy

```
Koopman operator (theoretical, infinite-dimensional)
    ↓  approximated by
DMD matrix A (finite, numerical)
    ↓  decomposed into
DMD modes (vectors) + eigenvalues (scalars)
```

Full DMD output inventory

| Artifact | Type| What it tells you |
| ---- | ---- | ---- |
| A (or Ã, the reduced version) | Matrix | The linear operator approximating the dynamics |
| Φ (DMD modes) | Set of vectors | Spatial patterns in the data | 
| λ (eigenvalues) | Set of complex scalars | Growth rate + frequency of each mode | 
| ω (continuous eigenvalues) | Set of complex scalars | log(λ)/dt — more interpretable form | 
| b (mode amplitudes) | Set of scalars | How much each mode contributes at t=0 | 
| Φ·b (reconstruction) | Tensor | Optional: the reconstructed/projected data

Stored structure

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

``` ModelVariant → HypothesizedDecomposition → DecompositionResults ```

```
System has a symmetry group G
    ↓
Decompose into irreducible representations of G
    ↓
Those representations ARE the natural "modes"
```

```
ModelVariant
  ├── architecture / hyperparameters
  ├── task
  │     ├── name (e.g. "modulo_addition")
  │     └── group_structure (e.g. "Z/nZ", n=113)
  └── checkpoints[]
          └── checkpoint
                ├── step / epoch
                ├── loss / accuracy
                └── decomposition_analyses[]  ←— the key join point
                        └── DecompositionAnalysis
                              ├── strategy (e.g. "fourier", "svd", "dmd")
                              ├── hypothesized (bool)
                              ├── artifacts{}  ←— strategy-specific
                              └── metrics{}   ←— comparable across strategies
```

```
FourierAnalysis:
  ├── dominant_frequencies[]     # which k values are active
  ├── amplitudes[]               # per frequency
  ├── phases[]                   # per frequency
  ├── explained_variance_ratio   # scalar — trackable over time
  ├── frequency_sparsity         # scalar — does it concentrate on a few k?
  └── reconstruction_error       # scalar
```