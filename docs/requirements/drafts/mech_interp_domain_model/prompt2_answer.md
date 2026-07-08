The Residual Stream
This is arguably the most foundational object. Every component reads from and writes to it, but nothing owns it — it's a shared communication channel.

Dimensionality and effective rank — how much of the residual stream's capacity is actually being used?
Basis alignment — do meaningful features tend to align with coordinate axes, or are they in arbitrary directions? (This connects to the superposition hypothesis.)
Layer-wise drift — how much does a token's residual stream representation change between layers? Is change concentrated in particular layers?
Norm growth — does the residual stream norm grow, shrink, or stay stable across layers? Instability here can be interpretively meaningful.
Interference between components — when two heads both write to the residual stream, how much do their outputs interact versus stay orthogonal?


Individual Attention Heads
Heads are composite objects — they have both a QK and OV circuit — but the head as a unit is its own object of study, because the interesting question is often what functional role the head plays.

Head type classification — is this head a previous-token head, an induction head, a duplicate-token head, a positional head? These are discovered motifs.
Q-composition, K-composition, V-composition — is this head reading from the raw embedding, or is it reading from the output of a previous head? The composition score (Frobenius norm of the product vs. the factors) quantifies this.
Ablation sensitivity — how much does the model's output change when you zero out or mean-ablate this head? Some heads are load-bearing, others are nearly vestigial.
Faithfulness of a functional description — if you hypothesize that a head "detects X," you can measure how well its attention pattern or output is predicted by a simple rule for X.


Virtual Weights / Composed Circuits
This is one of the more subtle first-class objects in the Framework paper. When two heads compose — the output of head A in layer 1 feeds into the query, key, or value of head B in layer 2 — the effective operation is a product of their weight matrices. These virtual weights are the actual computational primitive.

Rank of the composed matrix — low rank composition is interpretively legible.
Alignment with meaningful subspaces — does the composed circuit read and write in directions that correspond to features you can name?
Whether composition is "tight" or "loose" — a tight circuit means head B is specifically and strongly reading from head A. A loose one means B is reading from a diffuse mixture of sources.
Induction circuit as a test case — the classic example is a K-composition between a previous-token head and an induction head. You can measure the composition score and verify the circuit story.


MLP Layers
MLPs are often treated as second-class citizens in the circuits framework — harder to decompose — but they've become first-class in more recent work (especially the "Transformer Feed-Forward Layers are Key-Value Memories" line of thinking).

Neuron activation sparsity — what fraction of neurons fire on a given input? High sparsity suggests more modular function.
Input and output weights as keys and values — the first layer weights can be thought of as "key" detectors, the second as "value" writers. What do those directions correspond to?
Superposition in neurons — is each neuron doing one thing or many things? Polysemanticity is measurable via e.g. how many distinct input classes activate a neuron.
MLP contribution to logit difference — same framing as OV circuits: how much is this MLP pushing outputs in the right direction for the task?
Interaction with residual stream — does the MLP act mostly additively (writing fresh information) or does it seem to be correcting/suppressing something already in the stream?


The Embedding and Unembedding Matrices (W_E and W_U)
These are boundary objects — the interface between token space and representation space — and they deserve their own treatment.

Similarity structure — do semantically or functionally related tokens have similar embeddings? For modular arithmetic, do tokens representing numbers that share residues cluster?
Alignment between W_E and W_U — in a tied or near-tied model, the embedding and unembedding are nearly transposes of each other. In an untied model, how much do they diverge?
Fourier structure — in modulo addition specifically, you'd expect W_E to have structure organized around Fourier modes of the modulus. Measuring how much variance is explained by those modes is very natural.
Effective dimensionality — what is the intrinsic dimensionality of the embedding space actually being used?


The through-line across all of these is the same instinct you saw in my earlier answer: what is the object doing, in what subspace, and how legibly? The Framework paper's deepest contribution is probably the argument that you can answer those questions compositionally — by understanding how these objects chain together — rather than only looking at each in isolation.