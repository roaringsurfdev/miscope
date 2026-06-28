QK Circuit — this is about which tokens attend to which. The QK circuit computes attention patterns, so what you want to capture is the structure of those patterns:

Entropy of attention distributions — are heads attending sharply (low entropy, focused) or diffusely? In modular addition, you might expect heads to attend very selectively.
Symmetry of attention — does token A attend to token B as much as B attends to A? Or is it asymmetric?
Clustering / block structure — do certain token groups consistently attend to each other? You could measure this with something like the mutual information between query position and key position.
Consistency across inputs — does the attention pattern vary a lot depending on the specific operands, or is it relatively fixed? Fixed patterns suggest the head is doing something structural rather than content-dependent.
Eigenstructure of W_Q^T W_K — this matrix directly governs what the circuit "compares." Its eigenvectors tell you what features are being matched. In a modulo addition model, you might find these align with Fourier components.
Rank of W_Q^T W_K — low rank means only a few "dimensions of comparison" are actually active.


OV Circuit — this is about what information gets moved and how it's transformed. The OV circuit determines what the attended-to tokens actually write into the residual stream:

Eigenstructure of W_O W_V — the key matrix here. Its eigenvectors tell you what directions in residual stream space the circuit is reading from and writing to.
Rank — same idea. Low rank means the circuit is doing something relatively simple and legible.
Alignment with embedding/unembedding directions — do the OV circuit's output directions point toward meaningful output logits? In modular addition, you'd want to check whether they align with Fourier modes that the model uses to represent the answer.
Whether it acts as a copying circuit — does W_O W_V ≈ identity (up to a scalar) in some subspace? Copying circuits are a known motif.
Input-output linearity — how linear is the map from input token embedding to output contribution? Deviations from linearity suggest something more complex.
Contribution to the logit difference — for a classification task like mod addition, you can measure how much each head's OV output shifts the logits in the correct direction vs. incorrect ones.


One framing that ties both together: the QK circuit answers "what to look at" and the OV circuit answers "what to do with what you see." So jointly, you might also ask: does the QK circuit select tokens whose OV output is actually useful? That coupling is where a lot of the interesting mechanistic story lives in these toy models.