# REQ_025: Attention Head Pattern Visualization

## Summary

Add attention head pattern visualization from the original Nanda grokking notebook. For each checkpoint, extract the full attention pattern tensor and visualize per-head heatmaps showing which input positions each head attends to.

## Implementation

### Analyzer: `AttentionPatternsAnalyzer`

- **File:** `analysis/analyzers/attention_patterns.py`
- **Name:** `attention_patterns`
- Extracts attention pattern tensor `(batch, n_heads, n_pos, n_pos)` from cache
- Stores full patterns as artifact for flexible visualization

### Renderers

- **File:** `visualization/renderers/attention_patterns.py`
- `render_attention_heads(epoch_data, epoch, to_position, from_position)` — 2x2 grid of per-head heatmaps
- `render_attention_single_head(epoch_data, epoch, head_idx, to_position, from_position)` — single head heatmap
- Position pair selector enables viewing different attention relationships (e.g., `= attending to a`)

### Dashboard Integration

- Attention Relationship dropdown for selecting position pairs
- Full-width attention head patterns plot (4 heads in 2x2 grid)
- Updates on epoch slider and position pair changes

### Library Function

- `extract_attention_patterns(cache, layer)` in `analysis/library/activations.py`

## Status: Complete
