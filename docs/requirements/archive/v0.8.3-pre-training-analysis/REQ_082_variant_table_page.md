# REQ_082: Variant Table Page

**Status:** Complete
**Priority:** Medium
**Related:** REQ_074 (Variant Registry), REQ_076 (Peer Comparison)
**Last Updated:** 2026-03-26

---

## Problem Statement

Selecting a variant for analysis currently requires knowing its identifier in advance or scrolling through a dropdown in the left nav. There is no surface that shows all variants alongside their key metrics — loss trajectory shape, grokking timing, frequency health, performance classification — so the researcher can make an informed selection decision, notice patterns across variants, or quickly locate an outlier.

The `variant_registry.json` file already compiles per-variant metrics across all variants. The missing piece is a dashboard page that presents this data in a scannable, filterable table and wires row selection to the global variant selector so the clicked variant becomes active everywhere.

---

## Conditions of Satisfaction

### 1. Variant Table page

A dashboard page displaying all variants as rows in a data table.

**Required columns:**
- Family name
- Variant identifier (prime, seed, data_seed)
- Performance classification label (from `variant_summary.json`)
- Final test loss
- Grokking epoch (epoch of second descent onset; `-` if not grokked)
- Committed frequency count (at final epoch)

**Behavior:**
- Table is sortable by any column
- Table supports basic text filtering (family or variant ID substring match)
- Page loads without requiring a variant to be pre-selected

### 2. Row click → variant selection

Clicking a row selects that variant as if it had been chosen from the Variant Selector in the left nav.

- After click, all other dashboard pages that depend on variant state reflect the newly selected variant
- The Variant Selector in the left nav visually reflects the newly selected variant (stays in sync)
- Row click uses the same `variant-selector-store` mechanism as the left nav selector — no parallel state path

### 3. Visual distinction for active variant

The row corresponding to the currently active variant is visually highlighted so the researcher can see which variant is selected at a glance.

### 4. Tests

- Unit: table data loads from `variant_registry.json` and produces the correct number of rows and expected column values for at least one known variant
- Integration: clicking a row updates the `variant-selector-store` (tested via callback output)

---

## Constraints

- Data source is `variant_registry.json` — do not load per-variant artifact files to populate the table
- Row click must go through `variant-selector-store` (no separate selection state)
- Table component should use Dash AG Grid or Dash DataTable — whichever is already available in the dependency tree
- Sorting and filtering are client-side (no server round-trip required)
- The page is navigable from the top-level site nav

---

## Notes

- The current left nav Variant Selector is designed for single-variant selection and works well when you know what you want. This page is complementary — it serves the "survey mode" use case where the researcher is exploring the space of variants.
- Performance classification labels (`healthy`, `pathological`, etc.) from `variant_summary.json` are the most important single-column signal for finding variants worth studying.
- A future extension could add sparkline loss curves inline in each row, but this is not in scope for v1.
- AG Grid (Dash AG Grid) supports row selection highlighting natively and has better performance for larger tables than Dash DataTable — prefer it if it is in the dependency tree.

## Bug Fix (2026-04-05)

When clicking a row without a prior variant selection, the variant dropdown options were empty (`[]`), causing it to show blank even after `set_props` set its value. Root cause: `on_family_change` in `variant_selector.py` raises PreventUpdate when `stored_family == family_name` (the table had just written it), so it never populated options.

Fix: `on_row_selected` in `variant_table.py` now also calls `set_props` on the variant dropdown's `options` (computed from the registry for the selected family), alongside the existing `value` set_props call.
