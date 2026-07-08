# REQ_028: UI Improvements — Variant Dropdown

## Summary

Two usability improvements to the variant dropdown on the Analysis tab:

1. **Sort variants alphabetically** by variant name for consistent ordering
2. **Default to no selection** so users can select the first variant without switching away and back

## Implementation

### Changes

- **File:** `dashboard/components/family_selector.py`
  - `get_variant_choices()`: Added `choices.sort(key=lambda c: c[1])` after building choices
  - `get_variant_table_data()`: Added `rows.sort(key=lambda r: r[3])` for consistency
- **File:** `dashboard/app.py`
  - Variant dropdown initialized with `value=None` instead of implicit first-item default

### Behavior

- Variant dropdown now shows variants in alphabetical order (e.g., `p101_seed485` before `p101_seed999`)
- On page load, no variant is pre-selected — user must explicitly choose one
- Family change and refresh both reset to `value=None`

## Status: Complete
