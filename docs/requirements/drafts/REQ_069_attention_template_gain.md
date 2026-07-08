# REQ_069: Attention Tuning from Healthy Model Template (nice-to-have)

**Status:** Draft — deferred, pending REQ_068 results
**Priority:** Low
**Depends on:** REQ_068 (Frequency-Selective Attention Tuning)
**Last Updated:** 2026-03-11

## Problem Statement

REQ_068 requires manually specifying the gain values for a frequency-selective intervention. If the intervention proves effective, a natural next step is to derive the gain function automatically from a healthy reference variant with the same prime, rather than choosing values by hand. The healthy model's attention frequency profile at plateau onset defines the "target" distribution; the gap between that and the intervention variant's profile determines the gain vector.

## Sketch

- Load a healthy same-prime variant at its plateau onset epoch
- Compute its attention Fourier profile (mean QK^T fraction per frequency, per REQ_066 attention panel computation)
- Load the intervention variant at its plateau onset epoch; compute the same profile
- Gain vector = healthy profile / intervention profile, clipped to [min_gain, max_gain]
- Use this gain vector as input to the REQ_068 intervention config

## Conditions of Satisfaction

TBD — to be written once REQ_068 results are in hand and the value of this approach is confirmed.

## Notes

- If REQ_068 shows that manual gain specification is effective, this becomes a usability improvement (reduces guesswork, makes interventions reproducible across variants with the same prime)
- If REQ_068 shows that the intervention is ineffective or highly sensitive to gain values, this requirement may not be worth pursuing
- The config schema in REQ_068 already supports dict-valued gain (frequency → scalar), so no schema migration needed
