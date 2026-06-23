# Experiment 22 Context-Masked Local Calibration Design

## Goal

Use continuous 256-trace F3 narrow-band context while restricting every
wide-band supervised loss to a 32-trace calibration window around each
available training well.

## Data Condition

Experiment 22 keeps experiment 21's four-fold leave-one-well-out protocol.
For each fold, three wells provide local wide-band calibration and the fourth
well is held out for final inline and crossline evaluation.

- Calibration window: inline +/- 8, crossline +/- 16.
- Held-out guard: inline +/- 16, crossline +/- 32.
- Held-out wide data cannot affect samples, validation, checkpoint selection,
  normalization, or stopping.
- The held-out wide section is read only after the fold checkpoint is locked.

## Sample Geometry

Each local sample has shape `256 x 256`.

- Input context: a continuous F3 wide section is low-pass filtered to produce
  the complete 256-trace narrow-band input.
- Supervision region: exactly 32 contiguous traces inside the approved local
  calibration window.
- Target residual: `wide / narrow_p99 - narrow / narrow_p99` only inside the
  32-trace supervision region.
- Target values outside the supervision region are stored as zero.
- A binary mask records the supervised 32 traces and is stored separately.

The 256-trace context may extend beyond the wide calibration window because
only its low-pass version is exposed to the model. Wide values outside the
mask must never enter target arrays, losses, validation metrics, or plots.

## Training

The network remains the experiment 21 phase-consistent residual model.
Common pretraining remains unchanged and does not use F3 wide targets.

Fold fine-tuning uses:

- masked residual Smooth L1 loss;
- masked reconstructed-wide loss;
- masked trace correlation and phase/spectrum losses;
- full-context low-frequency leakage constraint derived only from the narrow
  input;
- the same fold configuration, epoch budget, optimizer, and checkpoint lock
  policy as experiment 21.

All masked losses divide by the number of valid supervised samples, not by the
full 256-trace patch area.

## Inference And Evaluation

Inference predicts the complete held-out inline and crossline sections.
Outputs remain:

- narrow input;
- residual prediction;
- direct recombination;
- high-pass residual recombination;
- wide reference read after checkpoint lock.

Evaluation uses the experiment 21 metrics and success gate. Additional sample
diagnostics compare adjacent-trace correlation and best time lag for experiment
21 local 32-trace samples, experiment 22 256-trace contexts, and experiment 18
samples.

## Required Artifacts

Create `22_context_masked_calibration/` with experiment-specific code, tests,
data, checkpoints, figures, and logs. Add a concise improvement record under
`docs/`. Generated arrays and checkpoints remain ignored by Git.

## Acceptance Criteria

- Sample input, residual target, and mask all have shape `256 x 256`.
- Every mask contains exactly `256 x 32` supervised values.
- Residual targets are exactly zero outside the mask.
- Reconstruction closure holds inside the mask.
- No held-out guard coordinate appears in a training or validation mask.
- Experiment 22 sample contexts have no artificial trace reordering or
  non-zero best-lag jumps introduced by extraction.
- All tests pass and all eight held-out sections are evaluated.
