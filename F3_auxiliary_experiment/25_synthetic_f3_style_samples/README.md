# 25_synthetic_f3_style_samples

Preview experiment for F3-style synthetic samples.

This folder only generates synthetic-section previews. It does not train a network.

Boundary:

- Training inputs are synthetic low-pass sections, not real F3 low-pass patches.
- Real F3 low-pass data is used only to estimate style statistics: envelope, dip/structure field, lateral coherence, and noise level.
- Labels remain synthetic residuals: `synthetic_wide - synthetic_lowpass`.
- Current preview uses a 0-35 Hz low-pass input definition and a 35 Hz+ residual label.
- Random texture strength is reduced and lateral smoothing is increased so the synthetic low-pass sections are closer to the coherent F3 low-pass style.
- Explicit middle/deep structural perturbations are applied before wavelet convolution: folds, local uplift/subsidence, and dipping faults. Shallow samples remain comparatively stable.
- Small segmented horizon-step offsets are added below sample 300 around strong reflectors to mimic minor reflector breaks seen in real F3 sections.
- F3 low-pass patches are now sampled in batch to estimate target lateral correlation and gradient-ratio statistics. Synthetic sections receive additional weak-reflection texture, amplitude mottling, phase jitter, and local amplitude loss to move those statistics closer to F3.
