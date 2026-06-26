# 26_f3style_wideband_direct

This experiment keeps the F3-style synthetic sample generator from experiment 25, but changes the training target from high-frequency residuals to direct wideband sections.

## Design

- Input: synthetic wide sections low-passed to 0-35 Hz.
- Label: full synthetic wideband section.
- Normalization: per-section P99 absolute amplitude of the clean low-pass section.
- Noise: added only to normalized inputs, with levels 0.01 and 0.03.
- Augmentation: left-right flip only.
- Training: 200 epochs, direct wideband prediction loss.

Real F3 wideband data is not used for training labels. It is used only after training as the reference for inline/crossline evaluation.

## Outputs

- Samples: `data/train_inputs.npy`, `data/train_labels.npy`, `data/val_inputs.npy`, `data/val_labels.npy`
- Checkpoints: `data/checkpoints/best_model.pth`, `data/checkpoints/last_model.pth`
- Predictions: `data/predictions/`
- Evaluation figures and reports: `figures/prediction_evaluation/`
- Final analysis: `figures/wide26_f3style_final_analysis.txt`
