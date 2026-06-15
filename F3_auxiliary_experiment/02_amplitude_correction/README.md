# Amplitude Correction Experiment

This folder contains the post-processing experiment for scheme 1:

```text
corrected = narrow_input + highpass(network_prediction)
```

The low-frequency background is forced to come from the low-pass F3 input, while
only the predicted high-frequency residual is taken from the network output.

Run from the repository root:

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\amplitude_correction_experiment\code\01_apply_amplitude_correction.py
```

Outputs:

- `data/`: corrected prediction arrays and metrics.
- `figures/`: section, spectrum, and phase comparison figures.
