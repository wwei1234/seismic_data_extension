# High-Pass Constrained Residual Experiment

This experiment trains the network to predict only the band-limited missing
component between the narrow input and the wide label.

```text
narrow input band:   3-6-25-35 Hz
wide target band:    3-6-55-75 Hz
residual output band: 25-35-55-75 Hz
```

Training logic:

```text
residual_label = bandpass(wide_label - narrow_input, 25-35-55-75 Hz)
raw_residual = network(narrow_input)
residual_prediction = bandpass(raw_residual, 25-35-55-75 Hz)
wide_prediction = narrow_input + residual_prediction
```

The bandpass filter is applied after the network output and before the loss, so
low-frequency/DC leakage from the residual branch cannot be added back into the
wide prediction.

Run from the repository root:

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\highpass_residual_experiment\code\01_prepare_residual_dataset.py
D:\Anaconda\python.exe F3_auxiliary_experiment\highpass_residual_experiment\code\02_train_residual.py --epochs 300
D:\Anaconda\python.exe F3_auxiliary_experiment\highpass_residual_experiment\code\03_predict_f3_residual.py
D:\Anaconda\python.exe F3_auxiliary_experiment\highpass_residual_experiment\code\04_evaluate_residual.py
```

Outputs:

- `data/`: band-limited residual labels, F3 predictions, and metrics.
- `checkpoints/`: high-pass constrained residual model weights.
- `logs/`: training history and curves.
- `figures/`: training sample, prediction, spectrum, and phase figures.
