# Residual Prediction Experiment

Scheme 2 trains the network to predict the bandwidth-extension residual.

```text
residual = wide_label - narrow_input
wide_prediction = narrow_input + network(narrow_input)
```

The loss is computed on `wide_prediction` against the wide label, so the network
learns only the missing component while the low-frequency background is inherited
from the input.

Run from the repository root:

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\residual_prediction_experiment\code\01_prepare_residual_dataset.py
D:\Anaconda\python.exe F3_auxiliary_experiment\residual_prediction_experiment\code\02_train_residual.py
D:\Anaconda\python.exe F3_auxiliary_experiment\residual_prediction_experiment\code\03_predict_f3_residual.py
D:\Anaconda\python.exe F3_auxiliary_experiment\residual_prediction_experiment\code\04_evaluate_residual.py
```

Outputs:

- `data/`: residual labels, F3 predictions, and metrics.
- `checkpoints/`: residual model weights.
- `logs/`: training history and curves.
- `figures/`: training, validation, prediction, spectrum, and phase figures.
