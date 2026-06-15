# Bias-Corrected Residual Prediction Experiment

This experiment follows `偏置修正.txt`.

Training target:

```text
residual_label = wide_label - narrow_input
raw_residual = UNetCBAM(narrow_input)
residual_prediction = raw_residual - mean(raw_residual)
wide_prediction = narrow_input + residual_prediction
```

Bias-control strategy:

- subtract the mean from every predicted residual patch;
- add a residual mean penalty to the loss;
- do not apply high-pass filtering before the training loss;
- apply a `25-35-55-75 Hz` residual post-filter only during F3 inference.

Run from the repository root:

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\bias_correction_experiment\code\01_prepare_residual_dataset.py
D:\Anaconda\python.exe F3_auxiliary_experiment\bias_correction_experiment\code\02_train_residual.py --epochs 300
D:\Anaconda\python.exe F3_auxiliary_experiment\bias_correction_experiment\code\03_predict_f3_residual.py
D:\Anaconda\python.exe F3_auxiliary_experiment\bias_correction_experiment\code\04_evaluate_residual.py
```
