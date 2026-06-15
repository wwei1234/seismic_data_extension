# High-Frequency Conservative Adjustment Experiment

This experiment follows `高频补充保守问题调整思路.txt`.

Steps:

1. Diagnose whether synthetic wide labels are high-frequency deficient compared
   with the F3 reference.
2. Quickly test inference-stage residual spectral gain correction.
3. Check whether synthetic narrow inputs and F3 low-pass inputs use consistent
   cutoff behavior.

Run from the repository root:

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\highfreq_conservative_adjustment_experiment\code\01_diagnose_spectra.py
D:\Anaconda\python.exe F3_auxiliary_experiment\highfreq_conservative_adjustment_experiment\code\02_inference_gain_correction.py --ratios 0.30
D:\Anaconda\python.exe F3_auxiliary_experiment\highfreq_conservative_adjustment_experiment\code\03_check_lowpass_consistency.py
D:\Anaconda\python.exe F3_auxiliary_experiment\highfreq_conservative_adjustment_experiment\code\04_train_freq_weighted.py --epochs 300 --hf-boost-factor 2.0 --hf-loss-weight 0.5
D:\Anaconda\python.exe F3_auxiliary_experiment\highfreq_conservative_adjustment_experiment\code\05_predict_freq_weighted_f3.py
D:\Anaconda\python.exe F3_auxiliary_experiment\highfreq_conservative_adjustment_experiment\code\06_evaluate_freq_weighted.py
```

The gain-correction script uses the bias-correction experiment outputs as the
baseline prediction and saves corrected predictions under this experiment's
`data/` directory.

The frequency-weighted training route is selected when the synthetic wide labels
already have enough 35-80 Hz energy, but the network prediction remains
conservative.
