# F3 Auxiliary Bandwidth Extension Experiment

This experiment follows the corrected auxiliary workflow:

1. Use F3 well logs to compute time-domain reflectivity.
2. Estimate the wide/reference wavelet from well reflectivity and well-tie seismic traces.
3. Generate the narrow wavelet by band-limiting the estimated wide wavelet.
4. Build synthetic 2D training pairs:
   - input: structural reflectivity convolved with the narrow wavelet plus noise
   - label: the same reflectivity convolved with the wide wavelet
5. Train U-Net + CBAM on the synthetic pairs.
6. Low-pass filter the F3 seismic volume and use it as the prediction input.
7. Evaluate the prediction against the original F3 seismic volume as the wide reference.

## Directory Layout

- `code/`: all runnable Python code.
- `Rawdata/`: local F3 SEG-Y and well-log inputs. This directory is ignored by Git because it contains large public data files.
- `data/`: generated wavelets, well reflectivity, and synthetic training arrays. Ignored by Git.
- `figures/`: generated wavelet, synthetic sample, prediction, and spectrum figures. Ignored by Git.
- `checkpoints/`: trained model weights. Ignored by Git.
- `logs/`: training curves and loss history. Ignored by Git.
- `predictions/`: generated F3 low-pass input, prediction, reference, and metrics. Ignored by Git.

Expected local raw-data layout:

```text
F3_auxiliary_experiment/
  Rawdata/
    Seismic_data.sgy
    Well_data/
      All_wells_RawData/
        Checkshot/
        Lasfiles/
        Track/
        Tops/
```

## Full Run

Run from the repository root:

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\code\01_estimate_wavelets.py
D:\Anaconda\python.exe F3_auxiliary_experiment\code\02_generate_synthetic_dataset.py --num-samples 3000 --patch-size 128
D:\Anaconda\python.exe F3_auxiliary_experiment\code\03_train.py --epochs 80 --batch-size 16 --base-c 32
D:\Anaconda\python.exe F3_auxiliary_experiment\code\04_predict_f3.py --inline-start 0 --num-inlines -1 --output-prefix f3_full
D:\Anaconda\python.exe F3_auxiliary_experiment\code\05_evaluate.py --prefix f3_full
```

## Quick Smoke Test

The smoke test uses a tiny synthetic dataset and a very small model. It only checks that the
pipeline works; its metrics are not meaningful.

```powershell
D:\Anaconda\python.exe F3_auxiliary_experiment\code\01_estimate_wavelets.py
D:\Anaconda\python.exe F3_auxiliary_experiment\code\02_generate_synthetic_dataset.py --num-samples 80 --patch-size 128
D:\Anaconda\python.exe F3_auxiliary_experiment\code\03_train.py --epochs 1 --batch-size 4 --base-c 8
D:\Anaconda\python.exe F3_auxiliary_experiment\code\04_predict_f3.py --base-c 8 --inline-start 260 --num-inlines 1 --output-prefix smoke_f3
D:\Anaconda\python.exe F3_auxiliary_experiment\code\05_evaluate.py --prefix smoke_f3
```
