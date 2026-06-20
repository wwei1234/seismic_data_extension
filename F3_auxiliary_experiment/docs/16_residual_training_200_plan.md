# 16 Residual Training 200 Plan

Goal: keep the 15/16 network and inference style, regenerate experiment 16 samples as single-channel high-frequency residual labels, train for 200 epochs, and evaluate the four well inline sections.

Steps:

1. Modify `code/02_generate_synthetic_dataset.py` so each well combination uses all 3 B-spline wavelets and both noise levels, producing `11 * 3 * 2 = 66` sections.
2. Save `train_labels.npy` and `val_labels.npy` as `wide - lowpass` residual labels. Keep separate section-level wide labels and residual labels for plotting and evaluation.
3. Modify `code/01_train.py` so Dataset treats labels as residual targets and reconstructs `wide_target = input + residual_target` inside the loss.
4. Fix `code/common.py` SEG-Y path to reuse `15_bspline_wavelet_bank/Rawdata/Seismic_data.sgy`.
5. Modify `code/02_predict_f3.py` to support selecting explicit SEG-Y inline numbers and saving inline metadata.
6. Add/modify evaluation to report metrics for each of the four well inline sections.
7. Run sample generation, 200-epoch training, selected-inline prediction, and evaluation.

