import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    BIAS_DATA_DIR,
    DATA_DIR,
    DT,
    FIGURE_DIR,
    HIGH_BAND,
    SOURCE_DATA_DIR,
    average_cube_spectrum,
    average_patch_spectrum,
    ensure_dirs,
    high_freq_ratio,
    spectrum_l1,
)


def main():
    ensure_dirs()

    synth_freq, synth_amp = average_patch_spectrum(
        SOURCE_DATA_DIR / "train_labels.npy", DT, max_patches=512
    )
    f3_freq, f3_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_wide_reference.npy", DT
    )
    narrow_freq, narrow_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_narrow_input.npy", DT
    )
    pred_freq, pred_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_wide_prediction.npy", DT
    )

    metrics = {
        "high_band_hz": HIGH_BAND,
        "synthetic_wide_high_freq_ratio": high_freq_ratio(synth_freq, synth_amp),
        "f3_reference_high_freq_ratio": high_freq_ratio(f3_freq, f3_amp),
        "f3_lowpass_high_freq_ratio": high_freq_ratio(narrow_freq, narrow_amp),
        "bias_prediction_high_freq_ratio": high_freq_ratio(pred_freq, pred_amp),
        "synthetic_vs_f3_spectrum_l1_0_100": spectrum_l1(
            synth_freq, synth_amp, f3_freq, f3_amp
        ),
        "bias_prediction_vs_f3_spectrum_l1_0_100": spectrum_l1(
            pred_freq, pred_amp, f3_freq, f3_amp
        ),
    }

    np.save(DATA_DIR / "diagnostic_spectra.npy", {
        "synthetic_wide": (synth_freq, synth_amp),
        "f3_reference": (f3_freq, f3_amp),
        "f3_lowpass": (narrow_freq, narrow_amp),
        "bias_prediction": (pred_freq, pred_amp),
    })
    np.save(DATA_DIR / "diagnostic_metrics.npy", metrics)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(synth_freq, synth_amp, lw=2.0, label="Synthetic wide label")
    ax.plot(f3_freq, f3_amp, "k--", lw=2.2, label="F3 reference")
    ax.plot(narrow_freq, narrow_amp, "b-", lw=1.8, alpha=0.85, label="F3 low-pass")
    ax.plot(pred_freq, pred_amp, "r-", lw=1.8, alpha=0.85, label="Bias-corrected prediction")
    ax.axvspan(HIGH_BAND[0], HIGH_BAND[1], color="red", alpha=0.12, label="35-80 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Synthetic wide label vs F3 reference spectra")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_synthetic_vs_f3_spectrum_diagnosis.png", dpi=300)
    plt.close(fig)

    print("Diagnostic metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if metrics["synthetic_wide_high_freq_ratio"] < 0.8 * metrics["f3_reference_high_freq_ratio"]:
        print("Diagnosis: synthetic wide labels are high-frequency deficient relative to F3.")
        print("Recommended route: wavelet high-frequency augmentation or target-spectrum matching.")
    else:
        print("Diagnosis: synthetic wide labels have sufficient high-frequency energy.")
        print("Recommended route: frequency-weighted loss.")


if __name__ == "__main__":
    main()
