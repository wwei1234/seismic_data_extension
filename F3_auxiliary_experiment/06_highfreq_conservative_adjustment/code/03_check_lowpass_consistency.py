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
    NARROW_BAND,
    SOURCE_DATA_DIR,
    average_cube_spectrum,
    average_patch_spectrum,
    ensure_dirs,
    high_freq_ratio,
    spectrum_l1,
)


def main():
    ensure_dirs()

    syn_freq, syn_amp = average_patch_spectrum(
        SOURCE_DATA_DIR / "train_inputs.npy", DT, max_patches=512
    )
    f3_freq, f3_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_narrow_input.npy", DT
    )

    metrics = {
        "narrow_band_hz": NARROW_BAND,
        "synthetic_narrow_high_freq_ratio_35_80": high_freq_ratio(syn_freq, syn_amp),
        "f3_lowpass_high_freq_ratio_35_80": high_freq_ratio(f3_freq, f3_amp),
        "synthetic_narrow_vs_f3_lowpass_spectrum_l1_0_100": spectrum_l1(
            syn_freq, syn_amp, f3_freq, f3_amp
        ),
    }
    np.save(DATA_DIR / "lowpass_consistency_metrics.npy", metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(syn_freq, syn_amp, lw=2.2, label="Synthetic narrow input")
    ax.plot(f3_freq, f3_amp, "k--", lw=2.2, label="F3 low-pass input")
    ax.axvline(25.0, color="orange", ls=":", lw=1.8, label="25 Hz")
    ax.axvline(35.0, color="red", ls=":", lw=1.8, label="35 Hz")
    ax.set_xlim(0, 80)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Synthetic narrow input vs F3 low-pass consistency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_lowpass_consistency.png", dpi=300)
    plt.close(fig)

    print("Low-pass consistency metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
