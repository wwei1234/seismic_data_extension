import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_DIR, DT, FIGURE_DIR, NARROW_BAND, WAVELET_LENGTH_MS, WELL_POSITIONS, WIDE_BAND
from signal_utils import average_amplitude_spectrum, estimate_wavelet_frequency_domain, shape_wavelet_to_band
from well_utils import load_well_reflectivity, load_well_trace


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    wavelets = []
    reflectivities = {}
    traces = {}

    for well_name in WELL_POSITIONS:
        reflectivity, _, _ = load_well_reflectivity(well_name)
        trace = load_well_trace(well_name)
        wavelet = estimate_wavelet_frequency_domain(
            reflectivity,
            trace,
            dt=DT,
            wavelet_length_ms=WAVELET_LENGTH_MS,
            water_level=0.03,
            smooth_sigma=1.2,
        )
        if wavelet[len(wavelet) // 2] < 0:
            wavelet = -wavelet
        wavelets.append(wavelet)
        reflectivities[well_name] = reflectivity
        traces[well_name] = trace

    estimated = np.mean(np.stack(wavelets, axis=0), axis=0)
    if estimated[len(estimated) // 2] < 0:
        estimated = -estimated
    estimated /= np.max(np.abs(estimated)) + 1e-8

    wide = shape_wavelet_to_band(estimated, DT, WIDE_BAND)
    narrow = shape_wavelet_to_band(estimated, DT, NARROW_BAND)

    np.save(DATA_DIR / "well_reflectivities.npy", reflectivities)
    np.save(DATA_DIR / "well_traces.npy", traces)
    np.save(DATA_DIR / "well_estimated_wavelets.npy", np.stack(wavelets, axis=0))
    np.save(DATA_DIR / "estimated_wide_wavelet.npy", wide)
    np.save(DATA_DIR / "estimated_narrow_wavelet.npy", narrow)

    time_ms = (np.arange(wide.size) - wide.size // 2) * DT * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for well_name, wavelet in zip(WELL_POSITIONS, wavelets):
        axes[0].plot(time_ms, wavelet, lw=1.2, alpha=0.55, label=well_name)
    axes[0].plot(time_ms, wide, "r-", lw=2.5, label="wide target")
    axes[0].plot(time_ms, narrow, "b-", lw=2.5, label="narrow input")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    freqs_w, amp_w = average_amplitude_spectrum(wide[:, None], DT)
    freqs_n, amp_n = average_amplitude_spectrum(narrow[:, None], DT)
    axes[1].plot(freqs_w, amp_w, "r-", lw=2.5, label="wide target")
    axes[1].plot(freqs_n, amp_n, "b-", lw=2.5, label="narrow input")
    axes[1].set_xlim(0, 125)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_wavelets_from_wells.png", dpi=300)
    plt.close(fig)

    print(f"Saved wavelets to {DATA_DIR}")
    print(f"Saved figure to {FIGURE_DIR / '01_wavelets_from_wells.png'}")


if __name__ == "__main__":
    main()
