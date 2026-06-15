"""Plot 4 well wavelets' spectra overlaid with well-tie seismic trace spectra."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from signal_utils import average_amplitude_spectrum

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = ROOT.parent / "01_basic_strategy" / "data"
DT = 0.004
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load pre-computed data
traces = np.load(SOURCE_DATA_DIR / "well_traces.npy", allow_pickle=True).item()
wavelets = np.load(SOURCE_DATA_DIR / "well_estimated_wavelets.npy", allow_pickle=True)
if wavelets.ndim == 2:
    # shape (n_wells, n_samples) — convert to dict keyed by well order
    well_names = list(traces.keys())
    wavelets = {wn: wavelets[i] for i, wn in enumerate(well_names)}

wide_wavelets = np.load(SOURCE_DATA_DIR / "well_wide_wavelets.npy", allow_pickle=True).item()
narrow_wavelets = np.load(SOURCE_DATA_DIR / "well_narrow_wavelets.npy", allow_pickle=True).item()

print("Well names:", list(traces.keys()))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, well_name in enumerate(traces):
    ax = axes[idx]
    trace = traces[well_name]

    # Seismic trace spectrum
    ft, at = average_amplitude_spectrum(trace[:, None], DT)
    ax.plot(ft, at, "k-", lw=2.5, label=f"{well_name} seismic trace")

    # Original estimated wavelet spectrum
    w = wavelets[well_name]
    fw, aw = average_amplitude_spectrum(w[:, None], DT)
    ax.plot(fw, aw, "gray", lw=1.5, alpha=0.6, label="Estimated wavelet")

    # Wide wavelet spectrum
    ww = wide_wavelets[well_name]
    fww, aww = average_amplitude_spectrum(ww[:, None], DT)
    ax.plot(fww, aww, "r-", lw=2, label="Wide wavelet (3-6-55-75 Hz)")

    # Narrow wavelet spectrum
    nw = narrow_wavelets[well_name]
    fnw, anw = average_amplitude_spectrum(nw[:, None], DT)
    ax.plot(fnw, anw, "b-", lw=2, label="Narrow wavelet (3-6-25-35 Hz)")

    ax.set_xlim(0, 125)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"{well_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Well-tie seismic trace vs estimated wavelet spectra", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = FIGURE_DIR / "wavelet_vs_seismic_spectrum.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved {out_path}")
