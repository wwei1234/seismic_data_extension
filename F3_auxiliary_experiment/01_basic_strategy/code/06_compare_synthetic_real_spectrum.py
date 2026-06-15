import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_DIR, DT, FIGURE_DIR, PREDICTION_DIR


def accumulate_average_spectrum(section_iter, dt):
    amp_sum = None
    n_sections = 0
    freqs = None
    for section in section_iter:
        section = np.asarray(section, dtype=np.float32)
        work = section - np.mean(section, axis=0, keepdims=True)
        spec = np.fft.rfft(work, axis=0)
        amp = np.mean(np.abs(spec), axis=1)
        if amp_sum is None:
            amp_sum = np.zeros_like(amp, dtype=np.float64)
            freqs = np.fft.rfftfreq(section.shape[0], dt)
        amp_sum += amp
        n_sections += 1
    if n_sections == 0:
        raise ValueError("No sections were provided for spectrum calculation.")
    amp = amp_sum / n_sections
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return freqs, amp


def iter_dict_sections(path):
    data = np.load(path, allow_pickle=True).item()
    for key in sorted(data):
        yield data[key]


def iter_cube_sections(path, max_inlines=None):
    cube = np.load(path, mmap_mode="r")
    n_inline = cube.shape[0] if max_inlines is None else min(cube.shape[0], max_inlines)
    for idx in range(n_inline):
        yield cube[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-prefix", default="request_f3_full")
    parser.add_argument("--max-real-inlines", type=int, default=None)
    parser.add_argument("--fmax", type=float, default=100.0)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_narrow_path = DATA_DIR / "synthetic_section_inputs.npy"
    synthetic_wide_path = DATA_DIR / "synthetic_section_labels.npy"
    real_narrow_path = PREDICTION_DIR / f"{args.real_prefix}_narrow_input.npy"
    real_wide_path = PREDICTION_DIR / f"{args.real_prefix}_wide_reference.npy"

    freq_syn_n, amp_syn_n = accumulate_average_spectrum(
        iter_dict_sections(synthetic_narrow_path), DT
    )
    freq_syn_w, amp_syn_w = accumulate_average_spectrum(
        iter_dict_sections(synthetic_wide_path), DT
    )
    freq_real_n, amp_real_n = accumulate_average_spectrum(
        iter_cube_sections(real_narrow_path, args.max_real_inlines), DT
    )
    freq_real_w, amp_real_w = accumulate_average_spectrum(
        iter_cube_sections(real_wide_path, args.max_real_inlines), DT
    )

    np.save(
        DATA_DIR / "synthetic_vs_real_lowpass_spectra.npy",
        {
            "freq_synthetic_narrow": freq_syn_n,
            "amp_synthetic_narrow": amp_syn_n,
            "freq_synthetic_wide_label": freq_syn_w,
            "amp_synthetic_wide_label": amp_syn_w,
            "freq_real_f3_lowpass": freq_real_n,
            "amp_real_f3_lowpass": amp_real_n,
            "freq_real_f3_wide_reference": freq_real_w,
            "amp_real_f3_wide_reference": amp_real_w,
            "real_prefix": args.real_prefix,
            "max_real_inlines": args.max_real_inlines,
        },
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_syn_n, amp_syn_n, "b-", lw=2.2, label="Synthetic narrow input")
    ax.plot(freq_real_n, amp_real_n, "k--", lw=2.2, label="F3 low-pass input")
    ax.plot(freq_syn_w, amp_syn_w, "r-", lw=1.8, alpha=0.8, label="Synthetic wide label")
    ax.plot(freq_real_w, amp_real_w, color="darkorange", ls="-.", lw=2.2, label="F3 wide reference")
    ax.set_xlim(0, args.fmax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title("Synthetic records vs F3 low-pass spectra")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = FIGURE_DIR / "06_synthetic_vs_f3_lowpass_spectrum.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure: {out_path}")
    print(f"Saved spectra: {DATA_DIR / 'synthetic_vs_real_lowpass_spectra.npy'}")
    print(f"Synthetic narrow peak frequency: {freq_syn_n[int(np.argmax(amp_syn_n))]:.2f} Hz")
    print(f"F3 low-pass peak frequency: {freq_real_n[int(np.argmax(amp_real_n))]:.2f} Hz")
    print(f"F3 wide-reference peak frequency: {freq_real_w[int(np.argmax(amp_real_w))]:.2f} Hz")


if __name__ == "__main__":
    main()
