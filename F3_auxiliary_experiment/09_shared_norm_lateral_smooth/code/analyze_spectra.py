"""Print numerical comparison of wavelet vs seismic trace spectra."""
import numpy as np
from pathlib import Path

DT = 0.004
p = Path(r'd:\桌面\基于深度学习的地震数据处理\地震数据拓频\F3_auxiliary_experiment\01_basic_strategy\data')
traces = np.load(p / "well_traces.npy", allow_pickle=True).item()
wavelets_arr = np.load(p / "well_estimated_wavelets.npy")
well_names = list(traces.keys())
wavelets = {wn: wavelets_arr[i] for i, wn in enumerate(well_names)}
wide = np.load(p / "well_wide_wavelets.npy", allow_pickle=True).item()
narrow = np.load(p / "well_narrow_wavelets.npy", allow_pickle=True).item()

def spec(x):
    s = np.abs(np.fft.rfft(x - x.mean()))
    f = np.fft.rfftfreq(x.size, DT)
    s = s / (s.max() + 1e-12)
    return f, s

def band_pct(f, s, lo, hi):
    m = (f >= lo) & (f <= hi)
    return 100.0 * s[m].sum() / (s.sum() + 1e-12)

print("=" * 80)
print("4-Well Wavelet vs Seismic Trace Spectrum Analysis")
print("=" * 80)

for wn in traces:
    ft, st = spec(traces[wn])
    fw, sw = spec(wavelets[wn])
    fww, sww = spec(wide[wn])
    fnw, snw = spec(narrow[wn])
    print(f"\n--- {wn} ---")
    hdr = f"  {'':20s}  {'Peak Hz':>8s}  {'0-35Hz%':>8s}  {'35-75Hz%':>9s}  {'75-125Hz%':>10s}"
    print(hdr)
    print("  " + "-" * 65)
    for tag, f, s in [("Seismic trace", ft, st), ("Est. wavelet", fw, sw),
                       ("Wide wavelet", fww, sww), ("Narrow wavelet", fnw, snw)]:
        peak = f[np.argmax(s)]
        lo = band_pct(f, s, 0, 35)
        md = band_pct(f, s, 35, 75)
        hi = band_pct(f, s, 75, 125)
        print(f"  {tag:20s}  {peak:8.1f}  {lo:8.1f}  {md:9.1f}  {hi:10.1f}")

print("\n--- Mean across 4 wells ---")
for tag, data in [("Seismic trace", traces), ("Est. wavelet", wavelets),
                   ("Wide wavelet", wide), ("Narrow wavelet", narrow)]:
    peaks, los, mids, his = [], [], [], []
    for wn in traces:
        f, s = spec(data[wn])
        peaks.append(f[np.argmax(s)])
        los.append(band_pct(f, s, 0, 35))
        mids.append(band_pct(f, s, 35, 75))
        his.append(band_pct(f, s, 75, 125))
    print(f"  {tag:20s}  peak={np.mean(peaks):.1f} Hz  "
          f"low={np.mean(los):.1f}%  mid={np.mean(mids):.1f}%  hi={np.mean(his):.1f}%")
