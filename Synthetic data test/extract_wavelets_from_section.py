import argparse
import os
import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey


def load_seismic_section(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
    elif ext in {".sgy", ".segy"}:
        try:
            import segyio
        except ImportError as exc:
            raise ImportError("Reading SEG-Y needs segyio: pip install segyio") from exc

        with segyio.open(path, "r", ignore_geometry=True) as f:
            data = np.asarray([np.copy(trace) for trace in f.trace]).T
    else:
        raise ValueError("Only .npy, .sgy and .segy inputs are supported.")

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D seismic section, got shape {data.shape}.")

    return data.astype(np.float64)


def read_segy_basic_info(path):
    with open(path, "rb") as f:
        f.seek(3200)
        binary_header = f.read(400)

    dt_us = struct.unpack(">H", binary_header[16:18])[0]
    ns = struct.unpack(">H", binary_header[20:22])[0]
    sample_format = struct.unpack(">H", binary_header[24:26])[0]
    bytes_per_sample = {2: 4, 3: 2, 5: 4, 8: 1}.get(sample_format)
    if bytes_per_sample is None:
        raise ValueError(
            f"SEG-Y sample format {sample_format} is not supported by this reader."
        )

    trace_size = 240 + ns * bytes_per_sample
    trace_count = (os.path.getsize(path) - 3600) // trace_size
    return {
        "dt": dt_us / 1_000_000.0,
        "dt_us": dt_us,
        "ns": ns,
        "sample_format": sample_format,
        "bytes_per_sample": bytes_per_sample,
        "trace_size": trace_size,
        "trace_count": trace_count,
    }


def segy_sample_dtype(sample_format):
    if sample_format == 2:
        return ">i4"
    if sample_format == 3:
        return ">i2"
    if sample_format == 5:
        return ">f4"
    if sample_format == 8:
        return "i1"
    raise ValueError(f"Unsupported SEG-Y sample format: {sample_format}")


def next_power_of_two(n):
    return 1 << int(np.ceil(np.log2(max(2, n))))


def bandpass_shape(freqs, f1, f2, f3, f4):
    """
    Trapezoid bandpass shape:
    0 below f1, linear ramp f1-f2, pass f2-f3, linear taper f3-f4.
    """
    shape = np.zeros_like(freqs, dtype=np.float64)

    ramp_up = (freqs >= f1) & (freqs < f2)
    pass_band = (freqs >= f2) & (freqs <= f3)
    ramp_down = (freqs > f3) & (freqs <= f4)

    if f2 > f1:
        shape[ramp_up] = (freqs[ramp_up] - f1) / (f2 - f1)
    shape[pass_band] = 1.0
    if f4 > f3:
        shape[ramp_down] = (f4 - freqs[ramp_down]) / (f4 - f3)

    return np.clip(shape, 0.0, 1.0)


def estimate_average_spectrum(section, dt, time_window=None, trace_window=None, nfft=None,
                              smooth_sigma=2.0):
    """
    Estimate an average amplitude spectrum from a 2D post-stack section.

    section shape is [time_samples, traces].
    """
    nt, nx = section.shape

    t0, t1 = time_window if time_window else (0, nt)
    x0, x1 = trace_window if trace_window else (0, nx)
    t0, t1 = max(0, t0), min(nt, t1)
    x0, x1 = max(0, x0), min(nx, x1)

    work = section[t0:t1, x0:x1].copy()
    if work.size == 0:
        raise ValueError("Selected time/trace window is empty.")

    work -= np.mean(work, axis=0, keepdims=True)
    work *= tukey(work.shape[0], alpha=0.1)[:, None]

    if nfft is None:
        nfft = next_power_of_two(work.shape[0] * 2)

    spec = np.fft.rfft(work, n=nfft, axis=0)
    amp = np.mean(np.abs(spec), axis=1)
    freqs = np.fft.rfftfreq(nfft, dt)

    amp[0] = 0.0
    if smooth_sigma and smooth_sigma > 0:
        amp = gaussian_filter1d(amp, sigma=smooth_sigma)

    if np.max(amp) > 0:
        amp = amp / np.max(amp)

    return freqs, amp, nfft


def estimate_average_spectrum_from_cube(cube, dt, time_window=None, trace_window=None,
                                        nfft=None, smooth_sigma=2.0):
    """
    Estimate average spectrum from data shaped [inline, time, crossline].
    """
    n_inline, nt, n_crossline = cube.shape
    t0, t1 = time_window if time_window else (0, nt)
    t0, t1 = max(0, t0), min(nt, t1)
    if t1 <= t0:
        raise ValueError("Selected time window is empty.")

    x0, x1 = trace_window if trace_window else (0, n_inline * n_crossline)
    x0, x1 = max(0, x0), min(n_inline * n_crossline, x1)
    if x1 <= x0:
        raise ValueError("Selected trace window is empty.")

    if nfft is None:
        nfft = next_power_of_two((t1 - t0) * 2)

    taper = tukey(t1 - t0, alpha=0.1)
    amp_sum = np.zeros(nfft // 2 + 1, dtype=np.float64)
    used_traces = 0

    for il_idx in range(n_inline):
        start = il_idx * n_crossline
        end = start + n_crossline
        use0 = max(x0, start)
        use1 = min(x1, end)
        if use1 <= use0:
            continue

        c0 = use0 - start
        c1 = use1 - start
        work = cube[il_idx, t0:t1, c0:c1].astype(np.float64, copy=True)
        work -= np.mean(work, axis=0, keepdims=True)
        work *= taper[:, None]
        amp_sum += np.sum(np.abs(np.fft.rfft(work, n=nfft, axis=0)), axis=1)
        used_traces += work.shape[1]

    if used_traces == 0:
        raise ValueError("No traces were selected from cube.")

    amp = amp_sum / used_traces
    freqs = np.fft.rfftfreq(nfft, dt)
    amp[0] = 0.0

    if smooth_sigma and smooth_sigma > 0:
        amp = gaussian_filter1d(amp, sigma=smooth_sigma)

    if np.max(amp) > 0:
        amp = amp / np.max(amp)

    return freqs, amp, nfft, used_traces


def estimate_average_spectrum_from_segy(path, dt=None, time_window=None, trace_window=None,
                                        nfft=None, smooth_sigma=2.0, trace_step=5,
                                        batch_size=4096):
    """
    Estimate average amplitude spectrum directly from SEG-Y traces in batches.
    This avoids loading a full 3D volume into memory.
    """
    info = read_segy_basic_info(path)
    if dt is None:
        dt = info["dt"]

    ns = info["ns"]
    trace_count = info["trace_count"]
    trace_size = info["trace_size"]
    bytes_per_sample = info["bytes_per_sample"]
    dtype = np.dtype(segy_sample_dtype(info["sample_format"]))

    t0, t1 = time_window if time_window else (0, ns)
    t0, t1 = max(0, t0), min(ns, t1)
    if t1 <= t0:
        raise ValueError("Selected time window is empty.")

    x0, x1 = trace_window if trace_window else (0, trace_count)
    x0, x1 = max(0, x0), min(trace_count, x1)
    if x1 <= x0:
        raise ValueError("Selected trace window is empty.")

    if nfft is None:
        nfft = next_power_of_two((t1 - t0) * 2)

    taper = tukey(t1 - t0, alpha=0.1)
    amp_sum = np.zeros(nfft // 2 + 1, dtype=np.float64)
    batch = []
    used_traces = 0

    step = max(1, trace_step)
    with open(path, "rb") as f:
        for trace_idx in range(x0, x1, step):
            data_offset = 3600 + trace_idx * trace_size + 240 + t0 * bytes_per_sample
            f.seek(data_offset)
            raw = f.read((t1 - t0) * bytes_per_sample)
            if len(raw) != (t1 - t0) * bytes_per_sample:
                continue

            trace = np.frombuffer(raw, dtype=dtype).astype(np.float64)
            trace -= np.mean(trace)
            trace *= taper
            batch.append(trace)

            if len(batch) >= batch_size:
                arr = np.asarray(batch).T
                amp_sum += np.sum(np.abs(np.fft.rfft(arr, n=nfft, axis=0)), axis=1)
                used_traces += arr.shape[1]
                batch.clear()

        if batch:
            arr = np.asarray(batch).T
            amp_sum += np.sum(np.abs(np.fft.rfft(arr, n=nfft, axis=0)), axis=1)
            used_traces += arr.shape[1]

    if used_traces == 0:
        raise ValueError("No traces were read from SEG-Y.")

    amp = amp_sum / used_traces
    freqs = np.fft.rfftfreq(nfft, dt)
    amp[0] = 0.0

    if smooth_sigma and smooth_sigma > 0:
        amp = gaussian_filter1d(amp, sigma=smooth_sigma)

    if np.max(amp) > 0:
        amp = amp / np.max(amp)

    return freqs, amp, nfft, info, used_traces


def zero_phase_wavelet_from_spectrum(freqs, amplitude, dt, nfft, wavelet_length_ms):
    """
    Build a centered zero-phase wavelet from an amplitude spectrum.
    """
    full = np.fft.irfft(amplitude.astype(np.float64), n=nfft)
    full = np.fft.fftshift(full)

    wavelet_len = int(round((wavelet_length_ms / 1000.0) / dt))
    if wavelet_len % 2 == 0:
        wavelet_len += 1
    wavelet_len = max(3, min(wavelet_len, nfft - 1 if (nfft - 1) % 2 else nfft - 2))

    center = nfft // 2
    half = wavelet_len // 2
    wavelet = full[center - half:center + half + 1].copy()
    wavelet -= np.mean(wavelet)
    wavelet *= tukey(wavelet.size, alpha=0.2)

    max_abs = np.max(np.abs(wavelet))
    if max_abs > 0:
        wavelet /= max_abs

    time_ms = (np.arange(wavelet.size) - half) * dt * 1000.0
    return time_ms, wavelet


def make_wavelet(freqs, average_amp, dt, nfft, band, wavelet_length_ms):
    shape = bandpass_shape(freqs, *band)
    shaped_amp = average_amp * shape
    if np.max(shaped_amp) > 0:
        shaped_amp = shaped_amp / np.max(shaped_amp)
    time_ms, wavelet = zero_phase_wavelet_from_spectrum(
        freqs, shaped_amp, dt, nfft, wavelet_length_ms
    )
    return time_ms, wavelet, shaped_amp, shape


def save_outputs(output_dir, freqs, avg_amp, narrow, wide, estimated):
    os.makedirs(output_dir, exist_ok=True)

    time_ms, estimated_wavelet, estimated_amp = estimated
    _, narrow_wavelet, narrow_amp, narrow_shape = narrow
    _, wide_wavelet, wide_amp, wide_shape = wide

    np.save(os.path.join(output_dir, "estimated_statistical_wavelet.npy"), estimated_wavelet)
    np.save(os.path.join(output_dir, "narrow_band_wavelet.npy"), narrow_wavelet)
    np.save(os.path.join(output_dir, "wide_band_wavelet.npy"), wide_wavelet)
    np.save(os.path.join(output_dir, "freqs.npy"), freqs)
    np.save(os.path.join(output_dir, "average_amplitude_spectrum.npy"), avg_amp)
    np.save(os.path.join(output_dir, "narrow_band_spectrum.npy"), narrow_amp)
    np.save(os.path.join(output_dir, "wide_band_spectrum.npy"), wide_amp)

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_ms, estimated_wavelet, "k-", lw=2, label="Estimated")
    axes[0].plot(time_ms, narrow_wavelet, "b-", lw=2, label="Narrow")
    axes[0].plot(time_ms, wide_wavelet, "r-", lw=2, label="Wide target")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(freqs, avg_amp, "k-", lw=2, label="Average spectrum")
    axes[1].plot(freqs, narrow_shape, "b--", lw=1.5, label="Narrow band shape")
    axes[1].plot(freqs, wide_shape, "r--", lw=1.5, label="Wide band shape")
    axes[1].plot(freqs, narrow_amp, "b-", lw=2, alpha=0.8, label="Narrow spectrum")
    axes[1].plot(freqs, wide_amp, "r-", lw=2, alpha=0.8, label="Wide target spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].set_xlim(0, min(freqs[-1], 150))
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wavelet_extraction_result.png"), dpi=300)
    plt.close(fig)


def parse_band(text):
    values = [float(x) for x in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Band must be f1,f2,f3,f4.")
    if values != sorted(values):
        raise argparse.ArgumentTypeError("Band frequencies must be increasing.")
    return tuple(values)


def parse_window(text):
    if text is None:
        return None
    values = [int(x) for x in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Window must be start,end.")
    return tuple(values)


def main():
    parser = argparse.ArgumentParser(
        description="Extract statistical narrow/wide wavelets from a 2D post-stack section."
    )
    parser.add_argument("--input", required=True, help="Input 2D section: .npy, .sgy or .segy")
    parser.add_argument("--dt", type=float, required=True, help="Sample interval in seconds")
    parser.add_argument("--output-dir", default="extracted_wavelets")
    parser.add_argument("--wavelet-length-ms", type=float, default=160.0)
    parser.add_argument("--narrow-band", type=parse_band, default=parse_band("3,6,25,35"))
    parser.add_argument("--wide-band", type=parse_band, default=parse_band("3,6,55,75"))
    parser.add_argument("--time-window", type=parse_window, default=None,
                        help="Optional sample window start,end, e.g. 500,1800")
    parser.add_argument("--trace-window", type=parse_window, default=None,
                        help="Optional trace window start,end, e.g. 100,900")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--trace-step", type=int, default=5,
                        help="For SEG-Y only, use every Nth trace while estimating spectrum.")
    parser.add_argument("--use-segy-reader", action="store_true",
                        help="Read SEG-Y with segy_reader.read_segy instead of streaming reader.")
    parser.add_argument("--shotnum", type=int, default=0,
                        help="shotnum passed to segy_reader.read_segy, e.g. 651 for F3.")
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    segy_info = None
    used_traces = None
    cube_shape = None

    if ext in {".sgy", ".segy"} and args.use_segy_reader:
        from segy_reader import read_segy

        cube = read_segy(args.input, shotnum=args.shotnum)
        cube_shape = cube.shape
        freqs, avg_amp, nfft, used_traces = estimate_average_spectrum_from_cube(
            cube,
            dt=args.dt,
            time_window=args.time_window,
            trace_window=args.trace_window,
            smooth_sigma=args.smooth_sigma,
        )
    elif ext in {".sgy", ".segy"}:
        freqs, avg_amp, nfft, segy_info, used_traces = estimate_average_spectrum_from_segy(
            args.input,
            dt=args.dt,
            time_window=args.time_window,
            trace_window=args.trace_window,
            smooth_sigma=args.smooth_sigma,
            trace_step=args.trace_step,
        )
    else:
        section = load_seismic_section(args.input)
        freqs, avg_amp, nfft = estimate_average_spectrum(
            section,
            dt=args.dt,
            time_window=args.time_window,
            trace_window=args.trace_window,
            smooth_sigma=args.smooth_sigma,
        )

    estimated_time, estimated_wavelet = zero_phase_wavelet_from_spectrum(
        freqs, avg_amp, args.dt, nfft, args.wavelet_length_ms
    )
    narrow = make_wavelet(freqs, avg_amp, args.dt, nfft, args.narrow_band,
                          args.wavelet_length_ms)
    wide = make_wavelet(freqs, avg_amp, args.dt, nfft, args.wide_band,
                        args.wavelet_length_ms)

    save_outputs(
        args.output_dir,
        freqs,
        avg_amp,
        narrow=narrow,
        wide=wide,
        estimated=(estimated_time, estimated_wavelet, avg_amp),
    )

    if cube_shape:
        print(f"Input cube from segy_reader.read_segy: {cube_shape}")
        print(f"Used traces for spectrum estimation: {used_traces}")
    elif segy_info:
        print(
            "Input SEG-Y: "
            f"{segy_info['trace_count']} traces, {segy_info['ns']} samples, "
            f"dt={segy_info['dt']} s, format={segy_info['sample_format']}"
        )
        print(f"Used traces for spectrum estimation: {used_traces}")
    else:
        print(f"Input section shape: {section.shape}")
    print(f"Saved wavelets and figures to: {args.output_dir}")
    print(f"Narrow band: {args.narrow_band} Hz")
    print(f"Wide target band: {args.wide_band} Hz")
    print("Note: the wide wavelet is a designed target wavelet, not true missing bandwidth.")


if __name__ == "__main__":
    main()
