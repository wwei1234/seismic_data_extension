"""
地震数据频谱分析工具
====================
支持一维（单道）和二维（剖面）地震数据的振幅谱分析与可视化。
支持 .npy 和 .sgy/.segy 格式输入。

用法:
    # 直接运行（使用下方的参数设置）
    python 频谱分析.py

    # 命令行覆盖参数
    python 频谱分析.py -i seismic.npy -d 0.004
    python 频谱分析.py -i trace.npy -d 0.001 --1d
    python 频谱分析.py -i a.npy b.npy --labels "窄频" "宽频" -d 0.004
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# ==================== 参数设置 ====================

# 输入文件路径列表（.npy / .sgy / .segy）
INPUT_FILES = [
    # r"prediction_results\noisy_normalized_input.npy",
    # r"prediction_results\network_output_normalized.npy",
    # r"prediction_results\target_normalized.npy",
    r"F3_Demo_2023\Rawdata\Seismic_data.sgy"
]

# 每条曲线的图例标签（数量需与 INPUT_FILES 一致，为空则使用文件名）
LABELS = ["Original Input", "Network Prediction", "Target (Wide Band)"]

# 采样间隔 (s)
DT = 0.004

# 输出目录
OUTPUT_DIR = "spectrum_analysis"

# 频率显示上限 (Hz)
FREQ_MAX = 150

# 图标题
TITLE = "Amplitude Spectrum Comparison"

# 输出文件名（不含扩展名）
FILENAME = "spectrum_comparison"

# SEG-Y 选项
PRESTACK = True              # 是否按叠前炮集方式读取
SHOTNUM = 651                   # 炮数（0 = 自动推断）
IS_1D = False                 # 强制按一维单道处理


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_npy(path):
    """加载 .npy 文件，返回 numpy 数组。"""
    data = np.load(path)
    if data.ndim not in (1, 2):
        raise ValueError(f"期望 1D（单道）或 2D（剖面）数据，实际维度: {data.ndim}")
    return data.astype(np.float64)


def load_segy_section(path):
    """加载 SEG-Y 格式的二维叠后剖面，返回 (time_samples, traces) 数组。"""
    import segyio
    with segyio.open(path, "r", ignore_geometry=True) as f:
        data = np.asarray([np.copy(trace) for trace in f.trace]).T
    if data.ndim != 2:
        raise ValueError(f"SEG-Y 数据期望为 2D 剖面，实际维度: {data.ndim}")
    return data.astype(np.float64)


def load_segy_prestack(path, shotnum=0):
    """加载 SEG-Y 叠前炮集，返回 (shot_num, time, traces_per_shot) 数组。"""
    from segy_reader import read_segy
    return read_segy(path, shotnum=shotnum)


def load_data(path, prestack=False, shotnum=0):
    """自动识别格式并加载数据。"""
    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path)

    if ext == ".npy":
        data = load_npy(path)
    elif ext in (".sgy", ".segy"):
        if prestack:
            data = load_segy_prestack(path, shotnum=shotnum)
        else:
            data = load_segy_section(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}。仅支持 .npy / .sgy / .segy")

    return data, basename


# ---------------------------------------------------------------------------
# 频谱计算
# ---------------------------------------------------------------------------

def compute_amplitude_spectrum(data, dt):
    """
    计算振幅谱。
    1D 输入 → 单道频谱；2D 输入 → 各道平均频谱。
    """
    n_samples = data.shape[0]

    if data.ndim == 1:
        trace = data - np.mean(data)
        spec = np.abs(fft(trace))
    elif data.ndim == 2:
        n_traces = data.shape[1]
        spec = np.zeros(n_samples)
        for i in range(n_traces):
            trace = data[:, i] - np.mean(data[:, i])
            spec += np.abs(fft(trace))
        spec /= n_traces
    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}")

    freqs = fftfreq(n_samples, dt)
    positive_freqs = freqs[:n_samples // 2]
    positive_amp = spec[:n_samples // 2]
    positive_amp[0] = 0.0  # 压制直流分量

    return positive_freqs, positive_amp


def normalize_spectrum(amp):
    """将振幅谱归一化到 [0, 1] 区间。"""
    max_val = np.max(amp)
    return amp / max_val if max_val > 0 else amp


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_spectrum(freqs_list, amp_list, labels, dt, output_dir, freq_max=150,
                  title="Amplitude Spectrum", filename="spectrum"):
    """绘制并保存频谱对比图。"""
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(freqs_list)))
    for freqs, amp, label, color in zip(freqs_list, amp_list, labels, colors):
        ax.plot(freqs, amp, color=color, linewidth=2.5, alpha=0.85, label=label)
        ax.fill_between(freqs, 0, amp, color=color, alpha=0.08)

        # 标注主频
        peak_idx = np.argmax(amp)
        peak_freq = freqs[peak_idx]
        ax.axvline(x=peak_freq, color=color, linestyle=":", linewidth=1.5, alpha=0.5)
        ax.annotate(
            f"{peak_freq:.1f} Hz",
            xy=(peak_freq, amp[peak_idx]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=11, color=color, fontweight="bold", ha="center",
        )

    ax.set_xlabel("Frequency (Hz)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Normalized Amplitude", fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=15)
    ax.set_xlim([0, freq_max])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    ax.legend(fontsize=14, loc="upper right", frameon=True, shadow=True)
    ax.tick_params(labelsize=14)

    # 采样信息
    textstr = f"dt = {dt*1000:.1f} ms  ({dt:.3f} s)"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    plt.tight_layout()

    for ext in ["png", "eps"]:
        out_path = os.path.join(output_dir, f"{filename}.{ext}")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"  已保存: {out_path}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    # 命令行参数（默认值引用顶部常量，命令行可覆盖）
    parser = argparse.ArgumentParser(
        description="地震数据频谱分析 —— 支持 1D/2D 数据，.npy/.sgy 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", nargs="+", default=INPUT_FILES,
                        help="输入文件路径（默认使用顶部 INPUT_FILES）")
    parser.add_argument("-d", "--dt", type=float, default=DT,
                        help=f"采样间隔，单位秒（默认: {DT}）")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR,
                        help=f"输出目录（默认: {OUTPUT_DIR}）")
    parser.add_argument("--labels", nargs="+", default=LABELS,
                        help="图例标签（默认使用顶部 LABELS）")
    parser.add_argument("--freq-max", type=float, default=FREQ_MAX,
                        help=f"频率显示上限 Hz（默认: {FREQ_MAX}）")
    parser.add_argument("--title", default=TITLE,
                        help=f"图标题（默认: {TITLE}）")
    parser.add_argument("--filename", default=FILENAME,
                        help=f"输出文件名（默认: {FILENAME}）")
    parser.add_argument("--prestack", action="store_true", default=PRESTACK,
                        help="按叠前炮集读取 SEG-Y")
    parser.add_argument("--shotnum", type=int, default=SHOTNUM,
                        help="炮数（prestack 模式，0=自动推断）")
    parser.add_argument("--1d", dest="is_1d", action="store_true", default=IS_1D,
                        help="强制按一维单道处理")
    args = parser.parse_args()

    # --- 标签校验 ---
    labels = args.labels
    if labels and len(labels) != len(args.input):
        print(f"警告: labels 数量 ({len(labels)}) 与输入文件数 ({len(args.input)}) 不匹配，将使用文件名。")
        labels = None

    # --- 加载数据 ---
    print("=" * 60)
    print("地震数据频谱分析")
    print("=" * 60)
    print(f"采样间隔 dt = {args.dt:.4f} s  ({args.dt * 1000:.1f} ms)")
    print(f"输入文件数: {len(args.input)}")
    print()

    all_freqs = []
    all_amp = []

    for idx, path in enumerate(args.input):
        print(f"[{idx + 1}/{len(args.input)}] 加载: {path}")
        data, basename = load_data(path, prestack=args.prestack, shotnum=args.shotnum)
        print(f"      数据形状: {data.shape}, 维度: {data.ndim}")

        if args.prestack and data.ndim == 3:
            print(f"      叠前数据: 共 {data.shape[0]} 炮，选取第 0 炮进行分析")
            data = data[0, :, :]

        if args.is_1d and data.ndim == 2:
            print("      一维模式: 取第一道进行分析")
            data = data[:, 0]

        freqs, amp = compute_amplitude_spectrum(data, args.dt)
        amp_norm = normalize_spectrum(amp)

        all_freqs.append(freqs)
        all_amp.append(amp_norm)

        if labels is None:
            labels_list = [basename for _ in args.input]
        else:
            labels_list = labels

        peak_idx = np.argmax(amp_norm)
        print(f"      主频: {freqs[peak_idx]:.1f} Hz")
        print()

    # --- 绘制 ---
    print("绘制频谱图...")
    plot_spectrum(
        freqs_list=all_freqs,
        amp_list=all_amp,
        labels=labels_list,
        dt=args.dt,
        output_dir=args.output_dir,
        freq_max=args.freq_max,
        title=args.title,
        filename=args.filename,
    )

    print("\n完成！")


if __name__ == "__main__":
    main()
