from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wilcoxon


DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = DATA_ANALYSIS_ROOT / "figures"

AVG_BLOCKS = [
    ("Jamie11", "10-04-26", "R6"),
    ("Jamie11", "28-04-26", "R2"),
]
HIGH_FS_BLOCK = ("Jamie11", "29-04-26", "R2")

SAVE_FIGURE = True
SHOW_FIGURE = True
FIG_DPI = 300
OUT_NAME = "fig_last_10hz_jamie11.png"

GEVI_DISPLAY_SCALE = 100.0
SPECTROGRAM_FMAX_HZ = 250.0
GEVI_SPECTROGRAM_REL_DB_RANGE = (-8.0, 8.0)
MPTA_FULL_XLIM_S = (-0.08, 0.30)
MPTA_ZOOM_XLIM_S = (-0.005, 0.030)
VM_TRANSIENT_WINDOW_S = (0.0, 0.2)
VM_SUSTAINED_WINDOW_S = (5.0, 10.0)
HILBERT_BASELINE_WINDOW_S = (-5.0, -0.5)
HILBERT_STIM_WINDOW_S = (0.0, 10.0)
PEAK_DETECTION_WINDOW_S = (0.0, 0.030)
PEAK_BASELINE_WINDOW_S = (-0.080, -0.020)
PEAK_THRESHOLD_SD = 1.0
PEAK_TOP_SAMPLES = 2


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def summary_path(mouse: str, date: str, block: str) -> Path:
    return DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block / f"{block}_summary.pkl"


def load_summary(spec: tuple[str, str, str]) -> dict[str, Any]:
    return load_pickle(summary_path(*spec))


def gevi_display(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * GEVI_DISPLAY_SCALE


def build_common_axis_1d(arrays: list[np.ndarray]) -> np.ndarray | None:
    valid = [np.asarray(a, float) for a in arrays if a is not None and len(a) >= 2]
    if not valid:
        return None
    lo = max(float(np.nanmin(a)) for a in valid)
    hi = min(float(np.nanmax(a)) for a in valid)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    steps = [float(np.nanmedian(np.diff(a))) for a in valid if len(a) >= 2]
    step = float(np.nanmedian(steps))
    if not np.isfinite(step) or step <= 0:
        return None
    axis = np.arange(lo, hi + 0.5 * step, step, dtype=float)
    return axis if len(axis) >= 2 else None


def interpolate_curve(x_ref: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    tol = max(1e-12, 0.25 * float(np.nanmedian(np.diff(x))))
    keep = (x_ref >= x[0] - tol) & (x_ref <= x[-1] + tol)
    out = np.full_like(x_ref, np.nan, dtype=float)
    out[keep] = np.interp(np.clip(x_ref[keep], x[0], x[-1]), x, y)
    return out


def nanmean_stack(stack: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def nanstd_stack(stack: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanstd(stack, axis=0)


def get_full_trace(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    proc = summary.get("summary", {}).get("processed_notched", {})
    sec = proc.get("stim", {})
    t = np.asarray(sec.get("t_common", []), dtype=float)
    y = np.asarray(sec.get("F_notched_mean", []), dtype=float)
    if len(t) < 2 or y.shape != t.shape:
        return None
    return t, y


def get_mpta(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sec = summary.get("summary", {}).get("train_pta", {})
    if not sec.get("available", False):
        return None
    display = sec.get("display", {})
    x = np.asarray(display.get("t_rel_s", sec.get("t_rel_s", [])), dtype=float)
    mean = np.asarray(display.get("mean", sec.get("mean_across_trials", [])), dtype=float)
    sd = np.asarray(display.get("sd", sec.get("sd_across_trials", [])), dtype=float)
    if len(x) < 2 or mean.shape != x.shape:
        return None
    if sd.shape != x.shape:
        sd = np.full_like(mean, np.nan)
    return x, mean, sd


def get_hilbert(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sec = summary.get("summary", {}).get("train_pta", {}).get("signal_hilbert", {})
    x = np.asarray(sec.get("time_s_display", sec.get("time_s_full", [])), dtype=float)
    mean = np.asarray(sec.get("relative_mean_display", sec.get("relative_mean_full", [])), dtype=float)
    sd = np.asarray(sec.get("relative_sd_display", sec.get("relative_sd_full", [])), dtype=float)
    if len(x) < 2 or mean.shape != x.shape:
        return None
    if sd.shape != x.shape:
        sd = np.full_like(mean, np.nan)
    return x, mean, sd


def average_curves(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x_ref = build_common_axis_1d([x for x, _ in items])
    if x_ref is None:
        return None
    stack = np.vstack([interpolate_curve(x_ref, x, y) for x, y in items])
    return x_ref, nanmean_stack(stack), nanstd_stack(stack)


def average_curve_with_sd(items: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x_ref = build_common_axis_1d([x for x, _, _ in items])
    if x_ref is None:
        return None
    stack = np.vstack([interpolate_curve(x_ref, x, y) for x, y, _ in items])
    return x_ref, nanmean_stack(stack), nanstd_stack(stack)


def average_spectrogram(summaries: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    items = []
    for summary in summaries:
        sec = summary.get("summary", {}).get("train_pta", {}).get("spectrogram", {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        z = sec.get("relative_linear_mean")
        if z is None and sec.get("relative_db_mean") is not None:
            z = 10.0 ** (np.asarray(sec["relative_db_mean"], dtype=float) / 10.0)
        z = np.asarray(z, dtype=float)
        if len(t) >= 2 and len(f) >= 2 and z.shape == (len(f), len(t)):
            items.append((t, f, z))
    if not items:
        return None
    t_ref = build_common_axis_1d([item[0] for item in items])
    f_ref = build_common_axis_1d([item[1] for item in items])
    if t_ref is None or f_ref is None:
        return None
    f_ref = f_ref[f_ref <= SPECTROGRAM_FMAX_HZ]
    ff, tt = np.meshgrid(f_ref, t_ref, indexing="ij")
    pts = np.column_stack([ff.ravel(), tt.ravel()])
    stack = []
    for t, f, z in items:
        interp = RegularGridInterpolator((f, t), z, bounds_error=False, fill_value=np.nan)
        stack.append(interp(pts).reshape(len(f_ref), len(t_ref)))
    mean_linear = nanmean_stack(np.stack(stack, axis=0))
    z_db = 10.0 * np.log10(np.maximum(mean_linear, 1e-12))
    return t_ref, f_ref, z_db


def trial_vm_values(summary: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    baseline_vals: list[float] = []
    early: list[float] = []
    late: list[float] = []
    trials = summary.get("trials", {}).get("processed_notched", {})
    for trial in trials.values():
        t = np.asarray(trial.get("t", []), dtype=float)
        y = np.asarray(trial.get("F_notched", []), dtype=float)
        if len(t) < 2 or y.shape != t.shape:
            continue
        base_vals = y[t < 0]
        baseline = float(np.nanmedian(base_vals)) if len(base_vals) else float(np.nanmedian(y))
        e = (t >= VM_TRANSIENT_WINDOW_S[0]) & (t <= VM_TRANSIENT_WINDOW_S[1])
        l = (t >= VM_SUSTAINED_WINDOW_S[0]) & (t <= VM_SUSTAINED_WINDOW_S[1])
        if np.any(e) and np.any(l):
            baseline_vals.append(float(baseline * GEVI_DISPLAY_SCALE))
            early.append(float(np.nanmedian(y[e]) * GEVI_DISPLAY_SCALE))
            late.append(float(np.nanmedian(y[l]) * GEVI_DISPLAY_SCALE))
    return baseline_vals, early, late


def trial_hilbert_values(summary: dict[str, Any]) -> tuple[list[float], list[float]]:
    baseline: list[float] = []
    stim: list[float] = []
    trials = summary.get("trials", {}).get("train_pta", {})
    for trial in trials.values():
        sec = trial.get("signal_hilbert", {})
        t = np.asarray(sec.get("time_s", sec.get("time_s_full", [])), dtype=float)
        rel = np.asarray(sec.get("relative", sec.get("relative_full", [])), dtype=float)
        if len(t) < 2 or rel.shape != t.shape:
            amp = np.asarray(sec.get("amplitude", sec.get("amplitude_full", [])), dtype=float)
            if len(t) < 2 or amp.shape != t.shape:
                continue
            base_mask = (t >= HILBERT_BASELINE_WINDOW_S[0]) & (t <= HILBERT_BASELINE_WINDOW_S[1])
            base = float(np.nanmedian(amp[base_mask])) if np.any(base_mask) else np.nan
            rel = amp / base if np.isfinite(base) and base > 0 else np.full_like(amp, np.nan)
        base_mask = (t >= HILBERT_BASELINE_WINDOW_S[0]) & (t <= HILBERT_BASELINE_WINDOW_S[1])
        stim_mask = (t >= HILBERT_STIM_WINDOW_S[0]) & (t <= HILBERT_STIM_WINDOW_S[1])
        if np.any(base_mask) and np.any(stim_mask):
            baseline.append(float(np.nanmedian(rel[base_mask])))
            stim.append(float(np.nanmedian(rel[stim_mask])))
    return baseline, stim


def paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float, int]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    arr_a = arr_a[:n]
    arr_b = arr_b[:n]
    keep = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[keep]
    arr_b = arr_b[keep]
    if len(arr_a) < 2:
        return np.nan, np.nan, int(len(arr_a))
    try:
        stat, p = wilcoxon(arr_a, arr_b)
    except ValueError:
        return np.nan, np.nan, int(len(arr_a))
    return float(stat), float(p), int(len(arr_a))


def detect_peak_events(t: np.ndarray, y: np.ndarray, n_events: int = 3) -> list[dict[str, float]]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(t), len(y))
    t = t[:n]
    y = y[:n]
    base_mask = (t >= PEAK_BASELINE_WINDOW_S[0]) & (t < PEAK_BASELINE_WINDOW_S[1]) & np.isfinite(y)
    if np.sum(base_mask) < 2:
        base_mask = (t < 0) & np.isfinite(y)
    baseline = float(np.nanmean(y[base_mask])) if np.any(base_mask) else 0.0
    sd = float(np.nanstd(y[base_mask], ddof=1)) if np.sum(base_mask) >= 2 else 0.0
    threshold = baseline + PEAK_THRESHOLD_SD * sd
    search = (t >= PEAK_DETECTION_WINDOW_S[0]) & (t <= PEAK_DETECTION_WINDOW_S[1]) & np.isfinite(y)
    above = search & (y > threshold)
    idx = np.flatnonzero(above)
    events: list[np.ndarray] = []
    if len(idx):
        start = 0
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                events.append(idx[start:i])
                start = i
        events.append(idx[start:])

    out: list[dict[str, float]] = []
    for event in events[:n_events]:
        if len(event) == 0:
            continue
        n_top = max(1, min(PEAK_TOP_SAMPLES, len(event)))
        top = event[np.argsort(y[event])[-n_top:]]
        out.append(
            {
                "latency_ms": float(np.nanmean(t[top]) * 1000.0),
                "amplitude": float(np.nanmean(y[top])),
                "start_ms": float(t[event[0]] * 1000.0),
                "end_ms": float(t[event[-1]] * 1000.0),
                "threshold": threshold,
            }
        )
    return out


def plot_line_with_sd(ax, x, mean, sd, color="tab:blue", xlim=None):
    ax.plot(x, mean, color=color, lw=1.8)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18)
    ax.axvline(0, color="red", ls="--", lw=1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))


def plot_vm_box(ax, early: list[float], late: list[float]):
    ax.axhline(0, color="0.35", lw=0.8)
    data = [early, late]
    ax.boxplot(data, positions=[1, 2], widths=0.45, showfliers=False)
    rng = np.random.default_rng(2)
    for pos, vals, color in [(1, early, "tab:blue"), (2, late, "tab:red")]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        x = pos + rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(x, vals, s=18, color=color, alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["trans", "sust"])
    ax.set_ylabel("dF/F")
    ax.set_title("Vm response")


def add_panel_letters(fig, axes):
    for i, ax in enumerate(axes):
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.035 * bbox.width, bbox.y1 + 0.035 * bbox.height, chr(97 + i), fontsize=13, fontweight="bold")


def main() -> None:
    summaries = [load_summary(spec) for spec in AVG_BLOCKS]
    high_summary = load_summary(HIGH_FS_BLOCK)

    full_items = [get_full_trace(s) for s in summaries]
    full_items = [item for item in full_items if item is not None]
    gevi = average_curves(full_items)

    mpta_items = [get_mpta(s) for s in summaries]
    mpta_items = [item for item in mpta_items if item is not None]
    mpta = average_curve_with_sd(mpta_items)
    high_mpta = get_mpta(high_summary)

    hilbert_items = [get_hilbert(s) for s in summaries]
    hilbert_items = [item for item in hilbert_items if item is not None]
    hilbert = average_curve_with_sd(hilbert_items)

    spec = average_spectrogram(summaries)

    vm_base: list[float] = []
    vm_early_abs: list[float] = []
    vm_late_abs: list[float] = []
    hilb_base: list[float] = []
    hilb_stim: list[float] = []
    for summary in summaries:
        b, e, l = trial_vm_values(summary)
        vm_base.extend(b)
        vm_early_abs.extend(e)
        vm_late_abs.extend(l)
        b, s = trial_hilbert_values(summary)
        hilb_base.extend(b)
        hilb_stim.extend(s)

    vm_early_rel = [e - b for b, e in zip(vm_base, vm_early_abs)]
    vm_late_rel = [l - b for b, l in zip(vm_base, vm_late_abs)]
    vm_stat, vm_p, vm_n = paired_wilcoxon(vm_early_abs, vm_late_abs)
    vm_early_base_stat, vm_early_base_p, vm_early_base_n = paired_wilcoxon(vm_early_abs, vm_base)
    vm_late_base_stat, vm_late_base_p, vm_late_base_n = paired_wilcoxon(vm_late_abs, vm_base)
    h_stat, h_p, h_n = paired_wilcoxon(hilb_base, hilb_stim)

    if mpta is None or high_mpta is None:
        raise RuntimeError("Missing MPTA data")
    mpta_peaks = detect_peak_events(mpta[0], gevi_display(mpta[1]))
    high_peaks = detect_peak_events(high_mpta[0], gevi_display(high_mpta[1]))

    fig, axes = plt.subplots(2, 3, figsize=(4.8 * 3, 2.6 * 2), constrained_layout=False)
    axes = axes.ravel()

    if gevi is not None:
        x, mean, sd = gevi
        plot_line_with_sd(axes[0], x, gevi_display(mean), gevi_display(sd), xlim=(-5, 15))
        axes[0].axvline(10, color="tab:orange", ls="--", lw=1)
        axes[0].set_title("GEVI signal")
        axes[0].set_ylabel("dF/F")
        axes[0].set_xlabel("time from stim onset (s)")

    if spec is not None:
        t, f, z = spec
        norm = TwoSlopeNorm(vmin=GEVI_SPECTROGRAM_REL_DB_RANGE[0], vcenter=0.0, vmax=GEVI_SPECTROGRAM_REL_DB_RANGE[1])
        im = axes[1].imshow(
            z,
            aspect="auto",
            origin="lower",
            extent=[float(np.nanmin(t)), float(np.nanmax(t)), float(np.nanmin(f)), float(np.nanmax(f))],
            cmap="RdBu_r",
            norm=norm,
            interpolation="bilinear",
        )
        axes[1].set_title("GEVI spectrogram")
        axes[1].set_ylabel("frequency (Hz)")
        axes[1].set_xlabel("time from stim onset (s)")
        cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
        cb.set_label("relative power (dB)")

    if hilbert is not None:
        x, mean, sd = hilbert
        plot_line_with_sd(axes[2], x, mean, sd, xlim=(-5, 15))
        axes[2].axvline(10, color="tab:orange", ls="--", lw=1)
        axes[2].set_title("DBS frequency amplitude")
        axes[2].set_ylabel("amp / baseline")
        axes[2].set_xlabel("time from stim onset (s)")

    x, mean, sd = mpta
    y = gevi_display(mean)
    ysd = gevi_display(sd)
    plot_line_with_sd(axes[3], x, y, ysd, color="black", xlim=MPTA_FULL_XLIM_S)
    axes[3].axvline(0.1, color="tab:orange", ls="--", lw=1)
    axes[3].set_title("Pulse-triggered response")
    axes[3].set_ylabel("dF/F")
    axes[3].set_xlabel("time from pulse (s)")

    plot_line_with_sd(axes[4], x, y, ysd, color="black", xlim=MPTA_ZOOM_XLIM_S)
    axes[4].set_title("Pulse-triggered response zoom")
    axes[4].set_ylabel("dF/F")
    axes[4].set_xlabel("time from pulse (s)")
    for peak in mpta_peaks:
        axes[4].plot(peak["latency_ms"] / 1000.0, peak["amplitude"], "o", color="crimson", ms=4)

    hx, hmean, hsd = high_mpta
    hy = gevi_display(hmean)
    hysd = gevi_display(hsd)
    plot_line_with_sd(axes[5], hx, hy, hysd, color="black", xlim=MPTA_ZOOM_XLIM_S)
    axes[5].set_title("2 kHz pulse-triggered zoom")
    axes[5].set_ylabel("dF/F")
    axes[5].set_xlabel("time from pulse (s)")
    for peak in high_peaks:
        axes[5].plot(peak["latency_ms"] / 1000.0, peak["amplitude"], "o", color="crimson", ms=4)

    fig.tight_layout(pad=1.2, w_pad=1.5, h_pad=1.7)
    add_panel_letters(fig, axes[:6])

    print("\nIncluded averaged blocks:")
    for spec in AVG_BLOCKS:
        print(f"  {spec[0]} | {spec[1]} | {spec[2]}")
    print(f"2 kHz zoom block: {HIGH_FS_BLOCK[0]} | {HIGH_FS_BLOCK[1]} | {HIGH_FS_BLOCK[2]}")
    print(f"\nVm transient ({VM_TRANSIENT_WINDOW_S[0]}-{VM_TRANSIENT_WINDOW_S[1]} s) vs sustained ({VM_SUSTAINED_WINDOW_S[0]}-{VM_SUSTAINED_WINDOW_S[1]} s):")
    print(f"  transient median={np.nanmedian(vm_early_rel):.4g}, sustained median={np.nanmedian(vm_late_rel):.4g}, Wilcoxon W={vm_stat:.4g}, p={vm_p:.4g}, n={vm_n}")
    print("\nVm baseline comparisons:")
    print(f"  transient vs baseline: median change={np.nanmedian(vm_early_rel):.4g}, Wilcoxon W={vm_early_base_stat:.4g}, p={vm_early_base_p:.4g}, n={vm_early_base_n}")
    print(f"  sustained vs baseline: median change={np.nanmedian(vm_late_rel):.4g}, Wilcoxon W={vm_late_base_stat:.4g}, p={vm_late_base_p:.4g}, n={vm_late_base_n}")
    print("\nHilbert baseline vs stim:")
    print(f"  baseline median={np.nanmedian(hilb_base):.4g}, stim median={np.nanmedian(hilb_stim):.4g}, Wilcoxon W={h_stat:.4g}, p={h_p:.4g}, n={h_n}")
    print("\nPeak events, 1 kHz average:")
    for i, peak in enumerate(mpta_peaks, start=1):
        print(f"  peak {i}: latency={peak['latency_ms']:.3g} ms, amp={peak['amplitude']:.4g}, event={peak['start_ms']:.3g}-{peak['end_ms']:.3g} ms")
    print("\nPeak events, 2 kHz R2:")
    for i, peak in enumerate(high_peaks, start=1):
        print(f"  peak {i}: latency={peak['latency_ms']:.3g} ms, amp={peak['amplitude']:.4g}, event={peak['start_ms']:.3g}-{peak['end_ms']:.3g} ms")

    if SAVE_FIGURE:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / OUT_NAME
        fig.savefig(out, dpi=FIG_DPI)
        print(f"\nSaved figure: {out}")
    if SHOW_FIGURE:
        try:
            plt.show()
        finally:
            plt.close(fig)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
