from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

from config import DATA_ANALYSIS_ROOT


# -------------------------
# SOURCE KERNEL
# -------------------------
# One low-frequency source trial provides the neural response kernel.
# That same fixed kernel is then driven by a synthetic DBS pulse train.

KERNEL_MOUSE = "Jamie11"
KERNEL_DATE = "10-04-26"
KERNEL_BLOCK = "R6"
KERNEL_MODE = "trial"          # "trial" or "block_mean"
KERNEL_TRIAL = "R6_5"          # used when KERNEL_MODE == "trial"


# -------------------------
# SYNTHETIC DBS INPUT
# -------------------------
SYNTH_FREQUENCY_HZ = 40.0
SYNTH_DURATION_S = 10.0
SYNTH_ONSET_S = 0.0
SYNTH_PERIOD_FRACTION = 0.95   # used for synthetic sPTA / mPTA display windows

SYNTH_PULSE_WIDTH_US = 100.0
SYNTH_AMPLITUDE_SCALE = 1.0


# -------------------------
# KERNEL OPTIONS
# -------------------------
KERNEL_END_SEC = None          # None = use 95% of the source pulse period
KERNEL_PERIOD_FRACTION = 0.95
BASELINE_STAT = "median"       # "median" or "mean"


# -------------------------
# SPECTROGRAM
# -------------------------
SPECTROGRAM_WINDOW_SEC = 0.528
SPECTROGRAM_OVERLAP_FRACTION = 0.95
SPECTROGRAM_FMAX_HZ = 250.0
SPECTROGRAM_REL_DB_RANGE = (-8.0, 8.0)


# -------------------------
# PLOTTING
# -------------------------
SHOW_PLOT = True
SAVE_FIGURE = False
OUTPUT_DIR = DATA_ANALYSIS_ROOT / "tables"

PLOT_FULL_TRACE = True
PLOT_SINGLE_PTA = True
PLOT_MEAN_PTA = True
PLOT_SPECTROGRAM = True
PLOT_INPUT_TRAIN = True
PLOT_KERNEL = True
PLOT_SOURCE_REFERENCE = False
LOCAL_PTA_BASELINE_CORRECTION = False


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summary_path(mouse: str, date: str, block: str) -> Path:
    return DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block / f"{block}_summary.pkl"


def safe_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def baseline_value(values: np.ndarray, mode: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    if str(mode).lower() == "mean":
        return float(np.nanmean(values))
    return float(np.nanmedian(values))


def get_trial(summary: dict, trial_name: str) -> dict:
    trial = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    if not trial:
        raise KeyError(f"Trial not found: {trial_name}")
    return trial


def get_source_trace(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trial = get_trial(summary, KERNEL_TRIAL)
    t_full = np.asarray(trial.get("t_full_s", []), dtype=float)
    y_full = np.asarray(trial.get("signal_full", []), dtype=float)
    pulse_times = np.asarray(trial.get("pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(t_full) == 0 or y_full.shape != t_full.shape or len(pulse_times) < 2:
        raise ValueError("Source trial is missing full trace or pulse times.")
    return t_full, y_full, pulse_times


def choose_kernel_section(summary: dict) -> tuple[np.ndarray, np.ndarray, float]:
    if KERNEL_MODE == "trial":
        sec = get_trial(summary, KERNEL_TRIAL)
        t_rel = np.asarray(sec.get("t_rel_s", []), dtype=float)
        kernel = np.asarray(sec.get("pta_mean", []), dtype=float)
        pulse_times = np.asarray(sec.get("pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        ipi_s = float(np.nanmedian(np.diff(pulse_times))) if len(pulse_times) >= 2 else np.nan
        return t_rel, kernel, ipi_s

    sec = summary.get("summary", {}).get("train_pta", {})
    t_rel = np.asarray(sec.get("t_rel_s", []), dtype=float)
    kernel = np.asarray(sec.get("mean_across_trials", []), dtype=float)
    second = safe_float(sec.get("display", {}).get("second_pulse_rel_s_mean"))
    return t_rel, kernel, second


def prepare_kernel(summary: dict, dt_source: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    t_rel, kernel_raw, ipi_s = choose_kernel_section(summary)
    if len(t_rel) == 0 or kernel_raw.shape != t_rel.shape:
        raise ValueError("Kernel trace missing or malformed.")

    pre_mask = t_rel < 0
    kernel_bc = kernel_raw - baseline_value(kernel_raw[pre_mask], BASELINE_STAT)

    if not np.isfinite(ipi_s) or ipi_s <= 0:
        raise ValueError("Could not determine source pulse period for kernel.")

    kernel_end_s = float(KERNEL_END_SEC) if KERNEL_END_SEC is not None else float(KERNEL_PERIOD_FRACTION) * float(ipi_s)
    keep = (t_rel >= 0.0) & (t_rel <= kernel_end_s)
    if np.count_nonzero(keep) < 3:
        raise ValueError("Kernel window is too short.")

    t_kernel = np.asarray(t_rel[keep], dtype=float)
    y_kernel = np.asarray(kernel_bc[keep], dtype=float)

    # Resample to the source full-trace grid for convolution.
    t_kernel_rs = np.arange(0.0, float(t_kernel[-1]) + 0.5 * dt_source, dt_source, dtype=float)
    y_kernel_rs = np.interp(t_kernel_rs, t_kernel, y_kernel)
    return t_kernel_rs, y_kernel_rs, t_kernel, y_kernel, float(ipi_s)


def build_synthetic_pulse_times() -> np.ndarray:
    if not np.isfinite(SYNTH_FREQUENCY_HZ) or SYNTH_FREQUENCY_HZ <= 0:
        raise ValueError("SYNTH_FREQUENCY_HZ must be positive.")
    if not np.isfinite(SYNTH_DURATION_S) or SYNTH_DURATION_S <= 0:
        raise ValueError("SYNTH_DURATION_S must be positive.")
    ipi_s = 1.0 / float(SYNTH_FREQUENCY_HZ)
    return np.arange(float(SYNTH_ONSET_S), float(SYNTH_ONSET_S + SYNTH_DURATION_S) + 0.25 * ipi_s, ipi_s, dtype=float)


def build_event_train(t_full: np.ndarray, pulse_times: np.ndarray) -> np.ndarray:
    event_train = np.zeros_like(t_full, dtype=float)
    for pulse_t in pulse_times:
        idx = int(np.argmin(np.abs(t_full - float(pulse_t))))
        event_train[idx] += float(SYNTH_AMPLITUDE_SCALE)
    return event_train


def build_pta_axis(dt_s: float, frequency_hz: float) -> np.ndarray:
    ipi_s = 1.0 / float(frequency_hz)
    window_s = float(SYNTH_PERIOD_FRACTION) * ipi_s
    return np.arange(-window_s, window_s + 0.5 * dt_s, dt_s, dtype=float)


def local_pre_baseline_correct(seg: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=float)
    pre = seg[t_rel < 0]
    pre = pre[np.isfinite(pre)]
    if len(pre) == 0:
        return seg
    return seg - float(np.median(pre))


def pulse_triggered_mean(
    trace: np.ndarray,
    t_full: np.ndarray,
    pulse_times: np.ndarray,
    t_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    for pulse_t in pulse_times:
        query_t = float(pulse_t) + t_rel
        win = np.interp(query_t, t_full, trace, left=np.nan, right=np.nan)
        if LOCAL_PTA_BASELINE_CORRECTION:
            win = local_pre_baseline_correct(win, t_rel)
        windows.append(win)
    if not windows:
        shape = np.full_like(t_rel, np.nan, dtype=float)
        return shape, shape.copy()
    arr = np.asarray(windows, dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def first_pulse_response(trace: np.ndarray, t_full: np.ndarray, first_pulse_s: float, t_rel: np.ndarray) -> np.ndarray:
    query_t = float(first_pulse_s) + t_rel
    seg = np.interp(query_t, t_full, trace, left=np.nan, right=np.nan)
    return local_pre_baseline_correct(seg, t_rel) if LOCAL_PTA_BASELINE_CORRECTION else seg


def compute_spectrogram(
    trace: np.ndarray,
    t_full: np.ndarray,
    source_trace_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt_s = float(np.nanmedian(np.diff(t_full)))
    fs_hz = 1.0 / dt_s
    nperseg = int(round(float(SPECTROGRAM_WINDOW_SEC) * fs_hz))
    nperseg = max(32, min(nperseg, len(trace)))
    noverlap = int(round(float(SPECTROGRAM_OVERLAP_FRACTION) * nperseg))
    noverlap = max(0, min(noverlap, nperseg - 1))

    freq_hz, t_spec_local, p_syn = scipy_spectrogram(
        trace,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    _, t_src_local, p_src = scipy_spectrogram(
        source_trace_bc,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    t_spec_s = float(t_full[0]) + np.asarray(t_spec_local, dtype=float)
    t_src_s = float(t_full[0]) + np.asarray(t_src_local, dtype=float)
    base_mask = t_src_s < 0.0
    if np.count_nonzero(base_mask) < 2:
        baseline = np.nanmedian(p_src, axis=1)
    else:
        baseline = np.nanmean(p_src[:, base_mask], axis=1)
    baseline = np.asarray(baseline, dtype=float)
    floor = np.nanmedian(baseline[np.isfinite(baseline) & (baseline > 0)])
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-12
    baseline = np.where(np.isfinite(baseline) & (baseline > 0), baseline, floor)

    rel_db = 10.0 * np.log10((p_syn + floor) / (baseline[:, None] + floor))
    keep = freq_hz <= float(SPECTROGRAM_FMAX_HZ)
    return np.asarray(freq_hz[keep], dtype=float), np.asarray(t_spec_s, dtype=float), np.asarray(rel_db[keep, :], dtype=float)


def run_model() -> dict:
    source_summary = load_pickle(summary_path(KERNEL_MOUSE, KERNEL_DATE, KERNEL_BLOCK))
    t_full, y_source_full, source_pulse_times = get_source_trace(source_summary)
    dt_source = float(np.nanmedian(np.diff(t_full)))
    if not np.isfinite(dt_source) or dt_source <= 0:
        raise ValueError("Source sampling interval is invalid.")

    source_baseline = baseline_value(y_source_full[t_full < 0], BASELINE_STAT)
    y_source_bc = np.asarray(y_source_full - source_baseline, dtype=float)

    t_kernel_rs, y_kernel_rs, t_kernel_plot, y_kernel_plot, source_ipi_s = prepare_kernel(source_summary, dt_source)

    synth_pulse_times = build_synthetic_pulse_times()
    event_train = build_event_train(t_full, synth_pulse_times)
    pred = np.convolve(event_train, y_kernel_rs, mode="full")[: len(t_full)]

    t_pta = build_pta_axis(dt_source, SYNTH_FREQUENCY_HZ)
    y_spta = first_pulse_response(pred, t_full, float(synth_pulse_times[0]), t_pta)
    y_mpta, y_mpta_sd = pulse_triggered_mean(pred, t_full, synth_pulse_times, t_pta)

    spec_f_hz, spec_t_s, spec_rel_db = compute_spectrogram(pred, t_full, y_source_bc)

    return {
        "t_full": t_full,
        "y_source_bc": y_source_bc,
        "pred": pred,
        "event_train": event_train,
        "pulse_times": synth_pulse_times,
        "t_pta": t_pta,
        "y_spta": y_spta,
        "y_mpta": y_mpta,
        "y_mpta_sd": y_mpta_sd,
        "spec_f_hz": spec_f_hz,
        "spec_t_s": spec_t_s,
        "spec_rel_db": spec_rel_db,
        "t_kernel_plot": t_kernel_plot,
        "y_kernel_plot": y_kernel_plot,
        "source_ipi_s": source_ipi_s,
        "synthetic_ipi_s": 1.0 / float(SYNTH_FREQUENCY_HZ),
        "dt_source": dt_source,
    }


def plot_result(result: dict) -> None:
    panel_specs = []
    if PLOT_FULL_TRACE:
        panel_specs.append(("full", 2.0))
    if PLOT_SINGLE_PTA:
        panel_specs.append(("spta", 1.2))
    if PLOT_MEAN_PTA:
        panel_specs.append(("mpta", 1.3))
    if PLOT_SPECTROGRAM:
        panel_specs.append(("spec", 1.6))
    if PLOT_INPUT_TRAIN:
        panel_specs.append(("input", 0.9))
    if PLOT_KERNEL:
        panel_specs.append(("kernel", 1.2))
    if not panel_specs:
        raise ValueError("No plot panels selected. Enable at least one PLOT_* toggle.")

    fig, ax = plt.subplots(
        len(panel_specs),
        1,
        figsize=(11, 2.2 + 1.8 * len(panel_specs)),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [h for _, h in panel_specs]},
    )
    if not isinstance(ax, np.ndarray):
        ax = np.asarray([ax], dtype=object)

    t_full = result["t_full"]
    pred = result["pred"]
    source_ref = result["y_source_bc"]
    pulse_times = result["pulse_times"]
    t_pta = result["t_pta"]
    y_spta = result["y_spta"]
    y_mpta = result["y_mpta"]
    spec_f_hz = result["spec_f_hz"]
    spec_t_s = result["spec_t_s"]
    spec_rel_db = result["spec_rel_db"]
    t_kernel_plot = result["t_kernel_plot"]
    y_kernel_plot = result["y_kernel_plot"]
    pulse_idx = np.flatnonzero(result["event_train"] > 0)

    for a, (kind, _) in zip(ax, panel_specs):
        if kind == "full":
            a.plot(t_full, pred, color="tab:red", lw=1.2, label="synthetic convolution trace")
            if PLOT_SOURCE_REFERENCE:
                a.plot(t_full, source_ref, color="tab:blue", lw=0.9, alpha=0.35, label="source trial raw trace")
            a.axvline(float(SYNTH_ONSET_S), color="k", ls="--", lw=0.9)
            a.axvline(float(SYNTH_ONSET_S + SYNTH_DURATION_S), color="tab:orange", ls="--", lw=0.9)
            a.set_title(
                f"Synthetic Trace | source={KERNEL_MOUSE} {KERNEL_DATE} {KERNEL_BLOCK} {KERNEL_TRIAL} | "
                f"freq={SYNTH_FREQUENCY_HZ:g} Hz | stim={SYNTH_DURATION_S:g} s"
            )
            a.set_xlabel("time from stim onset (s)")
            a.set_ylabel("signal")
            if PLOT_SOURCE_REFERENCE:
                a.legend(loc="best", fontsize=9)

        elif kind == "spta":
            a.plot(t_pta, y_spta, color="tab:red", lw=1.3)
            a.axvline(0.0, color="tab:red", ls="--", lw=0.9)
            a.axvline(float(result["synthetic_ipi_s"]), color="tab:orange", ls="--", lw=0.9)
            a.set_title("Synthetic First-Pulse PTA")
            a.set_xlabel("time from first pulse (s)")
            a.set_ylabel("signal")

        elif kind == "mpta":
            a.plot(t_pta, y_mpta, color="tab:red", lw=1.3)
            a.axvline(0.0, color="tab:red", ls="--", lw=0.9)
            a.axvline(float(result["synthetic_ipi_s"]), color="tab:orange", ls="--", lw=0.9)
            a.set_title("Synthetic Mean PTA")
            a.set_xlabel("time from pulse (s)")
            a.set_ylabel("signal")

        elif kind == "spec":
            im = a.pcolormesh(
                spec_t_s,
                spec_f_hz,
                spec_rel_db,
                shading="auto",
                cmap="RdBu_r",
                vmin=float(SPECTROGRAM_REL_DB_RANGE[0]),
                vmax=float(SPECTROGRAM_REL_DB_RANGE[1]),
            )
            a.axvline(float(SYNTH_ONSET_S), color="k", ls="--", lw=0.9)
            a.axvline(float(SYNTH_ONSET_S + SYNTH_DURATION_S), color="tab:orange", ls="--", lw=0.9)
            a.set_title("Synthetic Spectrogram | relative dB vs source pre-stim baseline")
            a.set_xlabel("time from stim onset (s)")
            a.set_ylabel("frequency (Hz)")
            fig.colorbar(im, ax=a, pad=0.01, label="relative power (dB)")

        elif kind == "input":
            if len(pulse_idx):
                a.vlines(t_full[pulse_idx], 0.0, result["event_train"][pulse_idx], color="tab:purple", lw=0.9)
            a.axvline(float(SYNTH_ONSET_S), color="k", ls="--", lw=0.9)
            a.axvline(float(SYNTH_ONSET_S + SYNTH_DURATION_S), color="tab:orange", ls="--", lw=0.9)
            a.set_title("Synthetic DBS Input Train")
            a.set_xlabel("time from stim onset (s)")
            a.set_ylabel("input")
            a.set_ylim(0.0, 1.15 * float(np.nanmax(result["event_train"][pulse_idx])) if len(pulse_idx) else 1.0)

        elif kind == "kernel":
            a.plot(t_kernel_plot, y_kernel_plot, color="black", lw=1.5)
            a.axvline(float(result["source_ipi_s"]), color="tab:orange", ls="--", lw=0.9, label="source next pulse")
            a.axvline(float(result["synthetic_ipi_s"]), color="tab:red", ls="--", lw=0.9, label="synthetic next pulse")
            a.set_title(
                f"Kernel | source={KERNEL_MOUSE} {KERNEL_DATE} {KERNEL_BLOCK} "
                + (f"{KERNEL_TRIAL}" if KERNEL_MODE == "trial" else "block mean")
            )
            a.set_xlabel("time after single pulse (s)")
            a.set_ylabel("kernel amplitude")
            a.legend(loc="best", fontsize=9)

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / (
            f"convolution_synthetic_{KERNEL_MOUSE}_{KERNEL_DATE}_{KERNEL_BLOCK}_{KERNEL_TRIAL}"
            f"__freq_{str(SYNTH_FREQUENCY_HZ).replace('.', 'p')}Hz__stim_{str(SYNTH_DURATION_S).replace('.', 'p')}s.png"
        )
        fig.savefig(out, dpi=200)
        print(f"[SAVED] {out}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    result = run_model()
    print(f"[INFO] source dt: {1000.0 * result['dt_source']:.3f} ms")
    print(f"[INFO] source next pulse: {1000.0 * result['source_ipi_s']:.2f} ms")
    print(f"[INFO] synthetic next pulse: {1000.0 * result['synthetic_ipi_s']:.2f} ms")
    print(f"[INFO] synthetic pulses: {len(result['pulse_times'])}")
    plot_result(result)


if __name__ == "__main__":
    main()
