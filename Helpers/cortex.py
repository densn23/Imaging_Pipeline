from __future__ import annotations

import argparse
import math
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.signal import butter, filtfilt, hilbert

from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = "Jamie11"
SINGLE_DATE = "03-04-26"
SINGLE_BLOCK = "R14"


# -------------------------
# EXECUTION TOGGLES
# -------------------------
SHOW_FIGURE = True
SAVE_FIGURE = False
FIG_DPI = 150


# -------------------------
# PANEL TOGGLES
# -------------------------
PLOT_SINGLE_PTA = True
PLOT_FULL_TRACE = True
PLOT_PULSE_WINDOWS = True
PLOT_FFT = False
PLOT_SPECTROGRAM = True
PLOT_BAND_SPECTRUM = True
PLOT_PULSOGRAM = False
PLOT_BETA_HILBERT = True


# -------------------------
# DISPLAY SETTINGS
# -------------------------
ONLY_TRIAL = None  # None = use block average, e.g. "R2_3" = use one trial from final pickle
MAX_PULSE_WINDOWS = 120
SINGLE_PTA_PRE_SEC = 0.010
FFT_FMAX_HZ = 250.0
SPEC_FMAX_HZ = 250.0
BAND_TRACE_RANGE_HZ = (1.0, 100.0)
BAND_TRACE_NORMALIZE_PERCENT = True
PULSOGRAM_TIME_RANGE_MS = (-30.0, 30.0)
SPECTROGRAM_CMAP = "turbo"
ENABLE_SPECTROGRAM_CONTRAST_SLIDER = True
SPECTROGRAM_CONTRAST_MIN = 1.0
SPECTROGRAM_CONTRAST_MAX = 20.0
SPECTROGRAM_CONTRAST_INIT = 1.0
SPECTROGRAM_CONTRAST_LOWER_PERCENTILE = 1.0
SPECTROGRAM_CONTRAST_UPPER_PERCENTILE = 99.0
SPECTROGRAM_MODE = "absolute"  # "absolute" or "relative"
SPECTROGRAM_REL_BASELINE = "pre_stim"  # "pre_stim" or "time_mean"
SPECTROGRAM_BASELINE_PRE_SEC = 0.5
FIRST_EPOCH_MIN_IPI_S = 0.003
FIRST_EPOCH_REL_TOL = 0.35
FIRST_EPOCH_STARTUP_INTERVALS = 3
BETA_BAND_HZ = (13.0, 30.0)
BETA_BASELINE_PRE_SEC = 0.5
BETA_RELATIVE_TO_PRE = True


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def array_or_empty(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    return np.asarray(x)


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def summary_path_from_parts(mouse: str, date: str, block: str) -> Path:
    return DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block / f"{block}_summary.pkl"


def choose_grid(n_panels: int) -> tuple[int, int]:
    n_cols = min(3, max(1, int(math.ceil(math.sqrt(n_panels)))))
    n_rows = int(math.ceil(n_panels / n_cols))
    return n_rows, n_cols


def has_block_stim_data(summary: dict) -> bool:
    train = summary.get("summary", {}).get("train_pta", {})
    return bool(train.get("available")) and int(train.get("n_trials", 0)) > 0


def has_block_baseline_spectrogram(summary: dict) -> bool:
    sec = summary.get("summary", {}).get("processed_notched", {}).get("baseline_spectrogram", {})
    t = np.asarray(sec.get("time_s", []), dtype=float)
    f = np.asarray(sec.get("freq_hz", []), dtype=float)
    p = np.asarray(sec.get("power_db", []), dtype=float)
    return len(t) > 0 and len(f) > 0 and p.shape == (len(f), len(t))


def has_trial_stim_data(summary: dict, trial_name: str | None) -> bool:
    if not trial_name:
        return False
    return trial_name in summary.get("trials", {}).get("train_pta", {})


def choose_train_trial(summary: dict) -> tuple[str | None, dict | None]:
    train_trials = summary["trials"]["train_pta"]
    names = sorted(train_trials.keys())
    if not names:
        return None, None

    if ONLY_TRIAL is not None and ONLY_TRIAL in train_trials:
        return ONLY_TRIAL, train_trials[ONLY_TRIAL]
    return None, None


def choose_pulsogram_trial(summary: dict) -> tuple[str | None, dict | None]:
    pulsogram_trials = summary["trials"]["pulsogram"]
    names = sorted(pulsogram_trials.keys())
    if not names:
        return None, None
    if ONLY_TRIAL is not None and ONLY_TRIAL in pulsogram_trials:
        return ONLY_TRIAL, pulsogram_trials[ONLY_TRIAL]
    return None, None


def build_pulsogram_from_trials(pulsogram_trials: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    by_pulse: dict[int, list[dict]] = {}
    for tr in pulsogram_trials.values():
        segs = tr.get("segment_dicts", [])
        used_idx = np.asarray(tr.get("used_pulse_indices", []), dtype=int)
        for pulse_idx, seg in zip(used_idx, segs):
            by_pulse.setdefault(int(pulse_idx), []).append(seg)

    if not by_pulse:
        return None, None, None

    pulse_numbers = []
    t_ref = None
    rows = []
    for pulse_idx in sorted(by_pulse.keys()):
        segs = by_pulse[pulse_idx]
        local_t_ref = None
        local_rows = []
        for seg in segs:
            t_rel = np.asarray(seg.get("t_rel_s", []), dtype=float)
            y = np.asarray(seg.get("signal", []), dtype=float)
            if local_t_ref is None and len(t_rel) and y.shape == t_rel.shape:
                local_t_ref = t_rel.copy()
            if local_t_ref is not None and len(t_rel) and t_rel.shape == local_t_ref.shape and np.allclose(t_rel, local_t_ref):
                local_rows.append(y)
        if local_t_ref is None or not local_rows:
            continue
        if t_ref is None:
            t_ref = local_t_ref.copy()
        if t_ref.shape != local_t_ref.shape or not np.allclose(t_ref, local_t_ref):
            continue
        pulse_numbers.append(int(pulse_idx) + 1)
        rows.append(np.nanmean(np.vstack(local_rows), axis=0))

    if t_ref is None or not rows:
        return None, None, None
    return np.asarray(pulse_numbers, dtype=int), np.asarray(t_ref, dtype=float), np.vstack(rows)


def build_pulsogram_from_summary(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    sec = summary["summary"]["pulsogram"]
    pulse_numbers = np.asarray(sec.get("pulse_numbers", []), dtype=int)
    if len(pulse_numbers) == 0:
        return None, None, None
    trial_results = summary["trials"]["pulsogram"]
    by_pulse: dict[int, list[dict]] = {}
    for tr in trial_results.values():
        segs = tr.get("segment_dicts", [])
        used_idx = np.asarray(tr.get("used_pulse_indices", []), dtype=int)
        for pulse_idx, seg in zip(used_idx, segs):
            by_pulse.setdefault(int(pulse_idx) + 1, []).append(seg)

    t_ref = None
    rows = []
    pulse_keep = []
    for pulse_num in pulse_numbers:
        segs = by_pulse.get(int(pulse_num), [])
        if not segs:
            continue
        local_t_ref = None
        local_rows = []
        for seg in segs:
            tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
            yx = np.asarray(seg.get("signal", []), dtype=float)
            if len(tx) == 0 or yx.shape != tx.shape:
                continue
            if local_t_ref is None:
                local_t_ref = tx.copy()
            if tx.shape == local_t_ref.shape and np.allclose(tx, local_t_ref):
                local_rows.append(yx)
        if local_t_ref is None or not local_rows:
            continue
        if t_ref is None:
            t_ref = local_t_ref.copy()
        if local_t_ref.shape != t_ref.shape or not np.allclose(local_t_ref, t_ref):
            continue
        pulse_keep.append(int(pulse_num))
        rows.append(np.nanmean(np.vstack(local_rows), axis=0))

    if t_ref is None or not rows:
        return None, None, None
    return np.asarray(pulse_keep, dtype=int), np.asarray(t_ref, dtype=float), np.vstack(rows)


def choose_processed_signal(td: dict) -> np.ndarray:
    for key in ("F_notched", "F_bleach_corr", "F_raw"):
        x = td.get(key)
        if x is not None:
            return np.asarray(x, dtype=float)
    return np.array([], dtype=float)


def estimate_fs(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return 1.0 / float(np.median(dt))


def bandpass_trace(x: np.ndarray, fs: float, band_hz: tuple[float, float], order: int = 3) -> np.ndarray | None:
    if not np.isfinite(fs) or fs <= 0:
        return None
    nyq = 0.5 * fs
    low, high = float(band_hz[0]), float(band_hz[1])
    if not (0 < low < high < nyq):
        return None
    b, a = butter(int(order), [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def extract_first_epoch_pulse_times(pulse_times: np.ndarray) -> np.ndarray:
    pulse_times = np.asarray(pulse_times, dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < 2:
        return pulse_times

    kept = [float(pulse_times[0])]
    startup: list[float] = []
    ref_isi = np.nan

    for tp in pulse_times[1:]:
        dt = float(tp - kept[-1])
        if not np.isfinite(dt) or dt <= 0:
            continue
        if dt < float(FIRST_EPOCH_MIN_IPI_S):
            continue
        if not np.isfinite(ref_isi):
            kept.append(float(tp))
            startup.append(dt)
            if len(startup) >= int(FIRST_EPOCH_STARTUP_INTERVALS):
                ref_isi = float(np.median(np.asarray(startup, dtype=float)))
            continue
        if abs(dt - ref_isi) <= float(FIRST_EPOCH_REL_TOL) * ref_isi:
            kept.append(float(tp))
        else:
            break

    if len(kept) >= 2:
        return np.asarray(kept, dtype=float)

    out = [float(pulse_times[0])]
    for tp in pulse_times[1:]:
        if float(tp - out[-1]) >= float(FIRST_EPOCH_MIN_IPI_S):
            out.append(float(tp))
    return np.asarray(out, dtype=float)


def estimate_stim_freq_hz(pulse_times: np.ndarray) -> float:
    pulse_times = np.asarray(pulse_times, dtype=float)
    if len(pulse_times) < 2:
        return np.nan
    isi = np.diff(pulse_times)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    if len(isi) == 0:
        return np.nan
    return 1.0 / float(np.median(isi))


def get_cortex_cache(summary: dict) -> dict:
    cache = summary.get("_cortex_cache")
    if isinstance(cache, dict):
        return cache

    trial_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    if not trial_names:
        trial_names = sorted(summary.get("trials", {}).get("train_pta", {}).keys())

    trials = {}
    f_vals = []

    for trial_name in trial_names:
        td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
        pulse_times_all = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times_all = pulse_times_all[np.isfinite(pulse_times_all)]
        pulse_times = extract_first_epoch_pulse_times(pulse_times_all)
        f_stim = estimate_stim_freq_hz(pulse_times)

        trials[trial_name] = {
            "pulse_times_first_epoch_s": np.asarray(pulse_times, dtype=float),
            "n_pulses_first_epoch": int(len(pulse_times)),
            "f_stim_hz_first_epoch": float(f_stim) if np.isfinite(f_stim) else np.nan,
        }

        if np.isfinite(f_stim):
            f_vals.append(float(f_stim))

    cache = {
        "trials": trials,
        "block": {
            "f_stim_hz_mean": float(np.nanmean(f_vals)) if f_vals else np.nan,
        },
    }
    summary["_cortex_cache"] = cache
    return cache


def compute_beta_envelope(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    fs = estimate_fs(t)
    xf = bandpass_trace(x, fs, BETA_BAND_HZ, order=3)
    if xf is None:
        return None, None
    env = np.abs(hilbert(xf))
    env = np.asarray(env, dtype=float)
    if BETA_RELATIVE_TO_PRE:
        pre = env[(t < 0) & (t >= -float(BETA_BASELINE_PRE_SEC))]
        ref = float(np.median(pre)) if len(pre) else np.nan
        if np.isfinite(ref) and ref > 0:
            env = 100.0 * env / ref
    return np.asarray(t, dtype=float), env


def extract_first_pta_display_segment(summary: dict, trial_name: str, x_hi: float) -> tuple[np.ndarray, np.ndarray, float, float] | tuple[None, None, float, float]:
    td = summary.get("trials", {}).get("processed_notched", {}).get(trial_name, {})
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(td.get("t", []), dtype=float)
    x = choose_processed_signal(td)
    if len(t) == 0 or x.shape != t.shape:
        return None, None, np.nan, np.nan

    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) == 0:
        return None, None, np.nan, np.nan

    ipi_ms = safe_float(td_e.get("median_ipi_ms"))
    pre_s = 0.001 * float(ipi_ms) if np.isfinite(ipi_ms) and ipi_ms > 0 else np.nan
    if not np.isfinite(pre_s) or pre_s <= 0:
        diffs = np.diff(pulse_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        pre_s = float(np.median(diffs)) if len(diffs) else float(SINGLE_PTA_PRE_SEC)

    t0 = float(pulse_times[0])
    t_rel = t - t0
    second_t = float(pulse_times[1] - pulse_times[0]) if len(pulse_times) >= 2 else np.nan

    pre = x[(t_rel < 0) & (t_rel >= -0.5)]
    baseline = float(np.median(pre)) if len(pre) else 0.0
    x_corr = x - baseline

    keep = (t_rel >= -float(pre_s)) & (t_rel <= float(x_hi))
    if not np.any(keep):
        return None, None, second_t, pre_s

    return np.asarray(t_rel[keep], dtype=float), np.asarray(x_corr[keep], dtype=float), second_t, pre_s


def rebuild_single_pta_display(summary: dict, trial_names: list[str], x_hi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], float] | tuple[None, None, None, list[float], float]:
    segments = []
    second_times: list[float] = []
    dts = []
    pre_vals = []
    for name in trial_names:
        t_seg, y_seg, second_t, pre_s = extract_first_pta_display_segment(summary, name, x_hi)
        if t_seg is None or y_seg is None or len(t_seg) < 2 or y_seg.shape != t_seg.shape:
            continue
        segments.append((t_seg, y_seg))
        if np.isfinite(second_t) and second_t > 0:
            second_times.append(float(second_t))
        if np.isfinite(pre_s) and pre_s > 0:
            pre_vals.append(float(pre_s))
        dt = np.diff(t_seg)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            dts.append(float(np.median(dt)))

    if not segments or not dts:
        return None, None, None, second_times, np.nan

    dt = float(np.median(np.asarray(dts, dtype=float)))
    if not np.isfinite(dt) or dt <= 0:
        return None, None, None, second_times, np.nan

    pre_s = float(np.nanmedian(pre_vals)) if pre_vals else float(SINGLE_PTA_PRE_SEC)
    t_grid = np.arange(-float(pre_s), float(x_hi) + 0.5 * dt, dt, dtype=float)
    rows = []
    for t_seg, y_seg in segments:
        rows.append(np.interp(t_grid, t_seg, y_seg, left=np.nan, right=np.nan))
    Y = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_mean = np.nanmean(Y, axis=0)
        y_sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.full_like(y_mean, np.nan)
    return t_grid, y_mean, y_sd, second_times, pre_s


def extract_pulse_window_segments(summary: dict, trial_name: str, post_periods: float = 3.0) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    td = summary.get("trials", {}).get("processed_notched", {}).get(trial_name, {})
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(td.get("t", []), dtype=float)
    x = choose_processed_signal(td)
    if len(t) == 0 or x.shape != t.shape:
        return None, None

    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < 2:
        return None, None

    ipi_ms = safe_float(td_e.get("median_ipi_ms"))
    ipi_s = 0.001 * float(ipi_ms) if np.isfinite(ipi_ms) and ipi_ms > 0 else float(np.median(np.diff(pulse_times)))
    if not np.isfinite(ipi_s) or ipi_s <= 0:
        return None, None

    pre_s = float(ipi_s)
    post_s = float(post_periods) * float(ipi_s)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None, None
    dt = float(np.median(dt))
    t_grid = np.arange(-pre_s, post_s + 0.5 * dt, dt, dtype=float)

    rows = []
    for tp in pulse_times:
        t_rel = t - float(tp)
        keep = (t_rel >= -pre_s) & (t_rel <= post_s)
        if int(np.sum(keep)) < 4:
            continue
        t_seg = np.asarray(t_rel[keep], dtype=float)
        x_seg = np.asarray(x[keep], dtype=float)
        pre = x_seg[(t_seg < 0) & (t_seg >= -pre_s)]
        baseline = float(np.median(pre)) if len(pre) else 0.0
        rows.append(np.interp(t_grid, t_seg, x_seg - baseline, left=np.nan, right=np.nan))

    if not rows:
        return None, None
    return t_grid, np.vstack(rows)


def rebuild_pulse_window_block_display(summary: dict, trial_names: list[str], post_periods: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    t_ref = None
    trial_means = []
    for name in trial_names:
        t_trial, segs = extract_pulse_window_segments(summary, name, post_periods=post_periods)
        if t_trial is None or segs is None or segs.ndim != 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y_trial = np.nanmean(segs, axis=0)
        if t_ref is None:
            t_ref = t_trial.copy()
        elif t_trial.shape != t_ref.shape or not np.allclose(t_trial, t_ref):
            y_trial = np.interp(t_ref, t_trial, y_trial, left=np.nan, right=np.nan)
        trial_means.append(y_trial)

    if t_ref is None or not trial_means:
        return None, None, None
    Y = np.vstack(trial_means)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(Y, axis=0)
        spread = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.full_like(mean_curve, np.nan)
    return t_ref, mean_curve, spread


def plot_single_pta(ax, summary: dict):
    second_times = []
    ipi_vals = []
    if ONLY_TRIAL is None:
        stim_names = summary["trials"].get("stim_trial_names", [])
        for name in stim_names:
            td_e = summary["trials"].get("ephys", {}).get(name, {})
            ipi_ms = safe_float(td_e.get("median_ipi_ms"))
            if np.isfinite(ipi_ms) and ipi_ms > 0:
                ipi_vals.append(0.001 * float(ipi_ms))
        if not ipi_vals:
            for seg in summary["trials"].get("first_pta_segments", []):
                second_t = safe_float(seg.get("second_pulse_rel_s"))
                if np.isfinite(second_t) and second_t > 0:
                    ipi_vals.append(float(second_t))
        ipi_s = float(np.nanmedian(ipi_vals)) if ipi_vals else np.nan
    else:
        ipi_s = np.nan
        td_e = summary["trials"].get("ephys", {}).get(ONLY_TRIAL, {})
        ipi_ms = safe_float(td_e.get("median_ipi_ms"))
        if np.isfinite(ipi_ms) and ipi_ms > 0:
            ipi_s = 0.001 * float(ipi_ms)
        if (not np.isfinite(ipi_s) or ipi_s <= 0) and second_times:
            second_t = float(np.nanmedian(second_times))
            if np.isfinite(second_t) and second_t > 0:
                ipi_s = second_t
    if not np.isfinite(ipi_s) or ipi_s <= 0:
        ipi_s = 1.0 / 135.0
    x_hi = 3.0 * float(ipi_s)

    if ONLY_TRIAL is None:
        t, y, s, second_times, pre_s = rebuild_single_pta_display(summary, stim_names, x_hi)
        n_trials = int(len(stim_names))
        title = f"First-Pulse PTA | {summary.get('date')} {summary.get('block')} | n={n_trials}"
    else:
        t, y, second_t, pre_s = extract_first_pta_display_segment(summary, ONLY_TRIAL, x_hi)
        s = None if t is None else np.full_like(y, np.nan)
        second_times = [second_t]
        title = f"{ONLY_TRIAL} | First-Pulse PTA"

    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        if ONLY_TRIAL is None:
            sec = summary["summary"]["single_pta"]
            t = np.asarray(sec.get("t_rel_s", []), dtype=float)
            y = np.asarray(sec.get("mean", []), dtype=float)
            s = np.asarray(sec.get("spread", []), dtype=float)
            n_trials = int(sec.get("n_trials", 0))
            title = f"First-Pulse PTA | {summary.get('date')} {summary.get('block')} | n={n_trials}"
            second_times = [
                safe_float(seg.get("second_pulse_rel_s"))
                for seg in summary["trials"].get("first_pta_segments", [])
                if np.isfinite(safe_float(seg.get("second_pulse_rel_s")))
            ]
        else:
            segments = summary["trials"].get("first_pta_segments", [])
            match = None
            for seg in segments:
                if seg.get("trial") == ONLY_TRIAL:
                    match = seg
                    break
            if match is None:
                ax.axis("off")
                return
            t = np.asarray(match.get("t_rel_s", []), dtype=float)
            y = np.asarray(match.get("signal", []), dtype=float)
            s = np.full_like(y, np.nan)
            title = f"{ONLY_TRIAL} | First-Pulse PTA"
            second_times = [safe_float(match.get("second_pulse_rel_s"))]
        if len(t) == 0 or y.shape != t.shape:
            ax.axis("off")
            return

    if ONLY_TRIAL is None:
        ax.plot(t, y, color="black", lw=2.0, label="mean")
    else:
        ax.plot(t, y, color="black", lw=2.0)
    if s is not None and s.shape == t.shape:
        fill_label = "SD" if ONLY_TRIAL is None else None
        ax.fill_between(t, y - s, y + s, color="black", alpha=0.2, label=fill_label)
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.2, label="first pulse")
    if second_times:
        second_t = float(np.nanmedian(second_times))
        if np.isfinite(second_t) and np.min(t) <= second_t <= np.max(t):
            ax.axvline(second_t, color="tab:orange", ls="--", lw=1.2, label="second pulse")
    if not np.isfinite(pre_s) or pre_s <= 0:
        pre_s = float(SINGLE_PTA_PRE_SEC)
    ax.set_xlim(-float(pre_s), x_hi)
    ax.set_title(title)
    ax.set_xlabel("time from first pulse (s)")
    ax.set_ylabel("signal (a.u. baseline-corrected)")
    if ONLY_TRIAL is None:
        ax.legend(loc="best", fontsize=8)


def plot_full_trace(ax, summary: dict):
    if ONLY_TRIAL is None:
        proc = summary["summary"]["processed_notched"]
        sec = proc.get("stim", {})
        is_baseline = False
        t = np.asarray(sec.get("t_common", []), dtype=float)
        x = np.asarray(sec.get("F_notched_mean", []), dtype=float)
        s = np.asarray(sec.get("F_notched_sd", []), dtype=float)
        if len(t) == 0 or x.shape != t.shape:
            sec = proc.get("baseline", {})
            is_baseline = True
        t = np.asarray(sec.get("t_common", []), dtype=float)
        x = np.asarray(sec.get("F_notched_mean", []), dtype=float)
        s = np.asarray(sec.get("F_notched_sd", []), dtype=float)
        if len(t) == 0 or x.shape != t.shape:
            ax.axis("off")
            return
        ax.plot(t, x, color="tab:blue", lw=1.5)
        if s.shape == t.shape:
            ax.fill_between(t, x - s, x + s, color="tab:blue", alpha=0.18)
        if not is_baseline:
            ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
            stim_names = summary["trials"].get("stim_trial_names", [])
            last_pulses = []
            for name in stim_names:
                td_e = summary["trials"].get("ephys", {}).get(name, {})
                pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
                pulse_times = pulse_times[np.isfinite(pulse_times)]
                if len(pulse_times):
                    last_pulses.append(float(pulse_times[-1]))
            if last_pulses:
                stim_off = float(np.nanmedian(last_pulses))
                ax.axvline(stim_off, color="tab:orange", ls="--", lw=1.0, alpha=0.9)
            ax.set_title("Stim-Aligned Full Trace | Block Mean")
            ax.set_xlabel("time from stim onset (s)")
        else:
            ax.set_title("Baseline Full Trace | Block Mean")
            ax.set_xlabel("time from trial start (s)")
        ax.set_ylabel("signal (a.u.)")
    else:
        trial_name, tr = choose_train_trial(summary)
        if tr is None:
            td = summary["trials"].get("processed_notched", {}).get(ONLY_TRIAL, {})
            t = np.asarray(td.get("t", []), dtype=float)
            x = np.asarray(td.get("F_notched", []), dtype=float)
            if len(t) == 0 or x.shape != t.shape:
                ax.axis("off")
                return
            ax.plot(t, x, color="tab:blue", lw=0.8)
            ax.set_title(f"{ONLY_TRIAL} | Full Trace")
            ax.set_xlabel("time from trial start (s)")
            ax.set_ylabel("signal (a.u.)")
            return
        t = np.asarray(tr.get("t_full_s", []), dtype=float)
        x = np.asarray(tr.get("signal_full", []), dtype=float)
        pulse_times = np.asarray(tr.get("pulse_times_s", []), dtype=float)
        if len(t) == 0 or x.shape != t.shape:
            ax.axis("off")
            return
        ax.plot(t, x, color="tab:blue", lw=0.8)
        for tp in pulse_times:
            ax.axvline(float(tp), color="tab:red", alpha=0.12, lw=0.7)
        if len(pulse_times):
            ax.axvline(0.0, color="tab:red", ls="--", lw=1.1, label="first pulse")
            stim_off = float(pulse_times[-1])
            ax.axvline(stim_off, color="tab:orange", ls="--", lw=1.1, label="last pulse")
        ax.set_title(f"{trial_name} | Full Trace")
        ax.set_xlabel("time from stim onset (s)")
        ax.set_ylabel("signal (a.u.)")
        if len(pulse_times):
            ax.legend(loc="best", fontsize=8)


def plot_pulse_windows(ax, summary: dict):
    if ONLY_TRIAL is None:
        stim_names = summary["trials"].get("stim_trial_names", [])
        second_times = []
        for name in stim_names:
            td_e = summary["trials"].get("ephys", {}).get(name, {})
            pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
            pulse_times = pulse_times[np.isfinite(pulse_times)]
            if len(pulse_times) >= 2:
                second_times.append(float(pulse_times[1] - pulse_times[0]))
        t_rel, mean_curve, spread = rebuild_pulse_window_block_display(summary, stim_names, post_periods=3.0)
        if t_rel is None or mean_curve is None or spread is None:
            sec = summary["summary"]["train_pta"]
            t_rel = np.asarray(sec.get("t_rel_s", []), dtype=float)
            mean_curve = np.asarray(sec.get("mean_across_trials", []), dtype=float)
            spread = np.asarray(sec.get("sd_across_trials", []), dtype=float)
        if len(t_rel) == 0 or mean_curve.shape != t_rel.shape:
            ax.axis("off")
            return
        ax.plot(t_rel, mean_curve, color="black", lw=2.0, label="within-trial mean")
        if spread.shape == t_rel.shape:
            ax.fill_between(t_rel, mean_curve - spread, mean_curve + spread, color="black", alpha=0.2, label="SD")
        if second_times:
            second_t = float(np.nanmedian(second_times))
            if np.isfinite(second_t) and t_rel[0] <= second_t <= t_rel[-1]:
                ax.axvline(second_t, color="tab:orange", ls="--", lw=1.1, label="second pulse")
        title = f"{summary.get('date')} {summary.get('block')} | pulse windows | block mean"
    else:
        trial_name, tr = choose_train_trial(summary)
        if tr is None:
            ax.axis("off")
            return
        t_rel, segs = extract_pulse_window_segments(summary, trial_name, post_periods=3.0)
        if t_rel is None or segs is None:
            t_rel = np.asarray(tr.get("t_rel_s", []), dtype=float)
            segs = np.asarray(tr.get("pulse_segments", []), dtype=float)
        if segs is not None and np.asarray(segs).ndim == 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_curve = np.nanmean(segs, axis=0)
                spread = np.nanstd(segs, axis=0, ddof=1) if np.asarray(segs).shape[0] >= 2 else np.full(segs.shape[1], np.nan, dtype=float)
        else:
            mean_curve = np.asarray(tr.get("pta_mean", []), dtype=float)
            spread = np.asarray(tr.get("pta_spread", []), dtype=float)
        if len(t_rel) == 0 or segs.ndim != 2 or segs.shape[1] != len(t_rel) or mean_curve.shape != t_rel.shape:
            ax.axis("off")
            return
        n_plot = min(MAX_PULSE_WINDOWS, segs.shape[0])
        for k in range(n_plot):
            ax.plot(t_rel, segs[k], color="tab:blue", alpha=0.12, lw=0.7)
        ax.plot(t_rel, mean_curve, color="black", lw=2.0, label="within-trial mean")
        if spread.shape == t_rel.shape:
            ax.fill_between(t_rel, mean_curve - spread, mean_curve + spread, color="black", alpha=0.2, label="SD")
        pulse_times = np.asarray(tr.get("pulse_times_s", []), dtype=float)
        if len(pulse_times) >= 2:
            last_pulse_rel = float(pulse_times[-1] - pulse_times[0])
            if t_rel[0] <= last_pulse_rel <= t_rel[-1]:
                ax.axvline(last_pulse_rel, color="tab:orange", ls="--", lw=1.1, label="last pulse")
        title = f"{trial_name} | Pulse-Triggered Windows"
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.1)
    ax.set_xlim(float(np.min(t_rel)), float(np.max(t_rel)))
    ax.set_title(title)
    ax.set_xlabel("time from pulse (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.legend(loc="best", fontsize=8)


def plot_fft(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary["summary"]["train_pta"]["power_spectrum"]
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p = np.asarray(sec.get("psd_db_mean", []), dtype=float)
        s = np.asarray(sec.get("psd_db_sd", []), dtype=float)
        title = f"{summary.get('date')} {summary.get('block')} | FFT / PSD | train mean"
        f_stim = safe_float(summary["summary"]["train_pta"]["metrics"].get("f_stim_hz_mean"))
        if len(f) == 0 or p.shape != f.shape:
            ax.axis("off")
            return
        ax.plot(f, p, color="tab:purple", lw=1.5)
        if s.shape == f.shape:
            ax.fill_between(f, p - s, p + s, color="tab:purple", alpha=0.15)
    else:
        trial_name, tr = choose_train_trial(summary)
        if tr is None:
            ax.axis("off")
            return
        spectral = tr.get("spectral", {})
        f = np.asarray(spectral.get("freq_hz", []), dtype=float)
        p = np.asarray(spectral.get("psd_db", []), dtype=float)
        if len(f) == 0 or p.shape != f.shape:
            ax.axis("off")
            return
        f_stim = safe_float(tr.get("f_stim_hz"))
        ax.plot(f, p, color="tab:purple", lw=1.0)
        title = f"{trial_name} | FFT spectrum | f={f_stim:.1f} Hz"
    if np.isfinite(f_stim):
        nyq = float(np.max(f))
        for mul in [1, 2, 3]:
            ft = mul * f_stim
            if ft < nyq:
                ax.axvline(ft, color="tab:red", ls="--", alpha=0.6)
    ax.set_xlim(0, min(float(FFT_FMAX_HZ), float(np.max(f))))
    ax.set_title(title)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB)")


def plot_spectrogram(ax, summary: dict):
    cortex = get_cortex_cache(summary)
    if ONLY_TRIAL is None:
        if has_block_stim_data(summary):
            sec = summary["summary"]["train_pta"]["spectrogram"]
            t = np.asarray(sec.get("time_s", []), dtype=float)
            f = np.asarray(sec.get("freq_hz", []), dtype=float)
            p = np.asarray(sec.get("power_db_mean", []), dtype=float)
            title = f"{summary.get('date')} {summary.get('block')} | spectrogram | train mean"
            f_stim = safe_float(cortex.get("block", {}).get("f_stim_hz_mean"))
        else:
            sec = summary["summary"]["processed_notched"].get("baseline_spectrogram", {})
            t = np.asarray(sec.get("time_s", []), dtype=float)
            f = np.asarray(sec.get("freq_hz", []), dtype=float)
            p = np.asarray(sec.get("power_db", []), dtype=float)
            title = f"{summary.get('date')} {summary.get('block')} | spectrogram | baseline mean"
            f_stim = np.nan
    else:
        trial_name, tr = choose_train_trial(summary)
        if tr is None:
            ax.axis("off")
            return
        sec = tr.get("spectrogram", {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p = np.asarray(sec.get("power_db", []), dtype=float)
        title = f"{trial_name} | spectrogram"
        f_stim = safe_float(cortex.get("trials", {}).get(trial_name, {}).get("f_stim_hz_first_epoch"))
    if len(t) == 0 or len(f) == 0 or p.shape != (len(f), len(t)):
        ax.axis("off")
        return
    p_show = p.copy()
    if SPECTROGRAM_MODE == "relative":
        if SPECTROGRAM_REL_BASELINE == "pre_stim":
            m_pre = (t < 0) & (t >= -float(SPECTROGRAM_BASELINE_PRE_SEC))
            if np.any(m_pre):
                baseline = np.nanmean(p[:, m_pre], axis=1, keepdims=True)
            else:
                baseline = np.nanmean(p, axis=1, keepdims=True)
        else:
            baseline = np.nanmean(p, axis=1, keepdims=True)
        p_lin = np.power(10.0, p / 10.0)
        baseline_lin = np.power(10.0, baseline / 10.0)
        ratio = p_lin / np.maximum(baseline_lin, 1e-30)
        p_show = 10.0 * np.log10(np.maximum(ratio, 1e-30))
    im = ax.pcolormesh(t, f, p_show, shading="auto", cmap=SPECTROGRAM_CMAP)
    ax.set_ylim(0, min(float(SPEC_FMAX_HZ), float(np.max(f))))
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    cbar_label = "power (dB)" if SPECTROGRAM_MODE == "absolute" else "relative power (dB re baseline)"
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)
    finite = p_show[np.isfinite(p_show)]
    if finite.size > 0:
        ax._spectrogram_im = im
        q_lo, q_hi = np.percentile(
            finite,
            [float(SPECTROGRAM_CONTRAST_LOWER_PERCENTILE), float(SPECTROGRAM_CONTRAST_UPPER_PERCENTILE)],
        )
        center = float(np.median(finite))
        half_range = float(max(center - q_lo, q_hi - center))
        if not np.isfinite(half_range) or half_range <= 0:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            center = 0.5 * (vmin + vmax)
            half_range = 0.5 * max(vmax - vmin, 1e-6)
        ax._spectrogram_center = center
        ax._spectrogram_half_range = half_range


def plot_beta_hilbert(ax, summary: dict):
    stim_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    if ONLY_TRIAL is None:
        curves = []
        t_ref = None
        for name in stim_names:
            td = summary.get("trials", {}).get("processed_notched", {}).get(name, {})
            t = np.asarray(td.get("t", []), dtype=float)
            x = choose_processed_signal(td)
            if len(t) == 0 or x.shape != t.shape:
                continue
            tb, yb = compute_beta_envelope(t, x)
            if tb is None or yb is None or len(tb) == 0 or yb.shape != tb.shape:
                continue
            if t_ref is None:
                t_ref = tb.copy()
                curves.append(yb.copy())
            else:
                curves.append(np.interp(t_ref, tb, yb, left=np.nan, right=np.nan))
        if t_ref is None or not curves:
            ax.axis("off")
            return
        Y = np.vstack(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y_mean = np.nanmean(Y, axis=0)
            y_sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.full_like(y_mean, np.nan)
        ax.plot(t_ref, y_mean, color="black", lw=2.0, label="mean")
        if y_sd.shape == t_ref.shape:
            ax.fill_between(t_ref, y_mean - y_sd, y_mean + y_sd, color="black", alpha=0.2, label="SD")
        title = f"Beta Hilbert Envelope | {summary.get('date')} {summary.get('block')}"
        pulse_last = []
        for name in stim_names:
            td_e = summary.get("trials", {}).get("ephys", {}).get(name, {})
            p = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
            p = p[np.isfinite(p)]
            if len(p):
                pulse_last.append(float(p[-1]))
        if pulse_last:
            ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
            ax.axvline(float(np.nanmedian(pulse_last)), color="tab:orange", ls="--", lw=1.0, alpha=0.9)
        ax.legend(loc="best", fontsize=8)
    else:
        td = summary.get("trials", {}).get("processed_notched", {}).get(ONLY_TRIAL, {})
        t = np.asarray(td.get("t", []), dtype=float)
        x = choose_processed_signal(td)
        if len(t) == 0 or x.shape != t.shape:
            ax.axis("off")
            return
        tb, yb = compute_beta_envelope(t, x)
        if tb is None or yb is None or len(tb) == 0 or yb.shape != tb.shape:
            ax.axis("off")
            return
        ax.plot(tb, yb, color="black", lw=2.0)
        td_e = summary.get("trials", {}).get("ephys", {}).get(ONLY_TRIAL, {})
        p = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        p = p[np.isfinite(p)]
        if len(p):
            ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
            ax.axvline(float(p[-1]), color="tab:orange", ls="--", lw=1.0, alpha=0.9)
        title = f"{ONLY_TRIAL} | Beta Hilbert Envelope"

    ax.set_title(title)
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("beta amp (% pre)" if BETA_RELATIVE_TO_PRE else "beta amp (a.u.)")


def plot_band_spectrum(ax, summary: dict):
    sec = summary["summary"]["processed_notched"].get("spectral_post_notch", {})
    f = np.asarray(sec.get("freq_hz", []), dtype=float)
    y = np.asarray(sec.get("psd_linear_mean", []), dtype=float)
    s = np.asarray(sec.get("psd_linear_sd", []), dtype=float)
    if len(f) == 0 or y.shape != f.shape:
        ax.axis("off")
        return

    lo, hi = BAND_TRACE_RANGE_HZ
    m = (f >= float(lo)) & (f <= float(hi))
    if np.sum(m) < 2:
        ax.axis("off")
        return
    x = f[m]
    y_mean = y[m]
    y_sd = s[m] if s.shape == f.shape else np.full(np.sum(m), np.nan, dtype=float)

    ylabel = "power"
    if BAND_TRACE_NORMALIZE_PERCENT:
        ref = float(np.trapezoid(y_mean, x))
        if np.isfinite(ref) and ref > 0:
            y_mean = 100.0 * y_mean / ref
            y_sd = 100.0 * y_sd / ref
            ylabel = "normalized power (%)"

    band_defs = sec.get("band_defs_hz", {})
    for band_name, band_range in band_defs.items():
        if len(band_range) != 2:
            continue
        left = max(float(lo), float(band_range[0]))
        right = min(float(hi), float(band_range[1]))
        if right <= left:
            continue
        ax.axvspan(left, right, color="0.93", alpha=0.8, zorder=0)
        ax.text(0.5 * (left + right), 0.98, band_name.replace("_", " "), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="0.35")

    ax.plot(x, y_mean, color="tab:blue", lw=2.0)
    if np.any(np.isfinite(y_sd)):
        ax.fill_between(x, y_mean - y_sd, y_mean + y_sd, color="tab:blue", alpha=0.2)
    ax.set_xlim(float(lo), float(hi))
    ax.set_title("Post-Notch Band Spectrum")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(ylabel)


def plot_pulsogram(ax, summary: dict):
    if ONLY_TRIAL is None:
        pulse_numbers, t_rel, M = build_pulsogram_from_summary(summary)
    else:
        _, trial = choose_pulsogram_trial(summary)
        if trial is None:
            ax.axis("off")
            return
        pulse_numbers, t_rel, M = build_pulsogram_from_trials({ONLY_TRIAL: trial})
    if pulse_numbers is None or t_rel is None or M is None:
        ax.axis("off")
        return
    t_ms = 1000.0 * t_rel
    lo_ms, hi_ms = PULSOGRAM_TIME_RANGE_MS
    m = (t_ms >= lo_ms) & (t_ms <= hi_ms)
    if np.sum(m) < 2:
        ax.axis("off")
        return
    M_show = M[:, m]
    finite = M_show[np.isfinite(M_show)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else np.nan
    im_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "extent": [float(t_ms[m][0]), float(t_ms[m][-1]), int(pulse_numbers[0]), int(pulse_numbers[-1])],
        "cmap": "RdBu_r",
    }
    if np.isfinite(max_abs) and max_abs > 0:
        im_kwargs["vmin"] = -max_abs
        im_kwargs["vmax"] = max_abs
    im = ax.imshow(
        M_show,
        **im_kwargs,
    )
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.set_title(
        f"Pulsogram heatmap | {summary.get('date')} {summary.get('block')}"
        if ONLY_TRIAL is None
        else f"Pulsogram heatmap | {ONLY_TRIAL}"
    )
    ax.set_xlabel("time from pulse (ms)")
    ax.set_ylabel("pulse number")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("signal (a.u.)")
    ax._pulsogram_im = im
    ax._pulsogram_max_abs = max_abs


def add_spectrogram_contrast_slider(fig, axes):
    if not SHOW_FIGURE or not ENABLE_SPECTROGRAM_CONTRAST_SLIDER:
        return None

    spectrogram_axes = [
        ax for ax in axes
        if hasattr(ax, "_spectrogram_im") and hasattr(ax, "_spectrogram_center") and hasattr(ax, "_spectrogram_half_range")
    ]
    if not spectrogram_axes:
        return None

    ranges = []
    for ax in spectrogram_axes:
        center = float(ax._spectrogram_center)
        half_range = float(ax._spectrogram_half_range)
        if np.isfinite(center) and np.isfinite(half_range) and half_range > 0:
            ranges.append((center, half_range))
    if not ranges:
        return None

    slider_ax = fig.add_axes([0.18, 0.03, 0.64, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="Spectrogram Contrast",
        valmin=float(SPECTROGRAM_CONTRAST_MIN),
        valmax=float(SPECTROGRAM_CONTRAST_MAX),
        valinit=float(SPECTROGRAM_CONTRAST_INIT),
        valstep=0.5,
    )

    def update(val):
        contrast = max(float(val), 1e-6)
        for ax in spectrogram_axes:
            im = ax._spectrogram_im
            center = float(ax._spectrogram_center)
            half_range = float(ax._spectrogram_half_range)
            if not np.isfinite(center) or not np.isfinite(half_range) or half_range <= 0:
                continue
            lim = half_range / contrast
            im.set_clim(center - lim, center + lim)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(float(SPECTROGRAM_CONTRAST_INIT))
    fig._spectrogram_contrast_slider = slider
    return slider


def build_panels(summary: dict):
    panels = []
    stim_available = has_block_stim_data(summary) if ONLY_TRIAL is None else has_trial_stim_data(summary, ONLY_TRIAL)
    baseline_spec_available = has_block_baseline_spectrogram(summary) if ONLY_TRIAL is None else False
    if PLOT_FULL_TRACE:
        panels.append(("full_trace", plot_full_trace))
    if PLOT_SINGLE_PTA and stim_available:
        panels.append(("single_pta", plot_single_pta))
    if PLOT_PULSE_WINDOWS and stim_available:
        panels.append(("pulse_windows", plot_pulse_windows))
    if PLOT_FFT and stim_available:
        panels.append(("fft", plot_fft))
    if PLOT_SPECTROGRAM and (stim_available or baseline_spec_available):
        panels.append(("spectrogram", plot_spectrogram))
    if PLOT_BETA_HILBERT and stim_available:
        panels.append(("beta_hilbert", plot_beta_hilbert))
    if PLOT_BAND_SPECTRUM:
        panels.append(("band_spectrum", plot_band_spectrum))
    if PLOT_PULSOGRAM and stim_available:
        panels.append(("pulsogram", plot_pulsogram))
    return panels


def plot_summary(summary: dict, save_path: Path | None = None):
    panels = build_panels(summary)
    if not panels:
        print("[SKIP] no active plot toggles")
        return

    n_rows, n_cols = choose_grid(len(panels))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    mode_label = "block mean" if ONLY_TRIAL is None else ONLY_TRIAL
    title = f"{summary.get('mouse')} | {summary.get('date')} | {summary.get('block')} | {mode_label}"
    for ax, (_, fn) in zip(axes, panels):
        fn(ax, summary)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    bottom = 0.10 if SHOW_FIGURE and ENABLE_SPECTROGRAM_CONTRAST_SLIDER else 0.03
    plt.tight_layout(rect=[0.0, bottom, 1.0, 0.98])
    add_spectrogram_contrast_slider(fig, axes)

    if SAVE_FIGURE and save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot summary panels from one final summary pickle.")
    parser.add_argument("--mouse", default=MOUSE_NAME)
    parser.add_argument("--date", default=SINGLE_DATE)
    parser.add_argument("--block", default=SINGLE_BLOCK)
    parser.add_argument("--summary-path", default=None, help="Optional direct path to *_summary.pkl")
    parser.add_argument("--save", action="store_true", help="Save figure next to summary pickle")
    parser.add_argument("--no-show", action="store_true", help="Build/save figure without showing it")
    args = parser.parse_args()

    global SHOW_FIGURE, SAVE_FIGURE
    SHOW_FIGURE = not args.no_show
    SAVE_FIGURE = bool(args.save or SAVE_FIGURE)

    summary_path = Path(args.summary_path) if args.summary_path else summary_path_from_parts(args.mouse, args.date, args.block)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary pickle not found: {summary_path}")

    summary = load_pickle(summary_path)
    out_path = summary_path.with_name(summary_path.stem + "_plot.png")
    plot_summary(summary, save_path=out_path)


if __name__ == "__main__":
    main()
