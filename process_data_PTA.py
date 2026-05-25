from pathlib import Path
import pickle
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_ANALYSIS_ROOT

MOUSE_NAME = "Jamie10"  # change to "Jamie5", etc.
SINGLE_DATE = "09-12-25"
SINGLE_BLOCK = "R2"


def single_mouse_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "all"}:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts[0] if len(parts) == 1 else None


def parse_mouse_names(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "all"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def discover_mouse_names() -> list[str]:
    out = []
    for mouse_dir in sorted(DATA_ANALYSIS_ROOT.iterdir()):
        if not mouse_dir.is_dir():
            continue
        if not (mouse_dir / "Imaging_Data").exists():
            continue
        out.append(mouse_dir.name)
    return out


def resolve_mouse_names(raw: str | None) -> list[str]:
    names = parse_mouse_names(raw)
    return names if names else discover_mouse_names()


_SINGLE_MOUSE_NAME = single_mouse_name(MOUSE_NAME)

# -------------------------
# INPUT (single-block mode)
# -------------------------
processed_pkl = None if SINGLE_BLOCK is None else (
    None if _SINGLE_MOUSE_NAME is None else (
        DATA_ANALYSIS_ROOT / _SINGLE_MOUSE_NAME / "Imaging_Data" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_traces_processed_notched.pkl"
    )
)
ephys_pkl = None if SINGLE_BLOCK is None else (
    None if _SINGLE_MOUSE_NAME is None else (
        DATA_ANALYSIS_ROOT / _SINGLE_MOUSE_NAME / "Open_Ephys" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_epoched_ephys.pkl"
    )
)

# -------------------------
# ROOTS (batch mode)
# -------------------------
IMAGING_ROOT = Path(
    DATA_ANALYSIS_ROOT / (_SINGLE_MOUSE_NAME or str(MOUSE_NAME)) / "Imaging_Data"
)
EPHYS_ROOT = Path(
    DATA_ANALYSIS_ROOT / (_SINGLE_MOUSE_NAME or str(MOUSE_NAME)) / "Open_Ephys"
)
BLOCK_RE = re.compile(r"^R\d+$")

# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = False
SAVE_OUTPUT = False
SHOW_PLOTS = True

# -------------------------
# SETTINGS
# -------------------------

SIGNAL_MODE = "notched"  # "notched_or_bleach_or_raw", "notched", "bleach", "raw"
PLOT_PRE_SEC = 0.010
PLOT_POST_SEC = 0.020
BASELINE_PRE_SEC = 0.500
FIXED_RESPONSE_WINDOW_SEC = 0.020
POST_WINDOW_MODE = "freq_aware"  #'fixed' or 'freq_aware'
POST_WINDOW_SCALE = 1.0       # Number of pulse periods to average in freq_aware mode
PERIOD_FRACTION = 0.95        # Use this fraction of each pulse period to avoid bleed into the next pulse
BASELINE_MODE = "median_pre"  # "none", "median_pre", "mean_pre"
MIN_SAMPLES_PER_SEGMENT = 8
MAX_PLOT_TRIALS = 10
INTERP_MODE = "linear"  # "linear" or "nearest"
SPREAD_MODE = "sd"  # "sem" or "sd"
LATENCY_THRESHOLD_SD = 1.0
LATENCY_MAX_PEAKS = 3
LATENCY_TOP_SAMPLES = 2
ONLY_TRIAL = None # e.g. "R1_1", "R1_8", "R1_10"; None = use all trials



def trial_sort_key(name: str):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def choose_signal(trial_dict: dict, mode: str) -> np.ndarray | None:
    raw = trial_dict.get("F_raw")
    bleach = trial_dict.get("F_bleach_corr")
    notched = trial_dict.get("F_notched")

    if mode == "raw":
        return None if raw is None else np.asarray(raw, dtype=float)
    if mode == "bleach":
        return None if bleach is None else np.asarray(bleach, dtype=float)
    if mode == "notched":
        return None if notched is None else np.asarray(notched, dtype=float)
    if mode == "notched_or_bleach_or_raw":
        if notched is not None:
            return np.asarray(notched, dtype=float)
        if bleach is not None:
            return np.asarray(bleach, dtype=float)
        return None if raw is None else np.asarray(raw, dtype=float)
    return None


def first_pulse_time_s(ephys_trial: dict) -> float | None:
    pulse_t = np.asarray(ephys_trial.get("stim_pulse_times_s", []), dtype=float)
    if len(pulse_t) == 0:
        return None
    return float(pulse_t[0])


def second_pulse_rel_time_s(ephys_trial: dict) -> float | None:
    pulse_t = np.asarray(ephys_trial.get("stim_pulse_times_s", []), dtype=float)
    if len(pulse_t) < 2:
        return None
    return float(pulse_t[1] - pulse_t[0])


def estimate_ipi_s(pulse_times: np.ndarray) -> float:
    if len(pulse_times) < 2:
        return np.nan
    isi = np.diff(pulse_times)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    if len(isi) == 0:
        return np.nan
    return float(np.median(isi))


def resolve_analysis_post_s(ephys_trial: dict) -> float:
    if POST_WINDOW_MODE != "freq_aware":
        return float(FIXED_RESPONSE_WINDOW_SEC)

    pulse_t = np.asarray(ephys_trial.get("stim_pulse_times_s", []), dtype=float)
    pulse_t = pulse_t[np.isfinite(pulse_t)]
    if len(pulse_t) < 2:
        return float(FIXED_RESPONSE_WINDOW_SEC)

    ipi = estimate_ipi_s(pulse_t)
    if not np.isfinite(ipi) or ipi <= 0:
        return float(FIXED_RESPONSE_WINDOW_SEC)

    frac = min(1.0, max(0.0, float(PERIOD_FRACTION)))
    n_periods = max(1, int(round(float(POST_WINDOW_SCALE))))
    return float(max(0.0, (n_periods - 1) * ipi + frac * ipi))


def compute_folded_response_waveform(
    t_rel: np.ndarray,
    x_corr: np.ndarray,
    ephys_trial: dict,
):
    if len(t_rel) < 2 or len(x_corr) != len(t_rel):
        return None, None

    dt = float(np.median(np.diff(t_rel)))
    if not np.isfinite(dt) or dt <= 0:
        return None, None

    if POST_WINDOW_MODE == "fixed":
        return None, None

    pulse_t = np.asarray(ephys_trial.get("stim_pulse_times_s", []), dtype=float)
    pulse_t = pulse_t[np.isfinite(pulse_t)]
    if len(pulse_t) < 2:
        return None, None

    ipi = estimate_ipi_s(pulse_t)
    if not np.isfinite(ipi) or ipi <= 0:
        return None, None

    frac = min(1.0, max(0.0, float(PERIOD_FRACTION)))
    if frac <= 0:
        return None, None

    n_periods = max(1, int(round(float(POST_WINDOW_SCALE))))
    if n_periods == 1:
        return None, None
    local_t = np.arange(0.0, frac * ipi, dt)
    if len(local_t) < 2:
        return None, None

    rows = []
    for k in range(n_periods):
        shifted_t = k * ipi + local_t
        if shifted_t[0] < t_rel[0] or shifted_t[-1] > t_rel[-1]:
            continue
        rows.append(np.interp(shifted_t, t_rel, x_corr))

    if not rows:
        return None, None

    local_y = np.nanmean(np.vstack(rows), axis=0)
    return local_t.astype(np.float64), local_y.astype(np.float64)


def compute_baseline_from_full_trace(
    x_full: np.ndarray,
    t_rel_full: np.ndarray,
    mode: str,
    baseline_pre_s: float,
) -> float:
    if mode == "none":
        return 0.0

    pre = x_full[(t_rel_full < 0) & (t_rel_full >= -baseline_pre_s)]
    if len(pre) == 0:
        return 0.0

    if mode == "median_pre":
        return float(np.median(pre))
    if mode == "mean_pre":
        return float(np.mean(pre))
    return float(np.median(pre))


def extract_first_pulse_segment(
    trial_name: str,
    img_trials: dict,
    e_trials: dict,
    plot_pre_s: float,
    plot_post_s: float,
    baseline_pre_s: float,
):
    if trial_name not in e_trials:
        return None, "missing_ephys_trial"

    td_img = img_trials[trial_name]
    td_e = e_trials[trial_name]

    sig = choose_signal(td_img, SIGNAL_MODE)
    if sig is None:
        return None, "missing_signal"

    t = np.asarray(td_img.get("t", []), dtype=float)
    if len(t) != len(sig):
        n = min(len(t), len(sig))
        t = t[:n]
        sig = sig[:n]
    if len(t) < MIN_SAMPLES_PER_SEGMENT:
        return None, "too_short_trace"

    t_pulse = first_pulse_time_s(td_e)
    if t_pulse is None:
        return None, "no_pulse"
    t_second_rel = second_pulse_rel_time_s(td_e)
    analysis_post_s = resolve_analysis_post_s(td_e)

    t_rel = t - t_pulse
    baseline = compute_baseline_from_full_trace(sig, t_rel, BASELINE_MODE, baseline_pre_s)
    x_corr_full = sig - baseline
    folded_t, folded_y = compute_folded_response_waveform(t_rel, x_corr_full, td_e)
    x_display_full = x_corr_full.copy()
    if folded_t is not None and folded_y is not None:
        m_fold = (t_rel >= folded_t[0]) & (t_rel <= folded_t[-1])
        if np.any(m_fold):
            x_display_full[m_fold] = np.interp(t_rel[m_fold], folded_t, folded_y)

    extract_pre_s = max(float(plot_pre_s), float(baseline_pre_s))
    extract_post_s = max(float(plot_post_s), float(analysis_post_s))

    keep = (t_rel >= -extract_pre_s) & (t_rel <= extract_post_s)
    n_keep = int(np.sum(keep))
    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
        expected = int(np.floor((extract_pre_s + extract_post_s) / dt)) + 1 if dt > 0 else MIN_SAMPLES_PER_SEGMENT
    else:
        expected = MIN_SAMPLES_PER_SEGMENT
    min_needed = min(MIN_SAMPLES_PER_SEGMENT, max(3, expected))
    if n_keep < min_needed:
        return None, "too_short_segment"

    t_seg = t_rel[keep]
    x_seg = x_display_full[keep]

    return (
        {
            "trial": trial_name,
            "t_rel_s": t_seg.astype(np.float64),
            "signal": x_seg.astype(np.float64),
            "pulse_time_s": float(t_pulse),
            "second_pulse_rel_s": t_second_rel,
            "analysis_post_s": float(analysis_post_s),
            "folded_window_s": (
                (float(folded_t[0]), float(folded_t[-1]))
                if folded_t is not None and len(folded_t) >= 2
                else None
            ),
            "baseline_pre_s": float(baseline_pre_s),
            "baseline_value": float(baseline),
        },
        None,
    )


def build_common_grid(segments: list[dict], plot_pre_s: float, plot_post_s: float) -> np.ndarray | None:
    dts = []
    for s in segments:
        tr = s["t_rel_s"]
        if len(tr) >= 2:
            dt = float(np.median(np.diff(tr)))
            if np.isfinite(dt) and dt > 0:
                dts.append(dt)
    if not dts:
        return None

    dt_med = float(np.median(dts))
    t0 = -float(plot_pre_s)
    t1 = float(plot_post_s)
    n = int(np.floor((t1 - t0) / dt_med)) + 1
    if n < MIN_SAMPLES_PER_SEGMENT:
        return None
    return t0 + np.arange(n) * dt_med


def interpolate_segments(segments: list[dict], t_grid: np.ndarray) -> np.ndarray:
    Y = np.full((len(segments), len(t_grid)), np.nan, dtype=float)
    for i, s in enumerate(segments):
        tx = s["t_rel_s"]
        yx = s["signal"]
        if len(tx) < 2:
            continue
        # Interpolate only where grid overlaps this trial segment.
        m = (t_grid >= tx[0]) & (t_grid <= tx[-1])
        if np.any(m):
            if INTERP_MODE == "nearest":
                tg = t_grid[m]
                idx = np.searchsorted(tx, tg, side="left")
                idx = np.clip(idx, 0, len(tx) - 1)
                left = np.maximum(idx - 1, 0)
                choose_left = np.abs(tg - tx[left]) <= np.abs(tx[idx] - tg)
                idx = np.where(choose_left, left, idx)
                Y[i, m] = yx[idx]
            else:
                Y[i, m] = np.interp(t_grid[m], tx, yx)
    return Y


def nansem(y: np.ndarray, axis=0):
    n = np.sum(np.isfinite(y), axis=axis)
    sd = np.nanstd(y, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = sd / np.sqrt(n)
    return se


def nansd(y: np.ndarray, axis=0):
    return np.nanstd(y, axis=axis, ddof=1)


def compute_peak_latency_events(
    t_rel: np.ndarray,
    y: np.ndarray,
    cycle_s: float,
    threshold_sd: float = LATENCY_THRESHOLD_SD,
    max_peaks: int = LATENCY_MAX_PEAKS,
    top_samples: int = LATENCY_TOP_SAMPLES,
) -> dict:
    t_rel = np.asarray(t_rel, dtype=float)
    y = np.asarray(y, dtype=float)
    out = {
        "threshold_sd": float(threshold_sd),
        "top_samples": int(top_samples),
        "search_end_s": float(cycle_s) if np.isfinite(cycle_s) else np.nan,
        "baseline_mean": np.nan,
        "baseline_sd": np.nan,
        "threshold": np.nan,
        "n_peak_events": 0,
        "events": [],
    }
    for i in range(1, int(max_peaks) + 1):
        out[f"peak_{i}_latency_ms"] = np.nan
        out[f"peak_{i}_amplitude"] = np.nan
        out[f"peak_{i}_event_start_ms"] = np.nan
        out[f"peak_{i}_event_end_ms"] = np.nan
        out[f"peak_{i}_n_samples"] = 0

    if len(t_rel) == 0 or y.shape != t_rel.shape or not np.isfinite(cycle_s) or cycle_s <= 0:
        return out

    baseline = y[(t_rel < 0) & np.isfinite(y)]
    if len(baseline) < 2:
        return out

    base_mean = float(np.nanmean(baseline))
    base_sd = float(np.nanstd(baseline, ddof=1))
    threshold = base_mean + float(threshold_sd) * base_sd
    search_end = min(float(cycle_s), float(np.nanmax(t_rel)))
    out.update({
        "search_end_s": search_end,
        "baseline_mean": base_mean,
        "baseline_sd": base_sd,
        "threshold": float(threshold),
    })

    above = np.where((t_rel >= 0) & (t_rel <= search_end) & np.isfinite(y) & (y >= threshold))[0]
    if len(above) == 0:
        return out

    splits = np.where(np.diff(above) > 1)[0] + 1
    groups = np.split(above, splits)
    out["n_peak_events"] = int(len(groups))
    events = []
    for group in groups[: int(max_peaks)]:
        if len(group) == 0:
            continue
        n_top = min(int(top_samples), len(group))
        top_idx = group[np.argsort(y[group])[-n_top:]]
        latency_s = float(np.nanmean(t_rel[top_idx]))
        event = {
            "latency_ms": 1000.0 * latency_s,
            "amplitude": float(np.nanmax(y[group])),
            "event_start_ms": 1000.0 * float(t_rel[group[0]]),
            "event_end_ms": 1000.0 * float(t_rel[group[-1]]),
            "n_samples": int(len(group)),
        }
        events.append(event)

    out["events"] = events
    for i, event in enumerate(events, start=1):
        out[f"peak_{i}_latency_ms"] = float(event["latency_ms"])
        out[f"peak_{i}_amplitude"] = float(event["amplitude"])
        out[f"peak_{i}_event_start_ms"] = float(event["event_start_ms"])
        out[f"peak_{i}_event_end_ms"] = float(event["event_end_ms"])
        out[f"peak_{i}_n_samples"] = int(event["n_samples"])
    return out


def summarize_latency_across_segments(segments: list[dict]) -> dict:
    out = {}
    for i in range(1, LATENCY_MAX_PEAKS + 1):
        vals = np.asarray(
            [safe_float(s.get("latency", {}).get(f"peak_{i}_latency_ms")) for s in segments],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        out[f"peak_{i}_latency_ms_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
        out[f"peak_{i}_latency_ms_median"] = float(np.nanmedian(vals)) if len(vals) else np.nan
        out[f"peak_{i}_jitter_ms"] = float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else np.nan
        out[f"peak_{i}_n_latency_trials"] = int(len(vals))
    return out


def rebuild_first_pulse_display_from_segments(segments: list[dict], post_periods: float = 3.0):
    second_times = [
        float(s["second_pulse_rel_s"])
        for s in segments
        if s.get("second_pulse_rel_s") is not None and np.isfinite(s["second_pulse_rel_s"]) and float(s["second_pulse_rel_s"]) > 0
    ]
    if not second_times:
        return None, None, None, []

    pre_s = float(np.median(second_times))
    post_s = float(post_periods) * pre_s

    dts = []
    for s in segments:
        tx = np.asarray(s.get("t_rel_s", []), dtype=float)
        if len(tx) < 2:
            continue
        dt = np.diff(tx)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            dts.append(float(np.median(dt)))
    if not dts:
        return None, None, None, second_times

    dt = float(np.median(np.asarray(dts, dtype=float)))
    t_plot = np.arange(-pre_s, post_s + 0.5 * dt, dt, dtype=float)
    rows = []
    for s in segments:
        tx = np.asarray(s.get("t_rel_s", []), dtype=float)
        yx = np.asarray(s.get("signal", []), dtype=float)
        if len(tx) == 0 or yx.shape != tx.shape:
            continue
        rows.append(np.interp(t_plot, tx, yx, left=np.nan, right=np.nan))
    if not rows:
        return None, None, None, second_times

    Y_plot = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_mean = np.nanmean(Y_plot, axis=0)
        if SPREAD_MODE == "sd":
            y_spread = np.nanstd(Y_plot, axis=0, ddof=1) if Y_plot.shape[0] >= 2 else np.full_like(y_mean, np.nan)
        else:
            y_spread = nansem(Y_plot, axis=0)
    return t_plot, y_mean, y_spread, second_times


def extract_first_pulse_display_segment(
    trial_name: str,
    img_trials: dict,
    e_trials: dict,
    post_periods: float = 3.0,
):
    if trial_name not in e_trials or trial_name not in img_trials:
        return None

    td_img = img_trials[trial_name]
    td_e = e_trials[trial_name]

    sig = choose_signal(td_img, SIGNAL_MODE)
    if sig is None:
        return None

    t = np.asarray(td_img.get("t", []), dtype=float)
    sig = np.asarray(sig, dtype=float)
    n = min(len(t), len(sig))
    t = t[:n]
    sig = sig[:n]
    if len(t) < MIN_SAMPLES_PER_SEGMENT:
        return None

    t_pulse = first_pulse_time_s(td_e)
    if t_pulse is None:
        return None

    t_second_rel = second_pulse_rel_time_s(td_e)
    if t_second_rel is None or not np.isfinite(t_second_rel) or float(t_second_rel) <= 0:
        pulse_t = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        ipi_s = estimate_ipi_s(pulse_t)
        if not np.isfinite(ipi_s) or ipi_s <= 0:
            return None
        t_second_rel = float(ipi_s)

    t_rel = t - float(t_pulse)
    baseline = compute_baseline_from_full_trace(sig, t_rel, BASELINE_MODE, BASELINE_PRE_SEC)
    x_corr = sig - baseline

    pre_s = float(t_second_rel)
    post_s = float(post_periods) * float(t_second_rel)
    keep = (t_rel >= -pre_s) & (t_rel <= post_s)
    if int(np.sum(keep)) < MIN_SAMPLES_PER_SEGMENT:
        return None

    return {
        "trial": trial_name,
        "t_rel_s": np.asarray(t_rel[keep], dtype=np.float64),
        "signal": np.asarray(x_corr[keep], dtype=np.float64),
        "second_pulse_rel_s": float(t_second_rel),
    }


def build_first_pulse_display_from_trials(
    img_trials: dict,
    e_trials: dict,
    trial_names: list[str],
    post_periods: float = 3.0,
):
    segments = []
    second_times = []
    dts = []
    for name in trial_names:
        seg = extract_first_pulse_display_segment(name, img_trials, e_trials, post_periods=post_periods)
        if seg is None:
            continue
        tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
        yx = np.asarray(seg.get("signal", []), dtype=float)
        if len(tx) < 2 or yx.shape != tx.shape:
            continue
        segments.append(seg)
        second_t = safe_float(seg.get("second_pulse_rel_s"))
        if np.isfinite(second_t) and second_t > 0:
            second_times.append(float(second_t))
        dt = np.diff(tx)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            dts.append(float(np.median(dt)))

    if not segments or not second_times or not dts:
        return None, None

    pre_s = float(np.nanmedian(second_times))
    dt = float(np.nanmedian(dts))
    if not np.isfinite(pre_s) or pre_s <= 0 or not np.isfinite(dt) or dt <= 0:
        return None, None

    t_grid = np.arange(-pre_s, float(post_periods) * pre_s + 0.5 * dt, dt, dtype=float)
    rows = []
    for seg in segments:
        tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
        yx = np.asarray(seg.get("signal", []), dtype=float)
        rows.append(np.interp(t_grid, tx, yx, left=np.nan, right=np.nan))
    Y = np.vstack(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_mean = np.nanmean(Y, axis=0)
        y_sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.full_like(y_mean, np.nan)
        y_sem = nansem(Y, axis=0) if Y.shape[0] >= 2 else np.full_like(y_mean, np.nan)

    return (
        {
            "t_rel_s": np.asarray(t_grid, dtype=np.float64),
            "mean": np.asarray(y_mean, dtype=np.float64),
            "sd": np.asarray(y_sd, dtype=np.float64),
            "sem": np.asarray(y_sem, dtype=np.float64),
            "second_pulse_rel_s_mean": float(np.nanmedian(second_times)),
            "post_periods": float(post_periods),
        },
        segments,
    )


def plot_first_pulse_pta(
    block_label: str,
    segments: list[dict],
    t_grid: np.ndarray,
    Y: np.ndarray,
    display_summary: dict | None = None,
    display_segments: list[dict] | None = None,
):
    if display_summary is not None:
        t_plot = np.asarray(display_summary.get("t_rel_s", []), dtype=float)
        y_mean = np.asarray(display_summary.get("mean", []), dtype=float)
        y_spread = np.asarray(display_summary.get("sd" if SPREAD_MODE == "sd" else "sem", []), dtype=float)
        second_val = safe_float(display_summary.get("second_pulse_rel_s_mean"))
        second_times = [second_val] if np.isfinite(second_val) and second_val > 0 else []
        plot_segments = display_segments if display_segments is not None else segments
    else:
        display = rebuild_first_pulse_display_from_segments(segments, post_periods=3.0)
        plot_segments = segments
        if display[0] is not None:
            t_plot, y_mean, y_spread, second_times = display
        else:
            t_plot = t_grid
            y_mean = np.nanmean(Y, axis=0)
            if SPREAD_MODE == "sd":
                y_spread = nansd(Y, axis=0)
            else:
                y_spread = nansem(Y, axis=0)
            second_times = [
                float(s["second_pulse_rel_s"])
                for s in segments
                if s.get("second_pulse_rel_s") is not None and np.isfinite(s["second_pulse_rel_s"])
            ]

    if len(t_plot) == 0 or y_mean.shape != t_plot.shape:
        t_plot = t_grid
        y_mean = np.nanmean(Y, axis=0)
        y_spread = nansd(Y, axis=0) if SPREAD_MODE == "sd" else nansem(Y, axis=0)
        second_times = [
            float(s["second_pulse_rel_s"])
            for s in segments
            if s.get("second_pulse_rel_s") is not None and np.isfinite(s["second_pulse_rel_s"])
        ]
        plot_segments = segments
    if SPREAD_MODE == "sd":
        spread_label = "SD"
    else:
        spread_label = "SEM"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    n_plot = min(MAX_PLOT_TRIALS, len(plot_segments))
    for i in range(n_plot):
        tx = plot_segments[i]["t_rel_s"]
        yx = plot_segments[i]["signal"]
        m = (tx >= t_plot[0]) & (tx <= t_plot[-1])
        if np.any(m):
            ax.plot(tx[m], yx[m], lw=0.8, alpha=0.25, color="tab:blue")

    ax.plot(t_plot, y_mean, color="black", lw=2.0, label="mean")
    ax.fill_between(t_plot, y_mean - y_spread, y_mean + y_spread, color="black", alpha=0.2, label=spread_label)
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.2, label="first pulse")
    if second_times:
        second_t = float(np.median(second_times))
        if t_plot[0] <= second_t <= t_plot[-1]:
            ax.axvline(second_t, color="tab:orange", ls="--", lw=1.2, label="second pulse")
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    ax.set_title(f"First-Pulse PTA | {block_label} | n={len(segments)}")
    ax.set_xlabel("time from first pulse (s)")
    ax.set_ylabel("signal (a.u. baseline-corrected)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def run_single(imaging_path: Path, ephys_path: Path):
    with open(imaging_path, "rb") as f:
        img = pickle.load(f)
    with open(ephys_path, "rb") as f:
        eph = pickle.load(f)

    img_trials = img.get("trials", {})
    e_trials = eph.get("trials", {})
    if not img_trials or not e_trials:
        print(f"[SKIP] missing trials in {imaging_path.name}")
        return

    names = sorted(img_trials.keys(), key=trial_sort_key)
    segments = []
    fail_counts = {}

    for name in names:
        if ONLY_TRIAL is not None and name != ONLY_TRIAL:
            continue
        seg, err = extract_first_pulse_segment(name, img_trials, e_trials, PLOT_PRE_SEC, PLOT_POST_SEC, BASELINE_PRE_SEC)
        if seg is None:
            fail_counts[err] = fail_counts.get(err, 0) + 1
            continue
        segments.append(seg)

    if not segments:
        print(f"[SKIP] no valid first-pulse segments: {imaging_path}")
        if fail_counts:
            print(f"[INFO] fail reasons: {fail_counts}")
        return

    t_grid = build_common_grid(segments, PLOT_PRE_SEC, PLOT_POST_SEC)
    if t_grid is None:
        print(f"[SKIP] could not build interpolation grid: {imaging_path}")
        return

    Y = interpolate_segments(segments, t_grid)
    y_mean = np.nanmean(Y, axis=0)
    y_sem = nansem(Y, axis=0)
    y_spread = nansd(Y, axis=0) if SPREAD_MODE == "sd" else y_sem
    n_valid = int(np.sum(np.isfinite(Y[:, np.argmin(np.abs(t_grid))])))
    display_summary, display_segments = build_first_pulse_display_from_trials(
        img_trials,
        e_trials,
        [s["trial"] for s in segments],
        post_periods=3.0,
    )
    for seg in segments:
        cycle_s = safe_float(seg.get("second_pulse_rel_s"))
        seg["latency"] = compute_peak_latency_events(seg.get("t_rel_s", []), seg.get("signal", []), cycle_s)

    if display_summary is not None:
        latency_t = np.asarray(display_summary.get("t_rel_s", []), dtype=float)
        latency_y = np.asarray(display_summary.get("mean", []), dtype=float)
        latency_cycle_s = safe_float(display_summary.get("second_pulse_rel_s_mean"))
    else:
        latency_t = t_grid
        latency_y = y_mean
        latency_cycle_s = np.nanmedian([
            safe_float(s.get("second_pulse_rel_s"))
            for s in segments
            if np.isfinite(safe_float(s.get("second_pulse_rel_s")))
        ])
    latency = compute_peak_latency_events(latency_t, latency_y, latency_cycle_s)
    latency_jitter = summarize_latency_across_segments(segments)

    block_label = f"{img.get('date')} {img.get('block')}"
    print(f"[RUN] {block_label} | {imaging_path.name}")
    print(f"[INFO] PTA trials used: {len(segments)} / {len(names)} | n@pulse={n_valid}")
    if fail_counts:
        print(f"[INFO] skipped: {fail_counts}")

    if SHOW_PLOTS:
        plot_first_pulse_pta(
            block_label,
            segments,
            t_grid,
            Y,
            display_summary=display_summary,
            display_segments=display_segments,
        )
    if SAVE_OUTPUT:
        out_path = imaging_path.parent / f"{imaging_path.stem}_pta_first_pulse.pkl"
        out = {
            "mouse": img.get("mouse"),
            "date": img.get("date"),
            "block": img.get("block"),
            "analysis": "first_pulse_pta",
            "settings": {
                "signal_mode": SIGNAL_MODE,
                "plot_pre_sec": float(PLOT_PRE_SEC),
                "plot_post_sec": float(PLOT_POST_SEC),
                "baseline_pre_sec": float(BASELINE_PRE_SEC),
                "fixed_response_window_sec": float(FIXED_RESPONSE_WINDOW_SEC),
                "post_window_mode": POST_WINDOW_MODE,
                "post_window_scale": float(POST_WINDOW_SCALE),
                "period_fraction": float(PERIOD_FRACTION),
                "baseline_mode": BASELINE_MODE,
                "interp_mode": INTERP_MODE,
                "min_samples_per_segment": int(MIN_SAMPLES_PER_SEGMENT),
                "spread_mode": SPREAD_MODE,
                "latency_threshold_sd": float(LATENCY_THRESHOLD_SD),
                "latency_max_peaks": int(LATENCY_MAX_PEAKS),
                "latency_top_samples": int(LATENCY_TOP_SAMPLES),
                "only_trial": ONLY_TRIAL,
            },
            "trial_names_used": [s["trial"] for s in segments],
            "t_rel_s": t_grid.astype(np.float64),
            "pta_mean": y_mean.astype(np.float64),
            "pta_sem": y_sem.astype(np.float64),
            "pta_spread": y_spread.astype(np.float64),
            "segments": segments,
            "latency": latency,
            "latency_jitter": latency_jitter,
            "display": {} if display_summary is None else display_summary,
        }
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] {out_path}")


def run_batch(imaging_root: Path, ephys_root: Path):
    for date_dir in sorted(imaging_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
                continue
            block = block_dir.name
            img_p = block_dir / f"{block}_traces_processed_notched.pkl"
            eph_p = ephys_root / date_dir.name / block / f"{block}_epoched_ephys.pkl"
            if not img_p.exists() or not eph_p.exists():
                continue
            run_single(img_p, eph_p)


def run_single_date(imaging_root: Path, ephys_root: Path, date_name: str):
    date_dir = imaging_root / date_name
    if not date_dir.exists():
        print(f"[WARN] date folder not found: {date_dir}")
        return

    for block_dir in sorted(date_dir.iterdir()):
        if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
            continue
        block = block_dir.name
        img_p = block_dir / f"{block}_traces_processed_notched.pkl"
        eph_p = ephys_root / date_name / block / f"{block}_epoched_ephys.pkl"
        if not img_p.exists() or not eph_p.exists():
            continue
        run_single(img_p, eph_p)


def main() -> None:
    mouse_names = resolve_mouse_names(MOUSE_NAME)
    if not mouse_names:
        print("No mice found to process.")
        return

    if RUN_BATCH:
        for mouse_name in mouse_names:
            run_batch(
                DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data",
                DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys",
            )
    elif SINGLE_DATE is not None and SINGLE_BLOCK is None:
        for mouse_name in mouse_names:
            run_single_date(
                DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data",
                DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys",
                SINGLE_DATE,
            )
    else:
        if SINGLE_DATE is None or SINGLE_BLOCK is None:
            print("Set SINGLE_DATE to run one date, or set both SINGLE_DATE and SINGLE_BLOCK to run one block.")
            return
        for mouse_name in mouse_names:
            single_processed = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_traces_processed_notched.pkl"
            single_ephys = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_epoched_ephys.pkl"
            if not single_processed.exists() or not single_ephys.exists():
                print(f"[SKIP] missing single-block inputs for {mouse_name} | {SINGLE_DATE} | {SINGLE_BLOCK}")
                continue
            run_single(single_processed, single_ephys)


if __name__ == "__main__":
    main()





