from __future__ import annotations

import argparse
import pickle
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import spectrogram
from scipy.stats import wilcoxon

from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = None
SINGLE_DATE = None
SINGLE_BLOCK = None


# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = True
SAVE_OUTPUT = True
OVERWRITE = True


# -------------------------
# OUTPUT
# -------------------------
OUTPUT_SUFFIX = "_summary.pkl"
NFFT_SPEC = 512
SPECTROGRAM_WINDOW = "hann"
SPECTROGRAM_OVERLAP_FRAC = 0.95
DEFAULT_SPEC_REL_BASELINE_END_S = -0.5
DEFAULT_SPEC_REL_BASELINE_STAT = "median"
DEFAULT_DISPLAY_POST_PERIODS = 3.0
LFP_DISPLAY_MAX_POINTS = 12000
DEFAULT_HILBERT_REL_BASELINE_END_S = -0.5
DEFAULT_HILBERT_REL_BASELINE_STAT = "median"
HILBERT_DISPLAY_MAX_POINTS = 20000


BLOCK_RE = re.compile(r"^R\d+$")


def parse_mouse_names(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"none", "all"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def discover_mouse_names() -> list[str]:
    names = []
    for child in sorted(DATA_ANALYSIS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if (child / "Imaging_Data").exists() or (child / "Open_Ephys").exists():
            names.append(child.name)
    return names


def resolve_mouse_names(raw: str | None) -> list[str]:
    names = parse_mouse_names(raw)
    return names if names else discover_mouse_names()


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


def plv_section_sort_key(key: str) -> int:
    if key == "plv":
        return 1
    m = re.fullmatch(r"plv_h(\d+)", str(key))
    return int(m.group(1)) if m else 10**9


def is_plv_section_key(key: str) -> bool:
    return key == "plv" or re.fullmatch(r"plv_h\d+", str(key)) is not None


def hilbert_section_sort_key(key: str) -> int:
    if key == "signal_hilbert":
        return 1
    m = re.fullmatch(r"signal_hilbert_h(\d+)", str(key))
    return int(m.group(1)) if m else 10**9


def is_signal_hilbert_section_key(key: str) -> bool:
    return key == "signal_hilbert" or re.fullmatch(r"signal_hilbert_h\d+", str(key)) is not None


def same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(a, b, equal_nan=True)


def nanmean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=float)
    return np.nanmean(np.vstack(arrays), axis=0)


def nansd_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=float)
    arr = np.vstack(arrays)
    if arr.shape[0] < 2:
        return np.full(arr.shape[1], np.nan, dtype=float)
    return np.nanstd(arr, axis=0, ddof=1)


def nansem_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=float)
    arr = np.vstack(arrays)
    if arr.shape[0] < 2:
        return np.full(arr.shape[1], np.nan, dtype=float)
    return np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])


def interpolate_to_ref_grid(x_ref: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_ref = np.asarray(x_ref, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(len(x_ref), np.nan, dtype=float)
    if len(x_ref) == 0 or len(x) < 2 or y.shape != x.shape:
        return out
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        return out
    keep = (x_ref >= x[0]) & (x_ref <= x[-1])
    if np.any(keep):
        out[keep] = np.interp(x_ref[keep], x, y)
    return out


def estimate_fs(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return float(1.0 / np.median(dt))


def db_from_linear(power_linear: np.ndarray) -> np.ndarray:
    power_linear = np.asarray(power_linear, dtype=float)
    if power_linear.size == 0:
        return np.asarray(power_linear, dtype=float)
    return np.asarray(10.0 * np.log10(np.maximum(power_linear, 1e-30)), dtype=np.float64)


def linear_from_db(power_db: np.ndarray) -> np.ndarray:
    power_db = np.asarray(power_db, dtype=float)
    if power_db.size == 0:
        return np.asarray(power_db, dtype=float)
    return np.asarray(np.power(10.0, power_db / 10.0), dtype=np.float64)


def spectrogram_linear_power(sec: dict, linear_key: str = "power_linear", db_key: str = "power_db") -> np.ndarray:
    p_linear = np.asarray(sec.get(linear_key, []), dtype=float)
    if p_linear.ndim == 2:
        return p_linear

    p_db = np.asarray(sec.get(db_key, []), dtype=float)
    if p_db.ndim == 2:
        return np.asarray(np.power(10.0, p_db / 10.0), dtype=np.float64)

    return np.array([], dtype=float)


def compute_relative_spectrogram(
    time_s: np.ndarray,
    power_linear: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
    baseline_stat: str,
) -> dict:
    time_s = np.asarray(time_s, dtype=float)
    power_linear = np.asarray(power_linear, dtype=float)
    if power_linear.ndim != 2 or len(time_s) == 0 or power_linear.shape[1] != len(time_s):
        return {
            "power_linear": np.array([], dtype=float),
            "power_db": np.array([], dtype=float),
            "baseline_start_s": np.nan,
            "baseline_end_s": np.nan,
            "baseline_stat": baseline_stat,
        }

    m_base = (time_s >= float(baseline_start_s)) & (time_s <= float(baseline_end_s))
    if not np.any(m_base):
        m_base = time_s < 0.0
    if not np.any(m_base):
        m_base = np.isfinite(time_s)
    if not np.any(m_base):
        return {
            "power_linear": np.array([], dtype=float),
            "power_db": np.array([], dtype=float),
            "baseline_start_s": np.nan,
            "baseline_end_s": np.nan,
            "baseline_stat": baseline_stat,
        }

    if str(baseline_stat).lower() == "mean":
        baseline = np.nanmean(power_linear[:, m_base], axis=1, keepdims=True)
    else:
        baseline = np.nanmedian(power_linear[:, m_base], axis=1, keepdims=True)

    baseline = np.where(np.isfinite(baseline) & (baseline > 0), baseline, np.nan)
    rel_linear = power_linear / baseline
    idx = np.flatnonzero(m_base)
    return {
        "power_linear": np.asarray(rel_linear, dtype=np.float64),
        "power_db": db_from_linear(rel_linear),
        "baseline_start_s": float(time_s[idx[0]]),
        "baseline_end_s": float(time_s[idx[-1]]),
        "baseline_stat": baseline_stat,
    }


def compute_relative_curve(
    time_s: np.ndarray,
    values: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
    baseline_stat: str,
) -> dict:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(time_s) == 0 or values.shape != time_s.shape:
        return {
            "relative": np.array([], dtype=float),
            "baseline_start_s": np.nan,
            "baseline_end_s": np.nan,
            "baseline_stat": baseline_stat,
        }

    m_base = (time_s >= float(baseline_start_s)) & (time_s <= float(baseline_end_s))
    if not np.any(m_base):
        m_base = time_s < 0.0
    if not np.any(m_base):
        m_base = np.isfinite(time_s)
    if not np.any(m_base):
        return {
            "relative": np.array([], dtype=float),
            "baseline_start_s": np.nan,
            "baseline_end_s": np.nan,
            "baseline_stat": baseline_stat,
        }

    if str(baseline_stat).lower() == "mean":
        baseline = float(np.nanmean(values[m_base]))
    else:
        baseline = float(np.nanmedian(values[m_base]))
    if not np.isfinite(baseline) or baseline <= 0:
        rel = np.full_like(values, np.nan, dtype=float)
    else:
        rel = values / baseline
    idx = np.flatnonzero(m_base)
    return {
        "relative": np.asarray(rel, dtype=np.float64),
        "baseline_start_s": float(time_s[idx[0]]),
        "baseline_end_s": float(time_s[idx[-1]]),
        "baseline_stat": baseline_stat,
    }


def empty_train_spectrogram_summary() -> dict:
    return {
        "time_s": np.array([], dtype=float),
        "freq_hz": np.array([], dtype=float),
        "trial_names_used": [],
        "n_trials_used": 0,
        "n_trials_per_bin": np.array([], dtype=float),
        "baseline_start_s": np.nan,
        "baseline_end_s": np.nan,
        "baseline_stat": DEFAULT_SPEC_REL_BASELINE_STAT,
        "power_linear_mean": np.array([], dtype=float),
        "power_linear_sd": np.array([], dtype=float),
        "power_db_mean": np.array([], dtype=float),
        "power_db_sd": np.array([], dtype=float),
        "relative_linear_mean": np.array([], dtype=float),
        "relative_db_mean": np.array([], dtype=float),
    }


def build_common_spectrogram_summary(
    trial_results: dict,
    section_key: str = "spectrogram",
    baseline_end_s: float = DEFAULT_SPEC_REL_BASELINE_END_S,
    baseline_stat: str = DEFAULT_SPEC_REL_BASELINE_STAT,
) -> dict:
    specs = []
    for name in sorted(trial_results.keys()):
        sec = trial_results[name].get(section_key, {})
        time_s = np.asarray(sec.get("time_s", []), dtype=float)
        freq_hz = np.asarray(sec.get("freq_hz", []), dtype=float)
        power_linear = spectrogram_linear_power(sec)
        if len(time_s) == 0 or len(freq_hz) == 0 or power_linear.shape != (len(freq_hz), len(time_s)):
            continue
        specs.append({
            "name": name,
            "time_s": time_s,
            "freq_hz": freq_hz,
            "power_linear": power_linear,
        })

    if not specs:
        return empty_train_spectrogram_summary()

    t_lo = max(float(np.min(s["time_s"])) for s in specs)
    t_hi = min(float(np.max(s["time_s"])) for s in specs)
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        return empty_train_spectrogram_summary()

    dt_vals = []
    df_vals = []
    for s in specs:
        dt = np.diff(s["time_s"])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            dt_vals.append(float(np.median(dt)))

        df = np.diff(s["freq_hz"])
        df = df[np.isfinite(df) & (df > 0)]
        if df.size:
            df_vals.append(float(np.median(df)))

    if not dt_vals or not df_vals:
        return empty_train_spectrogram_summary()

    dt_ref = float(np.median(dt_vals))
    df_ref = float(np.median(df_vals))
    f_lo = max(float(np.min(s["freq_hz"])) for s in specs)
    f_hi = min(float(np.max(s["freq_hz"])) for s in specs)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= f_lo:
        return empty_train_spectrogram_summary()

    time_ref = np.arange(t_lo, t_hi + 0.5 * dt_ref, dt_ref, dtype=float)
    freq_ref = np.arange(f_lo, f_hi + 0.5 * df_ref, df_ref, dtype=float)
    if len(time_ref) < 2 or len(freq_ref) < 2:
        return empty_train_spectrogram_summary()

    FF, TT = np.meshgrid(freq_ref, time_ref, indexing="ij")
    interp_points = np.column_stack([FF.ravel(), TT.ravel()])
    stack_linear = []
    used_names = []
    for s in specs:
        interp = RegularGridInterpolator(
            (s["freq_hz"], s["time_s"]),
            s["power_linear"],
            bounds_error=False,
            fill_value=np.nan,
        )
        out = interp(interp_points).reshape(len(freq_ref), len(time_ref))
        if not np.any(np.isfinite(out)):
            continue
        stack_linear.append(out)
        used_names.append(s["name"])

    if not stack_linear:
        return empty_train_spectrogram_summary()

    stack_linear = np.stack(stack_linear, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_linear = np.nanmean(stack_linear, axis=0)
        sd_linear = (
            np.nanstd(stack_linear, axis=0, ddof=1)
            if stack_linear.shape[0] >= 2
            else np.full((len(freq_ref), len(time_ref)), np.nan, dtype=float)
        )
        stack_db = db_from_linear(stack_linear)
        sd_db = (
            np.nanstd(stack_db, axis=0, ddof=1)
            if stack_db.shape[0] >= 2
            else np.full((len(freq_ref), len(time_ref)), np.nan, dtype=float)
        )

    n_trials_per_bin = np.sum(np.isfinite(stack_linear), axis=0).astype(np.float64)
    rel = compute_relative_spectrogram(
        time_s=time_ref,
        power_linear=mean_linear,
        baseline_start_s=float(time_ref[0]),
        baseline_end_s=float(baseline_end_s),
        baseline_stat=baseline_stat,
    )

    return {
        "time_s": np.asarray(time_ref, dtype=np.float64),
        "freq_hz": np.asarray(freq_ref, dtype=np.float64),
        "trial_names_used": used_names,
        "n_trials_used": int(len(used_names)),
        "n_trials_per_bin": n_trials_per_bin,
        "baseline_start_s": safe_float(rel.get("baseline_start_s")),
        "baseline_end_s": safe_float(rel.get("baseline_end_s")),
        "baseline_stat": rel.get("baseline_stat", baseline_stat),
        "power_linear_mean": np.asarray(mean_linear, dtype=np.float64),
        "power_linear_sd": np.asarray(sd_linear, dtype=np.float64),
        "power_db_mean": db_from_linear(mean_linear),
        "power_db_sd": np.asarray(sd_db, dtype=np.float64),
        "relative_linear_mean": np.asarray(rel.get("power_linear", []), dtype=np.float64),
        "relative_db_mean": np.asarray(rel.get("power_db", []), dtype=np.float64),
    }


def empty_hilbert_summary() -> dict:
    return {
        "time_s_full": np.array([], dtype=float),
        "amplitude_mean_full": np.array([], dtype=float),
        "amplitude_sd_full": np.array([], dtype=float),
        "amplitude_sem_full": np.array([], dtype=float),
        "relative_mean_full": np.array([], dtype=float),
        "relative_sd_full": np.array([], dtype=float),
        "relative_sem_full": np.array([], dtype=float),
        "time_s_display": np.array([], dtype=float),
        "amplitude_mean_display": np.array([], dtype=float),
        "amplitude_sd_display": np.array([], dtype=float),
        "amplitude_sem_display": np.array([], dtype=float),
        "relative_mean_display": np.array([], dtype=float),
        "relative_sd_display": np.array([], dtype=float),
        "relative_sem_display": np.array([], dtype=float),
        "trial_names_used": [],
        "n_trials_used": 0,
        "baseline_start_s": np.nan,
        "baseline_end_s": np.nan,
        "baseline_stat": DEFAULT_HILBERT_REL_BASELINE_STAT,
        "display_stride": 1,
        "f_center_hz_mean": np.nan,
        "band_hz_mean": np.array([], dtype=float),
    }


def build_common_hilbert_summary(
    trial_results: dict,
    section_key: str,
    baseline_end_s: float = DEFAULT_HILBERT_REL_BASELINE_END_S,
    baseline_stat: str = DEFAULT_HILBERT_REL_BASELINE_STAT,
) -> dict:
    items = []
    for name in sorted(trial_results.keys()):
        sec = trial_results[name].get(section_key, {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        y = np.asarray(sec.get("amplitude", []), dtype=float)
        if len(t) < 2 or y.shape != t.shape:
            continue
        items.append({
            "name": name,
            "time_s": t,
            "amplitude": y,
            "f_center_hz": safe_float(sec.get("f_center_hz")),
            "band_hz": np.asarray(sec.get("band_hz", []), dtype=float),
        })

    if not items:
        return empty_hilbert_summary()

    t_ref = build_common_axis_1d([item["time_s"] for item in items])
    if t_ref is None:
        return empty_hilbert_summary()

    abs_rows = []
    rel_rows = []
    used_names = []
    baseline_start_final = np.nan
    baseline_end_final = np.nan

    for item in items:
        y_abs = interpolate_to_ref_grid(t_ref, item["time_s"], item["amplitude"])
        rel = compute_relative_curve(
            time_s=t_ref,
            values=y_abs,
            baseline_start_s=float(t_ref[0]),
            baseline_end_s=float(baseline_end_s),
            baseline_stat=baseline_stat,
        )
        y_rel = np.asarray(rel.get("relative", []), dtype=float)
        if y_rel.shape != t_ref.shape:
            continue
        abs_rows.append(y_abs)
        rel_rows.append(y_rel)
        used_names.append(item["name"])
        if not np.isfinite(baseline_start_final):
            baseline_start_final = safe_float(rel.get("baseline_start_s"))
        if not np.isfinite(baseline_end_final):
            baseline_end_final = safe_float(rel.get("baseline_end_s"))

    if not abs_rows:
        return empty_hilbert_summary()

    abs_mean = nanmean_stack(abs_rows)
    abs_sd = nansd_stack(abs_rows)
    abs_sem = nansem_stack(abs_rows)
    rel_mean = nanmean_stack(rel_rows)
    rel_sd = nansd_stack(rel_rows)
    rel_sem = nansem_stack(rel_rows)
    t_disp, disp_arrays, stride = decimate_curve_bundle(
        t_ref,
        [abs_mean, abs_sd, abs_sem, rel_mean, rel_sd, rel_sem],
        max_points=HILBERT_DISPLAY_MAX_POINTS,
    )
    f_vals = [item["f_center_hz"] for item in items if np.isfinite(item["f_center_hz"])]
    band_rows = [item["band_hz"] for item in items if item["band_hz"].shape == (2,)]
    band_mean = np.nanmean(np.vstack(band_rows), axis=0) if band_rows else np.array([], dtype=float)

    return {
        "time_s_full": np.asarray(t_ref, dtype=np.float64),
        "amplitude_mean_full": np.asarray(abs_mean, dtype=np.float64),
        "amplitude_sd_full": np.asarray(abs_sd, dtype=np.float64),
        "amplitude_sem_full": np.asarray(abs_sem, dtype=np.float64),
        "relative_mean_full": np.asarray(rel_mean, dtype=np.float64),
        "relative_sd_full": np.asarray(rel_sd, dtype=np.float64),
        "relative_sem_full": np.asarray(rel_sem, dtype=np.float64),
        "time_s_display": np.asarray(t_disp, dtype=np.float64),
        "amplitude_mean_display": np.asarray(disp_arrays[0], dtype=np.float64),
        "amplitude_sd_display": np.asarray(disp_arrays[1], dtype=np.float64),
        "amplitude_sem_display": np.asarray(disp_arrays[2], dtype=np.float64),
        "relative_mean_display": np.asarray(disp_arrays[3], dtype=np.float64),
        "relative_sd_display": np.asarray(disp_arrays[4], dtype=np.float64),
        "relative_sem_display": np.asarray(disp_arrays[5], dtype=np.float64),
        "trial_names_used": list(used_names),
        "n_trials_used": int(len(used_names)),
        "baseline_start_s": safe_float(baseline_start_final),
        "baseline_end_s": safe_float(baseline_end_final),
        "baseline_stat": baseline_stat,
        "display_stride": int(stride),
        "f_center_hz_mean": safe_float(np.nanmean(f_vals)) if f_vals else np.nan,
        "band_hz_mean": np.asarray(band_mean, dtype=np.float64),
    }


def compute_spectrogram_data(t: np.ndarray, x: np.ndarray) -> dict:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0 or t.size < 32 or x.shape != t.shape:
        return {
            "time_s": np.array([], dtype=float),
            "freq_hz": np.array([], dtype=float),
            "power_db": np.array([], dtype=float),
        }

    nperseg = min(NFFT_SPEC, len(x))
    noverlap = min(nperseg - 1, int(float(SPECTROGRAM_OVERLAP_FRAC) * nperseg))
    fsg, tsg, psg = spectrogram(
        x,
        fs=fs,
        window=SPECTROGRAM_WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    t0 = float(t[0]) if t.size else 0.0
    return {
        "time_s": np.asarray(tsg + t0, dtype=np.float64),
        "freq_hz": np.asarray(fsg, dtype=np.float64),
        "power_linear": np.asarray(psg, dtype=np.float64),
        "power_db": np.asarray(10.0 * np.log10(np.maximum(psg, 1e-30)), dtype=np.float64),
    }


def decimate_curve_bundle(
    x: np.ndarray,
    arrays: list[np.ndarray],
    max_points: int = LFP_DISPLAY_MAX_POINTS,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    x = np.asarray(x, dtype=float)
    clean_arrays = [np.asarray(arr, dtype=float) for arr in arrays]
    if len(x) == 0:
        return np.array([], dtype=float), [np.array([], dtype=float) for _ in clean_arrays], 1

    max_points = max(2, int(max_points))
    stride = max(1, int(np.ceil(len(x) / max_points)))
    if stride == 1:
        return x.copy(), [arr.copy() if arr.shape == x.shape else np.array([], dtype=float) for arr in clean_arrays], 1

    idx = np.arange(0, len(x), stride, dtype=int)
    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)

    out_arrays = []
    for arr in clean_arrays:
        out_arrays.append(arr[idx] if arr.shape == x.shape else np.array([], dtype=float))
    return x[idx], out_arrays, int(stride)


def build_common_axis_1d(arrays: list[np.ndarray]) -> np.ndarray | None:
    valid = [np.asarray(a, dtype=float) for a in arrays if a is not None and len(a) >= 2]
    if not valid:
        return None
    lo = max(float(np.nanmin(a)) for a in valid)
    hi = min(float(np.nanmax(a)) for a in valid)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    steps = []
    for a in valid:
        dt = np.diff(a)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            steps.append(float(np.median(dt)))
    if not steps:
        return None
    step = float(np.median(steps))
    if not np.isfinite(step) or step <= 0:
        return None
    axis = np.arange(lo, hi + 0.5 * step, step, dtype=float)
    return axis if len(axis) >= 2 else None


def interpolate_surface_to_ref_grid(
    time_ref: np.ndarray,
    freq_ref: np.ndarray,
    time_s: np.ndarray,
    freq_hz: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    time_ref = np.asarray(time_ref, dtype=float)
    freq_ref = np.asarray(freq_ref, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full((len(freq_ref), len(time_ref)), np.nan, dtype=float)
    if (
        len(time_ref) < 2
        or len(freq_ref) < 2
        or len(time_s) < 2
        or len(freq_hz) < 2
        or values.shape != (len(freq_hz), len(time_s))
    ):
        return out
    FF, TT = np.meshgrid(freq_ref, time_ref, indexing="ij")
    interp_points = np.column_stack([FF.ravel(), TT.ravel()])
    interp = RegularGridInterpolator(
        (freq_hz, time_s),
        values,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(interp_points).reshape(len(freq_ref), len(time_ref))


def build_common_psd_summary(trial_results: dict) -> dict:
    items = []
    for name in sorted(trial_results.keys()):
        spectral = trial_results[name].get("spectral", {})
        freq_hz = np.asarray(spectral.get("freq_hz", []), dtype=float)
        psd_db = np.asarray(spectral.get("psd_db", []), dtype=float)
        if len(freq_hz) == 0 or psd_db.shape != freq_hz.shape:
            continue
        items.append({
            "name": name,
            "freq_hz": freq_hz,
            "psd_db": psd_db,
            "psd_linear": linear_from_db(psd_db),
        })

    if not items:
        return {
            "freq_hz": np.array([], dtype=float),
            "psd_db_mean": np.array([], dtype=float),
            "psd_db_sd": np.array([], dtype=float),
            "trial_names_used": [],
            "n_trials_used": 0,
        }

    freq_ref = build_common_axis_1d([item["freq_hz"] for item in items])
    if freq_ref is None:
        return {
            "freq_hz": np.array([], dtype=float),
            "psd_db_mean": np.array([], dtype=float),
            "psd_db_sd": np.array([], dtype=float),
            "trial_names_used": [],
            "n_trials_used": 0,
        }

    rows_linear = []
    rows_db = []
    used_names = []
    for item in items:
        row_linear = interpolate_to_ref_grid(freq_ref, item["freq_hz"], item["psd_linear"])
        if not np.any(np.isfinite(row_linear)):
            continue
        rows_linear.append(row_linear)
        rows_db.append(interpolate_to_ref_grid(freq_ref, item["freq_hz"], item["psd_db"]))
        used_names.append(item["name"])

    if not rows_linear:
        return {
            "freq_hz": np.array([], dtype=float),
            "psd_db_mean": np.array([], dtype=float),
            "psd_db_sd": np.array([], dtype=float),
            "trial_names_used": [],
            "n_trials_used": 0,
        }

    stack_linear = np.vstack(rows_linear)
    stack_db = np.vstack(rows_db)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_linear = np.nanmean(stack_linear, axis=0)
        sd_db = (
            np.nanstd(stack_db, axis=0, ddof=1)
            if stack_db.shape[0] >= 2
            else np.full(len(freq_ref), np.nan, dtype=float)
        )

    return {
        "freq_hz": np.asarray(freq_ref, dtype=np.float64),
        "psd_db_mean": db_from_linear(mean_linear),
        "psd_db_sd": np.asarray(sd_db, dtype=np.float64),
        "trial_names_used": used_names,
        "n_trials_used": int(len(used_names)),
    }


def empty_pta_display_summary() -> dict:
    return {
        "t_rel_s": np.array([], dtype=float),
        "mean": np.array([], dtype=float),
        "sd": np.array([], dtype=float),
        "sem": np.array([], dtype=float),
        "second_pulse_rel_s_mean": np.nan,
        "post_periods": float(DEFAULT_DISPLAY_POST_PERIODS),
    }


def build_first_pta_display_summary(first_pta: dict | None, post_periods: float = DEFAULT_DISPLAY_POST_PERIODS) -> dict:
    if first_pta is None:
        return empty_pta_display_summary()

    saved = dict(first_pta.get("display", {}))
    t_saved = np.asarray(saved.get("t_rel_s", []), dtype=float)
    y_saved = np.asarray(saved.get("mean", []), dtype=float)
    sd_saved = np.asarray(saved.get("sd", []), dtype=float)
    sem_saved = np.asarray(saved.get("sem", []), dtype=float)
    second_saved = safe_float(saved.get("second_pulse_rel_s_mean"))
    if len(t_saved) >= 2 and y_saved.shape == t_saved.shape:
        return {
            "t_rel_s": np.asarray(t_saved, dtype=np.float64),
            "mean": np.asarray(y_saved, dtype=np.float64),
            "sd": np.asarray(sd_saved, dtype=np.float64) if sd_saved.shape == t_saved.shape else nansd_stack([y_saved]),
            "sem": np.asarray(sem_saved, dtype=np.float64) if sem_saved.shape == t_saved.shape else nansem_stack([y_saved]),
            "second_pulse_rel_s_mean": float(second_saved) if np.isfinite(second_saved) else np.nan,
            "post_periods": safe_float(saved.get("post_periods")) if np.isfinite(safe_float(saved.get("post_periods"))) else float(post_periods),
        }

    segments = list(first_pta.get("segments", []))
    second_times = [
        float(seg.get("second_pulse_rel_s"))
        for seg in segments
        if np.isfinite(safe_float(seg.get("second_pulse_rel_s"))) and float(seg.get("second_pulse_rel_s")) > 0
    ]
    if not second_times:
        return empty_pta_display_summary()

    pre_s = float(np.nanmedian(second_times))
    if not np.isfinite(pre_s) or pre_s <= 0:
        return empty_pta_display_summary()

    rows = []
    dts = []
    for seg in segments:
        tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
        yx = np.asarray(seg.get("signal", []), dtype=float)
        if len(tx) < 2 or yx.shape != tx.shape:
            continue
        dt = np.diff(tx)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            dts.append(float(np.median(dt)))
        rows.append((tx, yx))
    if not rows or not dts:
        return empty_pta_display_summary()

    dt = float(np.median(dts))
    if not np.isfinite(dt) or dt <= 0:
        return empty_pta_display_summary()

    t_grid = np.arange(-pre_s, float(post_periods) * pre_s + 0.5 * dt, dt, dtype=float)
    if len(t_grid) < 2:
        return empty_pta_display_summary()

    stack = [interpolate_to_ref_grid(t_grid, tx, yx) for tx, yx in rows]
    return {
        "t_rel_s": np.asarray(t_grid, dtype=np.float64),
        "mean": nanmean_stack(stack),
        "sd": nansd_stack(stack),
        "sem": nansem_stack(stack),
        "second_pulse_rel_s_mean": float(np.nanmedian(second_times)),
        "post_periods": float(post_periods),
    }


def extract_train_pta_display_segments(
    trial_result: dict,
    post_periods: float = DEFAULT_DISPLAY_POST_PERIODS,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, float]:
    t = np.asarray(trial_result.get("t_full_s", []), dtype=float)
    x = np.asarray(trial_result.get("signal_full", []), dtype=float)
    pulse_times = np.asarray(trial_result.get("pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(t) < 2 or x.shape != t.shape or len(pulse_times) < 2:
        return None, None, np.nan

    diffs = np.diff(pulse_times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if not len(diffs):
        return None, None, np.nan
    ipi_s = float(np.median(diffs))
    if not np.isfinite(ipi_s) or ipi_s <= 0:
        return None, None, np.nan

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        return None, None, np.nan
    dt = float(np.median(dt))
    if not np.isfinite(dt) or dt <= 0:
        return None, None, np.nan

    t_grid = np.arange(-ipi_s, float(post_periods) * ipi_s + 0.5 * dt, dt, dtype=float)
    if len(t_grid) < 2:
        return None, None, np.nan

    rows = []
    for tp in pulse_times:
        t_rel = t - float(tp)
        keep = (t_rel >= -ipi_s) & (t_rel <= float(post_periods) * ipi_s)
        if int(np.sum(keep)) < 4:
            continue
        t_seg = np.asarray(t_rel[keep], dtype=float)
        x_seg = np.asarray(x[keep], dtype=float)
        pre = x_seg[(t_seg < 0) & (t_seg >= -ipi_s)]
        baseline = float(np.median(pre)) if len(pre) else 0.0
        rows.append(np.interp(t_grid, t_seg, x_seg - baseline, left=np.nan, right=np.nan))
    if not rows:
        return None, None, np.nan

    return np.asarray(t_grid, dtype=np.float64), np.vstack(rows).astype(np.float64), float(ipi_s)


def build_train_pta_display_summary(
    trial_results: dict,
    post_periods: float = DEFAULT_DISPLAY_POST_PERIODS,
) -> dict:
    if not isinstance(trial_results, dict) or not trial_results:
        return empty_pta_display_summary()

    t_ref = None
    trial_means = []
    second_times = []
    for name in sorted(trial_results.keys()):
        t_trial, segs, second_t = extract_train_pta_display_segments(trial_results[name], post_periods=post_periods)
        if t_trial is None or segs is None or segs.ndim != 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y_trial = np.nanmean(segs, axis=0)
        if t_ref is None:
            t_ref = t_trial.copy()
        elif t_trial.shape != t_ref.shape or not same_grid(t_trial, t_ref):
            y_trial = interpolate_to_ref_grid(t_ref, t_trial, y_trial)
        trial_means.append(y_trial)
        if np.isfinite(second_t) and second_t > 0:
            second_times.append(float(second_t))

    if t_ref is None or not trial_means:
        return empty_pta_display_summary()

    return {
        "t_rel_s": np.asarray(t_ref, dtype=np.float64),
        "mean": nanmean_stack(trial_means),
        "sd": nansd_stack(trial_means),
        "sem": nansem_stack(trial_means),
        "second_pulse_rel_s_mean": safe_float(np.nanmedian(second_times)) if second_times else np.nan,
        "post_periods": float(post_periods),
    }


def empty_pulsogram_heatmap_summary() -> dict:
    return {
        "pulse_numbers": np.array([], dtype=int),
        "t_rel_s": np.array([], dtype=float),
        "mean": np.array([], dtype=float),
    }


def build_pulsogram_heatmap_summary(pooled: list[dict]) -> dict:
    if not pooled:
        return empty_pulsogram_heatmap_summary()

    rows = []
    pulse_numbers = []
    time_axes = []
    for item in pooled:
        pulse_num = int(item.get("pulse_number", 0))
        t = np.asarray(item.get("full_rel_s", []), dtype=float)
        y = np.asarray(item.get("full_values_mean", []), dtype=float)
        if len(t) < 2 or y.shape != t.shape:
            continue
        pulse_numbers.append(pulse_num)
        time_axes.append(t)
        rows.append((pulse_num, t, y))
    if not rows:
        return empty_pulsogram_heatmap_summary()

    t_ref = build_common_axis_1d(time_axes)
    if t_ref is None:
        return empty_pulsogram_heatmap_summary()

    pulse_arr = np.asarray(sorted(pulse_numbers), dtype=int)
    matrix = np.full((len(pulse_arr), len(t_ref)), np.nan, dtype=float)
    pulse_to_row = {int(p): i for i, p in enumerate(pulse_arr)}
    for pulse_num, t, y in rows:
        matrix[pulse_to_row[int(pulse_num)]] = interpolate_to_ref_grid(t_ref, t, y)

    return {
        "pulse_numbers": pulse_arr,
        "t_rel_s": np.asarray(t_ref, dtype=np.float64),
        "mean": np.asarray(matrix, dtype=np.float64),
    }


def collect_common_grid_trials(trials: dict, field_names: list[str], t_key: str = "t") -> dict:
    out = {
        "trial_names": [],
        "t_common": np.array([], dtype=float),
    }
    for field in field_names:
        out[f"{field}_mean"] = np.array([], dtype=float)
        out[f"{field}_sd"] = np.array([], dtype=float)
        out[f"{field}_sem"] = np.array([], dtype=float)

    times = []
    stacks = {field: [] for field in field_names}
    used_names = {field: [] for field in field_names}

    for name in sorted(trials.keys()):
        td = trials[name]
        t = np.asarray(td.get(t_key, []), dtype=float)
        if len(t) < 2:
            continue
        times.append(t)

    t_ref = build_common_axis_1d(times)
    if t_ref is None:
        return out

    for name in sorted(trials.keys()):
        td = trials[name]
        t = np.asarray(td.get(t_key, []), dtype=float)
        if len(t) < 2:
            continue
        for field in field_names:
            arr = array_or_empty(td.get(field))
            if arr.shape != t.shape:
                continue
            stacks[field].append(interpolate_to_ref_grid(t_ref, t, arr))
            used_names[field].append(name)

    out["trial_names"] = sorted({name for names in used_names.values() for name in names})
    out["t_common"] = t_ref
    for field in field_names:
        out[f"{field}_trial_names"] = used_names[field]
        out[f"{field}_mean"] = nanmean_stack(stacks[field])
        out[f"{field}_sd"] = nansd_stack(stacks[field])
        out[f"{field}_sem"] = nansem_stack(stacks[field])
    return out


def summarize_processed_notched(processed_notched: dict, ephys: dict) -> dict:
    img_trials = processed_notched.get("trials", {})
    eph_trials = ephys.get("trials", {})

    all_trials = {}
    baseline_trials = {}
    stim_trials = {}

    for name, td in img_trials.items():
        all_trials[name] = td
        td_e = eph_trials.get(name, {})
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        if len(pulse_times) == 0:
            baseline_trials[name] = td
        else:
            stim_trials[name] = td

    fields = ["F_raw", "F_bleach_corr", "F_notched", "dff"]
    all_summary = collect_common_grid_trials(all_trials, fields, t_key="t")
    baseline_summary = collect_common_grid_trials(baseline_trials, fields, t_key="t")
    stim_summary = collect_common_grid_trials(stim_trials, fields, t_key="t")

    baseline_spec = compute_spectrogram_data(
        np.asarray(baseline_summary.get("t_common", []), dtype=float),
        np.asarray(baseline_summary.get("F_notched_mean", []), dtype=float),
    )

    fps_vals = [safe_float(td.get("fps_hz")) for td in img_trials.values()]

    return {
        "available": True,
        "n_trials_total": int(len(all_trials)),
        "n_trials_baseline": int(len(baseline_trials)),
        "n_trials_stim": int(len(stim_trials)),
        "fps_hz_mean": safe_float(np.nanmean(fps_vals)) if fps_vals else np.nan,
        "all": all_summary,
        "baseline": baseline_summary,
        "stim": stim_summary,
        "baseline_spectrogram": baseline_spec,
        "spectral_post_notch": processed_notched.get("spectral_summary_post_notch", {}),
    }


def empty_lfp_summary() -> dict:
    return {
        "available": False,
        "n_trials": 0,
        "trial_names": [],
        "sample_rate_hz": np.nan,
        "t_stim_s_full": np.array([], dtype=float),
        "mean_full": np.array([], dtype=float),
        "sd_full": np.array([], dtype=float),
        "sem_full": np.array([], dtype=float),
        "t_stim_s_display": np.array([], dtype=float),
        "mean_display": np.array([], dtype=float),
        "sd_display": np.array([], dtype=float),
        "sem_display": np.array([], dtype=float),
        "display_stride": 1,
    }


def summarize_lfp(ephys: dict, stim_trial_names: list[str]) -> dict:
    trials = ephys.get("trials", {})
    if not trials or not stim_trial_names:
        return empty_lfp_summary()

    time_axes = []
    signals = []
    used_names = []
    for name in stim_trial_names:
        td = trials.get(name, {})
        t = np.asarray(td.get("t_stim_s", []), dtype=float)
        x = np.asarray(td.get("channels", {}).get("LFP", []), dtype=float)
        if len(t) < 2 or x.shape != t.shape:
            continue
        time_axes.append(t)
        signals.append(x)
        used_names.append(name)

    if not time_axes:
        return empty_lfp_summary()

    t_ref = build_common_axis_1d(time_axes)
    if t_ref is None:
        return empty_lfp_summary()

    stack = np.vstack([interpolate_to_ref_grid(t_ref, t, x) for t, x in zip(time_axes, signals)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_full = nanmean_stack([row for row in stack])
        sd_full = nansd_stack([row for row in stack])
        sem_full = nansem_stack([row for row in stack])
    t_display, [mean_display, sd_display, sem_display], stride = decimate_curve_bundle(
        t_ref,
        [mean_full, sd_full, sem_full],
        max_points=LFP_DISPLAY_MAX_POINTS,
    )

    return {
        "available": True,
        "n_trials": int(len(used_names)),
        "trial_names": list(used_names),
        "sample_rate_hz": safe_float(ephys.get("sample_rate")),
        "t_stim_s_full": np.asarray(t_ref, dtype=np.float64),
        "mean_full": np.asarray(mean_full, dtype=np.float64),
        "sd_full": np.asarray(sd_full, dtype=np.float64),
        "sem_full": np.asarray(sem_full, dtype=np.float64),
        "t_stim_s_display": np.asarray(t_display, dtype=np.float64),
        "mean_display": np.asarray(mean_display, dtype=np.float64),
        "sd_display": np.asarray(sd_display, dtype=np.float64),
        "sem_display": np.asarray(sem_display, dtype=np.float64),
        "display_stride": int(stride),
    }


def summarize_first_pta(first_pta: dict | None) -> dict:
    if first_pta is None:
        return {
            "available": False,
            "n_trials": 0,
            "trial_names": [],
            "t_rel_s": np.array([], dtype=float),
            "mean": np.array([], dtype=float),
            "sem": np.array([], dtype=float),
            "spread": np.array([], dtype=float),
            "display": empty_pta_display_summary(),
            "latency": {},
            "latency_jitter": {},
        }

    return {
        "available": True,
        "n_trials": int(len(first_pta.get("trial_names_used", []))),
        "trial_names": list(first_pta.get("trial_names_used", [])),
        "t_rel_s": np.asarray(first_pta.get("t_rel_s", []), dtype=float),
        "mean": np.asarray(first_pta.get("pta_mean", []), dtype=float),
        "sem": np.asarray(first_pta.get("pta_sem", []), dtype=float),
        "spread": np.asarray(first_pta.get("pta_spread", []), dtype=float),
        "display": build_first_pta_display_summary(first_pta),
        "latency": first_pta.get("latency", {}),
        "latency_jitter": first_pta.get("latency_jitter", {}),
    }


def summarize_train_pta(train_pta: dict | None) -> dict:
    if train_pta is None:
        return {
            "available": False,
            "n_trials": 0,
            "trial_names": [],
            "t_rel_s": np.array([], dtype=float),
            "mean_across_trials": np.array([], dtype=float),
            "sd_across_trials": np.array([], dtype=float),
            "sem_across_trials": np.array([], dtype=float),
            "display": empty_pta_display_summary(),
            "spectrogram": empty_train_spectrogram_summary(),
            "lfp_spectrogram": empty_train_spectrogram_summary(),
            "signal_hilbert": empty_hilbert_summary(),
            "lfp_hilbert": empty_hilbert_summary(),
            "power_spectrum": {
                "freq_hz": np.array([], dtype=float),
                "psd_db_mean": np.array([], dtype=float),
                "psd_db_sd": np.array([], dtype=float),
                "trial_names_used": [],
                "n_trials_used": 0,
            },
            "plv": {
                "trial_names": [],
                "plv_by_trial": np.array([], dtype=float),
                "phase_pulses_rad": np.array([], dtype=float),
            },
            "hilbert_entrainment": {},
            "latency": {},
            "metrics": {},
        }

    trial_results = train_pta.get("trial_results", {})
    settings = train_pta.get("settings", {})
    names = sorted(trial_results.keys())
    spec_baseline_end_s = safe_float(settings.get("spectrogram_relative_baseline_end_s"))
    if not np.isfinite(spec_baseline_end_s):
        spec_baseline_end_s = float(DEFAULT_SPEC_REL_BASELINE_END_S)
    spec_baseline_stat = str(settings.get("spectrogram_relative_baseline_stat", DEFAULT_SPEC_REL_BASELINE_STAT)).lower()
    if spec_baseline_stat not in {"mean", "median"}:
        spec_baseline_stat = DEFAULT_SPEC_REL_BASELINE_STAT
    hilbert_baseline_end_s = safe_float(settings.get("hilbert_relative_baseline_end_s"))
    if not np.isfinite(hilbert_baseline_end_s):
        hilbert_baseline_end_s = float(DEFAULT_HILBERT_REL_BASELINE_END_S)
    hilbert_baseline_stat = str(settings.get("hilbert_relative_baseline_stat", DEFAULT_HILBERT_REL_BASELINE_STAT)).lower()
    if hilbert_baseline_stat not in {"mean", "median"}:
        hilbert_baseline_stat = DEFAULT_HILBERT_REL_BASELINE_STAT

    f_stim_vals = []
    plv_sections: dict[str, dict[str, list]] = {}
    signal_hilbert_sections = set()
    pta_items = []
    for name in names:
        tr = trial_results[name]

        t_rel = np.asarray(tr.get("t_rel_s", []), dtype=float)
        pta_mean = np.asarray(tr.get("pta_mean", []), dtype=float)
        if len(t_rel) and pta_mean.shape == t_rel.shape:
            pta_items.append((t_rel, pta_mean))

        f_stim_vals.append(safe_float(tr.get("f_stim_hz")))
        for key, sec in tr.items():
            if not is_plv_section_key(key) or not isinstance(sec, dict):
                continue
            bucket = plv_sections.setdefault(key, {"vals": [], "centers": [], "phases": []})
            bucket["vals"].append(safe_float(sec.get("plv")))
            bucket["centers"].append(safe_float(sec.get("f_center_hz")))
            ph = np.asarray(sec.get("phase_pulses_rad", []), dtype=float)
            ph = ph[np.isfinite(ph)]
            if len(ph):
                bucket["phases"].append(ph)
        for key, sec in tr.items():
            if is_signal_hilbert_section_key(key) and isinstance(sec, dict):
                signal_hilbert_sections.add(key)

    spectrogram_summary = build_common_spectrogram_summary(
        trial_results=trial_results,
        section_key="spectrogram",
        baseline_end_s=float(spec_baseline_end_s),
        baseline_stat=spec_baseline_stat,
    )
    lfp_spectrogram_summary = build_common_spectrogram_summary(
        trial_results=trial_results,
        section_key="lfp_spectrogram",
        baseline_end_s=float(spec_baseline_end_s),
        baseline_stat=spec_baseline_stat,
    )
    if "signal_hilbert" not in signal_hilbert_sections:
        signal_hilbert_sections.add("signal_hilbert")
    signal_hilbert_summaries = {
        key: build_common_hilbert_summary(
            trial_results=trial_results,
            section_key=key,
            baseline_end_s=float(hilbert_baseline_end_s),
            baseline_stat=hilbert_baseline_stat,
        )
        for key in sorted(signal_hilbert_sections, key=hilbert_section_sort_key)
    }
    lfp_hilbert_summary = build_common_hilbert_summary(
        trial_results=trial_results,
        section_key="lfp_hilbert",
        baseline_end_s=float(hilbert_baseline_end_s),
        baseline_stat=hilbert_baseline_stat,
    )
    display_summary = build_train_pta_display_summary(trial_results)
    t_ref = build_common_axis_1d([item[0] for item in pta_items]) if pta_items else None
    pta_stack = [interpolate_to_ref_grid(t_ref, tx, yx) for tx, yx in pta_items] if t_ref is not None else []
    power_spectrum_summary = build_common_psd_summary(trial_results)

    plv_summary = {}
    metrics = {
        "f_stim_hz_mean": safe_float(np.nanmean(f_stim_vals)) if f_stim_vals else np.nan,
    }
    for key in sorted(plv_sections.keys(), key=plv_section_sort_key):
        bucket = plv_sections[key]
        vals = np.asarray(bucket["vals"], dtype=np.float64)
        centers = np.asarray(bucket["centers"], dtype=np.float64)
        plv_summary[key] = {
            "trial_names": names,
            "plv_by_trial": vals,
            "f_center_hz_by_trial": centers,
            "f_center_hz_mean": safe_float(np.nanmean(centers)) if len(centers) else np.nan,
            "phase_pulses_rad": np.concatenate(bucket["phases"]).astype(np.float64) if bucket["phases"] else np.array([], dtype=float),
        }
        metric_name = "plv_mean" if key == "plv" else f"{key}_mean"
        metrics[metric_name] = safe_float(np.nanmean(vals)) if len(vals) else np.nan

    trial_metric_dicts = []
    for name in names:
        sec = trial_results.get(name, {}).get("metrics", {})
        if isinstance(sec, dict):
            trial_metric_dicts.append(sec)

    def metric_values(metric_name: str) -> np.ndarray:
        vals = np.asarray([safe_float(m.get(metric_name)) for m in trial_metric_dicts], dtype=float)
        return vals[np.isfinite(vals)]

    hilbert_entrainment = {}
    for metric_name in [
        "hilbert_amp_baseline_median",
        "hilbert_amp_stim_median",
        "hilbert_amp_ratio",
        "hilbert_amp_percent_change",
        "hilbert_amp_baseline_vs_stim_u",
        "hilbert_amp_baseline_vs_stim_p",
        "hilbert_amp_baseline_n_bins",
        "hilbert_amp_stim_n_bins",
        "hilbert_amp_baseline_bin_median",
        "hilbert_amp_stim_bin_median",
    ]:
        vals = metric_values(metric_name)
        hilbert_entrainment[f"{metric_name}_by_trial"] = vals
        hilbert_entrainment[f"{metric_name}_median"] = safe_float(np.nanmedian(vals)) if len(vals) else np.nan
        hilbert_entrainment[f"{metric_name}_mean"] = safe_float(np.nanmean(vals)) if len(vals) else np.nan
        hilbert_entrainment[f"{metric_name}_sd"] = safe_float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else np.nan
        metrics[metric_name] = hilbert_entrainment[f"{metric_name}_median"]
    p_vals = metric_values("hilbert_amp_baseline_vs_stim_p")
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_trial_mannwhitney_p_by_trial"] = p_vals
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_trial_mannwhitney_n_significant_0p05"] = int(np.sum(p_vals < 0.05))
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_trial_mannwhitney_frac_significant_0p05"] = (
        safe_float(np.mean(p_vals < 0.05)) if len(p_vals) else np.nan
    )
    base_pair = np.asarray([safe_float(m.get("hilbert_amp_baseline_median")) for m in trial_metric_dicts], dtype=float)
    stim_pair = np.asarray([safe_float(m.get("hilbert_amp_stim_median")) for m in trial_metric_dicts], dtype=float)
    keep_pair = np.isfinite(base_pair) & np.isfinite(stim_pair)
    base_pair = base_pair[keep_pair]
    stim_pair = stim_pair[keep_pair]
    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    if len(base_pair) >= 2 and np.any(np.abs(stim_pair - base_pair) > 1e-12):
        try:
            stat = wilcoxon(stim_pair, base_pair, alternative="two-sided", zero_method="wilcox")
            wilcoxon_stat = safe_float(stat.statistic)
            wilcoxon_p = safe_float(stat.pvalue)
        except ValueError:
            pass
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_test"] = "Wilcoxon signed-rank"
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_wilcoxon_stat"] = wilcoxon_stat
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_wilcoxon_p"] = wilcoxon_p
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_wilcoxon_n_pairs"] = int(len(base_pair))
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_wilcoxon_baseline_by_trial"] = base_pair.astype(np.float64)
    hilbert_entrainment["hilbert_amp_baseline_vs_stim_wilcoxon_stim_by_trial"] = stim_pair.astype(np.float64)
    metrics["hilbert_amp_baseline_vs_stim_test"] = "Wilcoxon signed-rank"
    metrics["hilbert_amp_baseline_vs_stim_stat"] = wilcoxon_stat
    metrics["hilbert_amp_baseline_vs_stim_p"] = wilcoxon_p
    metrics["hilbert_amp_baseline_vs_stim_n_pairs"] = int(len(base_pair))
    metrics["hilbert_amp_baseline_vs_stim_trial_mannwhitney_n_significant_0p05"] = hilbert_entrainment[
        "hilbert_amp_baseline_vs_stim_trial_mannwhitney_n_significant_0p05"
    ]
    metrics["hilbert_amp_baseline_vs_stim_trial_mannwhitney_frac_significant_0p05"] = hilbert_entrainment[
        "hilbert_amp_baseline_vs_stim_trial_mannwhitney_frac_significant_0p05"
    ]

    latency = {}
    for i in range(1, 4):
        lat_name = f"peak_{i}_latency_ms"
        amp_name = f"peak_{i}_amplitude"
        lat_vals = metric_values(lat_name)
        amp_vals = metric_values(amp_name)
        latency[f"{lat_name}_by_trial"] = lat_vals
        latency[f"{lat_name}_median"] = safe_float(np.nanmedian(lat_vals)) if len(lat_vals) else np.nan
        latency[f"{lat_name}_mean"] = safe_float(np.nanmean(lat_vals)) if len(lat_vals) else np.nan
        latency[f"peak_{i}_jitter_ms"] = safe_float(np.nanstd(lat_vals, ddof=1)) if len(lat_vals) >= 2 else np.nan
        latency[f"{amp_name}_median"] = safe_float(np.nanmedian(amp_vals)) if len(amp_vals) else np.nan
        latency[f"{amp_name}_mean"] = safe_float(np.nanmean(amp_vals)) if len(amp_vals) else np.nan
        latency[f"peak_{i}_n_latency_trials"] = int(len(lat_vals))
        metrics[f"{lat_name}_median"] = latency[f"{lat_name}_median"]
        metrics[f"{lat_name}_mean"] = latency[f"{lat_name}_mean"]
        metrics[f"peak_{i}_jitter_ms"] = latency[f"peak_{i}_jitter_ms"]
        metrics[f"{amp_name}_median"] = latency[f"{amp_name}_median"]

    out = {
        "available": True,
        "n_trials": int(len(names)),
        "trial_names": names,
        "t_rel_s": t_ref if t_ref is not None else np.array([], dtype=float),
        "mean_across_trials": nanmean_stack(pta_stack),
        "sd_across_trials": nansd_stack(pta_stack),
        "sem_across_trials": nansem_stack(pta_stack),
        "display": display_summary,
        "spectrogram": spectrogram_summary,
        "lfp_spectrogram": lfp_spectrogram_summary,
        "signal_hilbert": signal_hilbert_summaries.get("signal_hilbert", empty_hilbert_summary()),
        "lfp_hilbert": lfp_hilbert_summary,
        "power_spectrum": power_spectrum_summary,
        **plv_summary,
        "hilbert_entrainment": hilbert_entrainment,
        "latency": latency,
        "metrics": metrics,
    }
    for key, value in signal_hilbert_summaries.items():
        if key != "signal_hilbert":
            out[key] = value
    return out


def summarize_pulsogram(pulsogram: dict | None) -> dict:
    if pulsogram is None:
        return {
            "available": False,
            "n_trials": 0,
            "n_pulses": 0,
            "pulse_numbers": np.array([], dtype=int),
            "n_trials_by_pulse": np.array([], dtype=int),
            "first_peak_amp": np.array([], dtype=float),
            "first_peak_lat_s": np.array([], dtype=float),
            "second_peak_amp": np.array([], dtype=float),
            "second_peak_lat_s": np.array([], dtype=float),
            "waveform_t_rel_s": np.array([], dtype=float),
            "waveform_mean": np.array([], dtype=float),
            "waveform_sd": np.array([], dtype=float),
            "heatmap": empty_pulsogram_heatmap_summary(),
        }

    pooled = pulsogram.get("pooled_metrics", [])
    pulse_numbers = np.array([int(p.get("pulse_number", 0)) for p in pooled], dtype=int) if pooled else np.array([], dtype=int)
    n_trials_by_pulse = np.array([int(p.get("n_trials", 0)) for p in pooled], dtype=int) if pooled else np.array([], dtype=int)
    first_peak_amp = np.array([safe_float(p.get("first_peak_amp")) for p in pooled], dtype=float)
    first_peak_lat_s = np.array([safe_float(p.get("first_peak_lat_s")) for p in pooled], dtype=float)
    second_peak_amp = np.array([safe_float(p.get("second_peak_amp")) for p in pooled], dtype=float)
    second_peak_lat_s = np.array([safe_float(p.get("second_peak_lat_s")) for p in pooled], dtype=float)

    t_ref = None
    waves = []
    for p in pooled:
        t = np.asarray(p.get("full_rel_s", []), dtype=float)
        y = np.asarray(p.get("full_values_mean", []), dtype=float)
        if t_ref is None and len(t) and len(y):
            t_ref = t.copy()
        if t_ref is not None and len(t) and same_grid(t, t_ref) and y.shape == t_ref.shape:
            waves.append(y)

    return {
        "available": True,
        "n_trials": int(len(pulsogram.get("trial_results", {}))),
        "n_pulses": int(len(pooled)),
        "pulse_numbers": pulse_numbers,
        "n_trials_by_pulse": n_trials_by_pulse,
        "first_peak_amp": first_peak_amp,
        "first_peak_lat_s": first_peak_lat_s,
        "second_peak_amp": second_peak_amp,
        "second_peak_lat_s": second_peak_lat_s,
        "waveform_t_rel_s": t_ref if t_ref is not None else np.array([], dtype=float),
        "waveform_mean": nanmean_stack(waves),
        "waveform_sd": nansd_stack(waves),
        "heatmap": build_pulsogram_heatmap_summary(pooled),
    }


def summarize_ephys(ephys: dict) -> dict:
    trials = ephys.get("trials", {})
    names = sorted(trials.keys())
    baseline_names = []
    stim_names = []
    pulse_counts = []
    cam_fps = []
    stim_minus_cam = []
    median_ipi = []

    for name in names:
        td = trials[name]
        pulse_times = np.asarray(td.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        if len(pulse_times) == 0:
            baseline_names.append(name)
        else:
            stim_names.append(name)
        pulse_counts.append(len(pulse_times))
        cam_fps.append(safe_float(td.get("cam_fps_hz")))
        stim_minus_cam.append(safe_float(td.get("stim_minus_cam_s")))
        median_ipi.append(safe_float(td.get("median_ipi_ms")))

    return {
        "n_trials_total": int(len(names)),
        "n_trials_baseline": int(len(baseline_names)),
        "n_trials_stim": int(len(stim_names)),
        "baseline_trial_names": baseline_names,
        "stim_trial_names": stim_names,
        "sample_rate_hz": safe_float(ephys.get("sample_rate")),
        "cam_fps_hz_mean": safe_float(np.nanmean(cam_fps)) if cam_fps else np.nan,
        "stim_minus_cam_s_mean": safe_float(np.nanmean(stim_minus_cam)) if stim_minus_cam else np.nan,
        "median_ipi_ms_mean": safe_float(np.nanmean(median_ipi)) if median_ipi else np.nan,
        "n_pulses_mean": safe_float(np.nanmean(pulse_counts)) if pulse_counts else np.nan,
    }


def build_final_block_dict(
    mouse: str,
    date: str,
    block: str,
    paths: dict[str, Path],
) -> dict:
    ephys = load_pickle(paths["epoched_ephys"])
    processed_notched = load_pickle(paths["processed_notched"])
    first_pta = load_pickle(paths["first_pta"]) if paths["first_pta"].exists() else None
    train_pta = load_pickle(paths["train_pta"]) if paths["train_pta"].exists() else None
    pulsogram = load_pickle(paths["pulsogram"]) if paths["pulsogram"].exists() else None

    processed_trials = processed_notched.get("trials", {})
    ephys_trials = ephys.get("trials", {})
    baseline_trial_names = []
    stim_trial_names = []
    for name in sorted(processed_trials.keys()):
        td_e = ephys_trials.get(name, {})
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        if len(pulse_times) == 0:
            baseline_trial_names.append(name)
        else:
            stim_trial_names.append(name)

    return {
        "mouse": mouse,
        "date": date,
        "block": block,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_paths": {k: str(v) for k, v in paths.items() if k != "output"},
        "source_settings": {
            "processed_notched": processed_notched.get("notch_processing", {}),
            "first_pta": {} if first_pta is None else first_pta.get("settings", {}),
            "train_pta": {} if train_pta is None else train_pta.get("settings", {}),
            "pulsogram": {} if pulsogram is None else pulsogram.get("settings", {}),
        },
        "summary": {
            "trial_counts": {
                "total": int(len(processed_trials)),
                "baseline": int(len(baseline_trial_names)),
                "stim": int(len(stim_trial_names)),
                "single_pta_valid": 0 if first_pta is None else int(len(first_pta.get("trial_names_used", []))),
                "train_pta_valid": 0 if train_pta is None else int(len(train_pta.get("trial_results", {}))),
                "pulsogram_valid": 0 if pulsogram is None else int(len(pulsogram.get("trial_results", {}))),
            },
            "ephys": summarize_ephys(ephys),
            "lfp": summarize_lfp(ephys, stim_trial_names),
            "processed_notched": summarize_processed_notched(processed_notched, ephys),
            "single_pta": summarize_first_pta(first_pta),
            "train_pta": summarize_train_pta(train_pta),
            "pulsogram": summarize_pulsogram(pulsogram),
        },
        "trials": {
            "baseline_trial_names": baseline_trial_names,
            "stim_trial_names": stim_trial_names,
            "processed_notched": processed_trials,
            "ephys": ephys_trials,
            "first_pta_segments": [] if first_pta is None else first_pta.get("segments", []),
            "train_pta": {} if train_pta is None else train_pta.get("trial_results", {}),
            "pulsogram": {} if pulsogram is None else pulsogram.get("trial_results", {}),
        },
    }


def get_block_paths(mouse: str, date: str, block: str) -> dict[str, Path]:
    mouse_root = DATA_ANALYSIS_ROOT / mouse
    img_block = mouse_root / "Imaging_Data" / date / block
    eph_block = mouse_root / "Open_Ephys" / date / block
    return {
        "epoched_ephys": eph_block / f"{block}_epoched_ephys.pkl",
        "processed_notched": img_block / f"{block}_traces_processed_notched.pkl",
        "first_pta": img_block / f"{block}_traces_processed_notched_pta_first_pulse.pkl",
        "train_pta": img_block / f"{block}_traces_processed_notched_pta_train.pkl",
        "pulsogram": img_block / f"{block}_traces_processed_notched_pulsogram.pkl",
        "output": img_block / f"{block}{OUTPUT_SUFFIX}",
    }


def process_block(mouse: str, date: str, block: str) -> dict:
    paths = get_block_paths(mouse, date, block)
    label = f"{date} | {block}"

    required = ["epoched_ephys", "processed_notched"]
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        print(f"[SKIP] {label} | missing: {', '.join(missing)}")
        return {"status": "missing_inputs", "label": label, "missing": missing}

    if SAVE_OUTPUT and paths["output"].exists() and not OVERWRITE:
        print(f"[SKIP] {label} | output exists")
        return {"status": "skipped_existing", "label": label, "path": str(paths['output'])}

    try:
        final_dict = build_final_block_dict(mouse, date, block, paths)
    except Exception as e:
        print(f"[SKIP] {label} | failed to build summary")
        print(f"  {type(e).__name__}: {e}")
        return {"status": "failed", "label": label, "reason": f"{type(e).__name__}: {e}"}

    if SAVE_OUTPUT:
        with open(paths["output"], "wb") as f:
            pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] {label} -> {paths['output'].name}")
        return {"status": "saved", "label": label, "path": str(paths["output"])}

    print(f"[DONE] {label} | SAVE_OUTPUT=False")
    return {"status": "done", "label": label}


def run_batch(mouse: str) -> list[dict]:
    imaging_root = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data"
    results = []
    if not imaging_root.exists():
        raise FileNotFoundError(f"Imaging root not found: {imaging_root}")

    for date_dir in sorted(imaging_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
                continue
            results.append(process_block(mouse, date_dir.name, block_dir.name))
    return results


def run_single_date(mouse: str, date: str) -> list[dict]:
    results = []
    imaging_root = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date
    if not imaging_root.exists():
        raise FileNotFoundError(f"Imaging date root not found: {imaging_root}")

    for block_dir in sorted(imaging_root.iterdir()):
        if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
            continue
        results.append(process_block(mouse, date, block_dir.name))
    return results


def print_batch_summary(results: list[dict]) -> None:
    if not results:
        print("No blocks found.")
        return

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("\nBatch summary")
    for key in sorted(counts.keys()):
        print(f"  {key}: {counts[key]}")

    failed = [r for r in results if r["status"] in {"failed", "missing_inputs"}]
    if failed:
        print("\nBlocks needing attention:")
        for r in failed:
            if r["status"] == "missing_inputs":
                print(f"  {r['label']} | missing: {', '.join(r['missing'])}")
            else:
                print(f"  {r['label']} | {r.get('reason', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one final block summary pickle from filter/PTA/pulsogram outputs.")
    parser.add_argument("--mouse", default=MOUSE_NAME, help="Mouse name(s), e.g. Jamie10 or Jamie10,Jamie11; use None for all mice")
    parser.add_argument("--date", default=SINGLE_DATE, help="Single date for non-batch mode")
    parser.add_argument("--block", default=SINGLE_BLOCK, help="Single block for non-batch mode")
    parser.add_argument("--batch", action="store_true", help="Run over all date/block folders for the selected mouse(s)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing summary pickles")
    parser.add_argument("--no-save", action="store_true", help="Build summaries but do not write output files")
    args = parser.parse_args()

    global OVERWRITE, SAVE_OUTPUT
    OVERWRITE = bool(args.overwrite or OVERWRITE)
    SAVE_OUTPUT = False if args.no_save else SAVE_OUTPUT

    run_as_batch = bool(args.batch or RUN_BATCH)
    mouse_names = resolve_mouse_names(args.mouse)
    if not mouse_names:
        print("No mice found to process.")
        return

    if run_as_batch:
        results = []
        for mouse_name in mouse_names:
            results.extend(run_batch(mouse_name))
        print_batch_summary(results)
    elif args.date is not None and args.block is None:
        results = []
        for mouse_name in mouse_names:
            results.extend(run_single_date(mouse_name, args.date))
        print_batch_summary(results)
    else:
        if args.date is None or args.block is None:
            raise ValueError("Set --date to run one date, or set both --date and --block to run one block.")
        results = []
        for mouse_name in mouse_names:
            results.append(process_block(mouse_name, args.date, args.block))
        print_batch_summary(results)


if __name__ == "__main__":
    main()
