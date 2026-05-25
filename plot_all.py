from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from config import DATA_ANALYSIS_ROOT


FIGURES_DIR = DATA_ANALYSIS_ROOT / "figures"

MOUSE_NAME = "Jamie11"
SINGLE_DATE = "28-04-26"
SINGLE_BLOCK = "R2"



# -------------------------
# EXECUTION TOGGLES
# -------------------------
SHOW_FIGURE = True
SAVE_FIGURE = True
FIG_DPI = 300


# -------------------------
# PANEL TOGGLES
# -------------------------
PLOT_SINGLE_PTA = "derivative"  # False/off, True/"normal", "derivative", or "both"
PLOT_FULL_TRACE = True
PLOT_STIM_TRACE = True
PLOT_LFP = True
PLOT_VELOCITY = True
PLOT_PULSE_WINDOWS = False
PLOT_FFT = False
PLOT_SPECTROGRAM = True
PLOT_LFP_SPECTROGRAM = False
PLOT_SIGNAL_HILBERT = False
PLOT_SIGNAL_HILBERT_HARMONICS = "1"  # e.g. "1", "2", "1+2+3", or "5+6"
PLOT_LFP_HILBERT = False
PLOT_PLV_HISTOGRAM = False
PLOT_PLV_HISTOGRAMS = ""  # 1=stim frequency, 2=2x stim, 3=3x stim; e.g. "1+2+3"
PLOT_BAND_SPECTRUM = False
PLOT_PULSOGRAM = False


# -------------------------
# DISPLAY SETTINGS
# -------------------------
SINGLE_PTA_PRE_SEC = 0.010
PULSOGRAM_TIME_RANGE_MS = (-12.0, 12.0)
LFP_DISPLAY_MAX_POINTS = 12000
USE_DECIMATED_LFP = True  # False = plot full 30 kHz LFP traces
FULL_TRACE_LINEWIDTH = 0.8
SINGLE_PTA_LINEWIDTH = 1.4
SPEC_FMAX_HZ = 300.0
SPECTROGRAM_VIEW = "absolute"  # "absolute" or "relative"
SPECTROGRAM_ABS_CMAP = "magma"
SPECTROGRAM_REL_CMAP = "RdBu_r"
SPECTROGRAM_BASELINE_END_S = -0.5
SPECTROGRAM_BASELINE_STAT = "median"
SPECTROGRAM_ABS_PERCENTILES = (5.0, 99.5)
SPECTROGRAM_REL_PERCENTILES = (2.0, 98.0)
SPECTROGRAM_SCALE_MODE = "manual"  # "manual" or "percentile"
GEVI_SPECTROGRAM_REL_DB_RANGE = (-8.0, 8.0)
LFP_SPECTROGRAM_REL_DB_RANGE = (-10.0, 50.0)
SPECTROGRAM_INTERPOLATION = "bilinear"
HILBERT_VIEW = "relative"  # "absolute" or "relative"
PRINT_EPHYS_SUMMARY = True
GEVI_DISPLAY_SCALE = 100.0  # stored dF/F is fractional; plots show percent dF/F
GEVI_YLABEL = "dF/F"
GEVI_DERIV_YLABEL = "d(dF/F)/dt (%/s)"
TRIAL_OVERVIEW_TIME_WINDOW_S = (-5.0, 15.0)
ROW_LABEL_FONTSIZE = 13
ROW_LABEL_X = -0.44
ROW_LABEL_Y = 1.12
TRIAL_PANEL_XLIMS = {
    "R1_10": {
        "full_trace": (-0.1, 2.5),
        "lfp": (-0.1, 2.5),
        "stim_trace": (-0.05, 0.2),
        "single_pta": (-0.001, 0.009),
    }
}


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def slugify_output_name(*parts) -> str:
    text = "_".join(str(part) for part in parts if part is not None and str(part).strip())
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "figure"


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def gevi_display(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * float(GEVI_DISPLAY_SCALE)


def values_in_time_window(t: np.ndarray, y: np.ndarray, window: tuple[float, float] = TRIAL_OVERVIEW_TIME_WINDOW_S) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(t), len(y))
    if n == 0:
        return np.array([], dtype=float)
    t = t[:n]
    y = y[:n]
    keep = (t >= float(window[0])) & (t <= float(window[1])) & np.isfinite(y)
    return y[keep]


def trial_panel_xlim(trial_name: str, panel_name: str):
    return TRIAL_PANEL_XLIMS.get(str(trial_name), {}).get(str(panel_name))


def tighten_ylim_to_xlim(ax, t: np.ndarray, y: np.ndarray, xlim, frac: float = 0.08) -> None:
    if xlim is None:
        return
    vals = values_in_time_window(t, y, xlim)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    ax.set_ylim(*expand_limits(float(np.nanmin(vals)), float(np.nanmax(vals)), frac=frac))


def single_pta_mode() -> str:
    if isinstance(PLOT_SINGLE_PTA, bool):
        return "normal" if PLOT_SINGLE_PTA else "off"
    mode = re.sub(r"\s+", " ", str(PLOT_SINGLE_PTA).strip().lower())
    mode = mode.replace("_", " ").replace("-", " ")
    if mode in {"", "0", "false", "no", "none", "off"}:
        return "off"
    if mode in {"1", "true", "yes", "on", "normal", "norm"}:
        return "normal"
    if mode in {"derivative", "deriv", "diff", "dy/dt", "dydt"}:
        return "derivative"
    if mode in {"both", "normal derivative", "normal plus derivative", "norm derivative", "norm deriv"}:
        return "both"
    raise ValueError("PLOT_SINGLE_PTA must be False/off, True/'normal', 'derivative', or 'both'.")


def first_derivative_curve(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    dy = np.diff(y)
    good = np.isfinite(dx) & np.isfinite(dy) & (dx > 0)
    if int(np.sum(good)) < 2:
        return None
    return (x[:-1][good] + 0.5 * dx[good]), dy[good] / dx[good]


HARMONIC_WORDS = {
    "first": 1,
    "fundamental": 1,
    "one": 1,
    "second": 2,
    "two": 2,
    "third": 3,
    "three": 3,
    "fourth": 4,
    "four": 4,
    "fifth": 5,
    "five": 5,
    "sixth": 6,
    "six": 6,
}


def parse_harmonic_selection(value) -> list[int]:
    if value is None or value is False:
        return []
    if value is True:
        return [1]
    if isinstance(value, (int, np.integer)):
        return [int(value)] if int(value) > 0 else []
    text = str(value).strip().lower()
    if not text or text in {"false", "none", "off", "no"}:
        return []
    text = text.replace("harmonic", "").replace("h", "")
    out = []
    for part in re.split(r"[+,\s;/]+", text):
        part = part.strip()
        if not part:
            continue
        if part in HARMONIC_WORDS:
            out.append(HARMONIC_WORDS[part])
            continue
        m = re.match(r"^(\d+)", part)
        if m:
            out.append(int(m.group(1)))
    clean = []
    for h in out:
        if h > 0 and h not in clean:
            clean.append(h)
    return clean


def plv_section_key(harmonic: int) -> str:
    return "plv" if int(harmonic) == 1 else f"plv_h{int(harmonic)}"


def plv_label(harmonic: int) -> str:
    return "PLV" if int(harmonic) == 1 else f"PLV H{int(harmonic)}"


def signal_hilbert_section_key(harmonic: int) -> str:
    return "signal_hilbert" if int(harmonic) == 1 else f"signal_hilbert_h{int(harmonic)}"


def signal_hilbert_label(harmonic: int) -> str:
    return "GEVI Hilbert" if int(harmonic) == 1 else f"GEVI Hilbert H{int(harmonic)}"


def db_from_linear(power_linear: np.ndarray) -> np.ndarray:
    power_linear = np.asarray(power_linear, dtype=float)
    if power_linear.size == 0:
        return np.asarray(power_linear, dtype=float)
    return 10.0 * np.log10(np.maximum(power_linear, 1e-30))


def spectrogram_linear_from_section(sec: dict, linear_key: str = "power_linear", db_key: str = "power_db") -> np.ndarray:
    p_linear = np.asarray(sec.get(linear_key, []), dtype=float)
    if p_linear.ndim == 2:
        return p_linear

    p_db = np.asarray(sec.get(db_key, []), dtype=float)
    if p_db.ndim == 2:
        return np.power(10.0, p_db / 10.0)

    return np.array([], dtype=float)


def relative_spectrogram_db(
    time_s: np.ndarray,
    power_linear: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
    baseline_stat: str,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    power_linear = np.asarray(power_linear, dtype=float)
    if power_linear.ndim != 2 or len(time_s) == 0 or power_linear.shape[1] != len(time_s):
        return np.array([], dtype=float)

    m_base = (time_s >= float(baseline_start_s)) & (time_s <= float(baseline_end_s))
    if not np.any(m_base):
        m_base = time_s < 0.0
    if not np.any(m_base):
        m_base = np.isfinite(time_s)
    if not np.any(m_base):
        return np.array([], dtype=float)

    if str(baseline_stat).lower() == "mean":
        baseline = np.nanmean(power_linear[:, m_base], axis=1, keepdims=True)
    else:
        baseline = np.nanmedian(power_linear[:, m_base], axis=1, keepdims=True)
    baseline = np.where(np.isfinite(baseline) & (baseline > 0), baseline, np.nan)
    return db_from_linear(power_linear / baseline)


def relative_curve(
    time_s: np.ndarray,
    values: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
    baseline_stat: str,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(time_s) == 0 or values.shape != time_s.shape:
        return np.array([], dtype=float)

    m_base = (time_s >= float(baseline_start_s)) & (time_s <= float(baseline_end_s))
    if not np.any(m_base):
        m_base = time_s < 0.0
    if not np.any(m_base):
        m_base = np.isfinite(time_s)
    if not np.any(m_base):
        return np.array([], dtype=float)

    if str(baseline_stat).lower() == "mean":
        baseline = float(np.nanmean(values[m_base]))
    else:
        baseline = float(np.nanmedian(values[m_base]))
    if not np.isfinite(baseline) or baseline <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return np.asarray(values / baseline, dtype=float)


def spectrogram_display_args(limits: dict, source: str) -> tuple[Any, Any]:
    norm = None
    clim = None
    if SPECTROGRAM_VIEW == "relative":
        if SPECTROGRAM_SCALE_MODE == "manual":
            if source == "lfp":
                vmin, vmax = [float(v) for v in LFP_SPECTROGRAM_REL_DB_RANGE]
            else:
                vmin, vmax = [float(v) for v in GEVI_SPECTROGRAM_REL_DB_RANGE]
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            span_key = "lfp_spectrogram_rel_span" if source == "lfp" else "spectrogram_rel_span"
            span = safe_float(limits.get(span_key))
            if np.isfinite(span) and span > 0:
                norm = TwoSlopeNorm(vmin=-float(span), vcenter=0.0, vmax=float(span))
    else:
        clim_key = "lfp_spectrogram_clim" if source == "lfp" else "spectrogram_clim"
        if clim_key in limits:
            clim = limits[clim_key]
    return norm, clim


def train_spectrogram_baseline_meta(summary: dict, source: str = "spectrogram") -> tuple[float, float, str]:
    sec = summary.get("summary", {}).get("train_pta", {}).get(source, {})
    baseline_start_s = safe_float(sec.get("baseline_start_s"))
    baseline_end_s = safe_float(sec.get("baseline_end_s"))
    baseline_stat = str(sec.get("baseline_stat", SPECTROGRAM_BASELINE_STAT))
    if not np.isfinite(baseline_end_s):
        baseline_end_s = float(SPECTROGRAM_BASELINE_END_S)
    return baseline_start_s, baseline_end_s, baseline_stat


def train_hilbert_baseline_meta(summary: dict, source: str) -> tuple[float, float, str]:
    sec = summary.get("summary", {}).get("train_pta", {}).get(source, {})
    baseline_start_s = safe_float(sec.get("baseline_start_s"))
    baseline_end_s = safe_float(sec.get("baseline_end_s"))
    baseline_stat = str(sec.get("baseline_stat", "median"))
    if not np.isfinite(baseline_end_s):
        baseline_end_s = -0.5
    return baseline_start_s, baseline_end_s, baseline_stat


def trial_spectrogram_view(summary: dict, trial_name: str, source: str = "spectrogram") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    sec = train.get(source, {})
    t = np.asarray(sec.get("time_s", []), dtype=float)
    f = np.asarray(sec.get("freq_hz", []), dtype=float)
    p_linear = spectrogram_linear_from_section(sec)
    if len(t) == 0 or len(f) == 0 or p_linear.shape != (len(f), len(t)):
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    if SPECTROGRAM_VIEW == "relative":
        baseline_start_s, baseline_end_s, baseline_stat = train_spectrogram_baseline_meta(summary, source)
        if not np.isfinite(baseline_start_s):
            baseline_start_s = float(t[0])
        p_show = relative_spectrogram_db(t, p_linear, baseline_start_s, baseline_end_s, baseline_stat)
    else:
        p_show = np.asarray(sec.get("power_db", []), dtype=float)
        if p_show.shape != (len(f), len(t)):
            p_show = db_from_linear(p_linear)

    return t, f, np.asarray(p_show, dtype=float)


def summary_path_from_parts(mouse: str, date: str, block: str) -> Path:
    return DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block / f"{block}_summary.pkl"


def trial_sort_key(name: str):
    if "_" not in name:
        return (name, 0)
    stem, suffix = name.split("_", 1)
    try:
        return (stem, int(suffix))
    except ValueError:
        return (stem, suffix)


def get_trial_names(summary: dict) -> list[str]:
    stim_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    if stim_names:
        return sorted(stim_names, key=trial_sort_key)
    train_trials = summary.get("trials", {}).get("train_pta", {})
    return sorted(train_trials.keys(), key=trial_sort_key)


def print_ephys_summary(summary: dict, trial_names: list[str]) -> None:
    if not PRINT_EPHYS_SUMMARY:
        return

    ephys_trials = summary.get("trials", {}).get("ephys", {})
    processed_trials = summary.get("trials", {}).get("processed_notched", {})
    if not ephys_trials:
        print("[INFO] no ephys trial summary found")
        return

    print(f"\n{summary.get('date')} | {summary.get('block')}: trials={len(trial_names)}")
    print(
        "Trial | ephys samples | cam frames | GEVI frames | stim_on(s) | "
        "stim-cam(s) | cam_fps(Hz) | n_pulses | median_IPI(ms)"
    )
    print("-" * 112)

    for trial_name in trial_names:
        td_e = ephys_trials.get(trial_name, {})
        td_p = processed_trials.get(trial_name, {})
        t_e = np.asarray(td_e.get("t_stim_s", []), dtype=float)
        cam_frames = np.asarray(td_e.get("cam_frame_times_stim_s", []), dtype=float)
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        gevi = np.asarray(td_p.get("F_notched", td_p.get("dff", td_p.get("F_raw", []))), dtype=float)

        stim_on = safe_float(td_e.get("stim_on_s_block"))
        stim_cam = safe_float(td_e.get("stim_minus_cam_s"))
        cam_fps = safe_float(td_e.get("cam_fps_hz"))
        median_ipi = safe_float(td_e.get("median_ipi_ms"))

        print(
            f"{trial_name:>7} | {len(t_e):>13} | {len(cam_frames):>10} | {len(gevi):>11} | "
            f"{stim_on:>9.3f} | {stim_cam:>11.3f} | {cam_fps:>11.1f} | "
            f"{len(pulse_times):>8} | {median_ipi:>14.3f}"
        )


def get_trial_plv_phases(summary: dict, trial_name: str, section_key: str = "plv") -> np.ndarray:
    tr = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    phases = np.asarray(tr.get(section_key, {}).get("phase_pulses_rad", []), dtype=float)
    return phases[np.isfinite(phases)]


def phases_to_plv(phases: np.ndarray) -> tuple[float, float]:
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if len(phases) == 0:
        return np.nan, np.nan
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z))


def decimate_curve_bundle(x: np.ndarray, arrays: list[np.ndarray], max_points: int = LFP_DISPLAY_MAX_POINTS):
    x = np.asarray(x, dtype=float)
    clean_arrays = [np.asarray(arr, dtype=float) for arr in arrays]
    if len(x) == 0:
        return np.array([], dtype=float), [np.array([], dtype=float) for _ in clean_arrays]

    stride = max(1, int(np.ceil(len(x) / max(2, int(max_points)))))
    if stride == 1:
        return x.copy(), [arr.copy() if arr.shape == x.shape else np.array([], dtype=float) for arr in clean_arrays]

    idx = np.arange(0, len(x), stride, dtype=int)
    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return x[idx], [arr[idx] if arr.shape == x.shape else np.array([], dtype=float) for arr in clean_arrays]


def get_trial_lfp_display(summary: dict, trial_name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    y = np.asarray(td_e.get("channels", {}).get("LFP", []), dtype=float)
    if len(t) == 0 or y.shape != t.shape:
        return None, None
    if not USE_DECIMATED_LFP:
        return t, y
    t_show, [y_show] = decimate_curve_bundle(t, [y], max_points=LFP_DISPLAY_MAX_POINTS)
    return t_show, y_show


def get_trial_velocity_display(summary: dict, trial_name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(td_e.get("vel_bin_t_s", []), dtype=float)
    y = np.asarray(td_e.get("vel_bin_cmps", []), dtype=float)
    if len(t) == 0 or y.shape != t.shape:
        return None, None
    return t, y


def get_trial_stim_display(summary: dict, trial_name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
    if len(t) == 0 or y.shape != t.shape:
        return None, None
    return t, y


def get_trial_mean_pta_display(summary: dict, trial_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    tr = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    t = np.asarray(tr.get("t_rel_s", []), dtype=float)
    y = np.asarray(tr.get("pta_mean", []), dtype=float)
    s = np.asarray(tr.get("pta_spread", []), dtype=float)
    if len(t) == 0 or y.shape != t.shape:
        return None, None, None
    if s.shape != t.shape:
        s = np.full_like(y, np.nan, dtype=float)
    return t, y, s


def get_trial_hilbert_display(summary: dict, trial_name: str, source: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    sec = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {}).get(source, {})
    t = np.asarray(sec.get("display_time_s", sec.get("time_s", [])), dtype=float)
    y = np.asarray(sec.get("display_amplitude", sec.get("amplitude", [])), dtype=float)
    if len(t) == 0 or y.shape != t.shape:
        return None, None
    if HILBERT_VIEW == "relative":
        baseline_start_s, baseline_end_s, baseline_stat = train_hilbert_baseline_meta(summary, source)
        if not np.isfinite(baseline_start_s):
            baseline_start_s = float(t[0]) if len(t) else np.nan
        y = relative_curve(t, y, baseline_start_s, baseline_end_s, baseline_stat)
    return t, y


def build_single_trial_pulsogram(trial: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    segs = trial.get("segment_dicts", [])
    used_idx = np.asarray(trial.get("used_pulse_indices", []), dtype=int)
    if not segs or len(used_idx) == 0:
        return None, None, None

    pulse_numbers = []
    t_ref = None
    rows = []
    for pulse_idx, seg in zip(used_idx, segs):
        t_rel = np.asarray(seg.get("t_rel_s", []), dtype=float)
        signal = np.asarray(seg.get("signal", []), dtype=float)
        if len(t_rel) == 0 or signal.shape != t_rel.shape:
            continue
        if t_ref is None:
            t_ref = t_rel.copy()
        if t_rel.shape != t_ref.shape or not np.allclose(t_rel, t_ref):
            continue
        pulse_numbers.append(int(pulse_idx) + 1)
        rows.append(signal)

    if t_ref is None or not rows:
        return None, None, None
    return np.asarray(pulse_numbers, dtype=int), t_ref, np.vstack(rows)


def get_first_pta_segment(summary: dict, trial_name: str) -> dict | None:
    for seg in summary.get("trials", {}).get("first_pta_segments", []):
        if seg.get("trial") == trial_name:
            return seg
    return None


def choose_processed_signal(td: dict) -> np.ndarray:
    for key in ("F_notched", "F_bleach_corr", "F_raw"):
        x = td.get(key)
        if x is not None:
            return np.asarray(x, dtype=float)
    return np.array([], dtype=float)


def get_trial_ipi_s(summary: dict, trial_name: str) -> float:
    train = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    pulse_times = np.asarray(train.get("pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) >= 2:
        diffs = np.diff(pulse_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            return float(np.nanmedian(diffs))

    ephys = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    ipi_ms = safe_float(ephys.get("median_ipi_ms"))
    if np.isfinite(ipi_ms) and ipi_ms > 0:
        return 0.001 * float(ipi_ms)

    seg = get_first_pta_segment(summary, trial_name)
    second_t = safe_float(None if seg is None else seg.get("second_pulse_rel_s"))
    if np.isfinite(second_t) and second_t > 0:
        return float(second_t)

    return 1.0 / 135.0


def extract_first_pta_display_segment(summary: dict, trial_name: str) -> tuple[np.ndarray, np.ndarray, float, float] | tuple[None, None, float, float]:
    proc = summary.get("trials", {}).get("processed_notched", {}).get(trial_name, {})
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(proc.get("t", []), dtype=float)
    x = choose_processed_signal(proc)
    if len(t) == 0 or x.shape != t.shape:
        return None, None, np.nan, np.nan

    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) == 0:
        return None, None, np.nan, np.nan

    ipi_s = get_trial_ipi_s(summary, trial_name)
    if not np.isfinite(ipi_s) or ipi_s <= 0:
        ipi_s = 1.0 / 135.0
    x_hi = 3.0 * float(ipi_s)
    pre_s = float(ipi_s)

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


def collect_row_limits(summary: dict, trial_names: list[str]) -> dict:
    limits: dict[str, tuple[float, float]] = {}

    full_x = []
    full_y = []
    stim_x = []
    stim_y = []
    lfp_x = []
    lfp_y = []
    vel_x = []
    vel_y = []
    mean_pta_x = []
    mean_pta_y = []
    sig_hil_x = []
    sig_hil_y = []
    lfp_hil_x = []
    lfp_hil_y = []
    pta_x = []
    pta_y = []
    pulsogram_vals = []
    spectrogram_vals = []
    lfp_spectrogram_vals = []

    for trial_name in trial_names:
        train = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
        proc = summary.get("trials", {}).get("processed_notched", {}).get(trial_name, {})
        puls = summary.get("trials", {}).get("pulsogram", {}).get(trial_name, {})

        t_full = np.asarray(train.get("t_full_s", proc.get("t", [])), dtype=float)
        y_full = np.asarray(train.get("signal_full", proc.get("F_notched", [])), dtype=float)
        if len(t_full) and y_full.shape == t_full.shape:
            y_full = gevi_display(y_full)
            window = trial_panel_xlim(trial_name, "full_trace") or TRIAL_OVERVIEW_TIME_WINDOW_S
            y_win = values_in_time_window(t_full, y_full, window)
            full_x.extend([float(window[0]), float(window[1])])
            if len(y_win):
                full_y.extend([float(np.nanmin(y_win)), float(np.nanmax(y_win))])
            else:
                full_y.extend([float(np.nanmin(y_full)), float(np.nanmax(y_full))])

        t_stim, y_stim = get_trial_stim_display(summary, trial_name)
        if t_stim is not None and y_stim is not None and len(t_stim) and y_stim.shape == t_stim.shape:
            window = trial_panel_xlim(trial_name, "stim_trace") or TRIAL_OVERVIEW_TIME_WINDOW_S
            y_win = values_in_time_window(t_stim, y_stim, window)
            stim_x.extend([float(window[0]), float(window[1])])
            if len(y_win):
                stim_y.extend([float(np.nanmin(y_win)), float(np.nanmax(y_win))])
            else:
                stim_y.extend([float(np.nanmin(y_stim)), float(np.nanmax(y_stim))])

        t_lfp, y_lfp = get_trial_lfp_display(summary, trial_name)
        if t_lfp is not None and y_lfp is not None and len(t_lfp) and y_lfp.shape == t_lfp.shape:
            window = trial_panel_xlim(trial_name, "lfp") or TRIAL_OVERVIEW_TIME_WINDOW_S
            y_win = values_in_time_window(t_lfp, y_lfp, window)
            lfp_x.extend([float(window[0]), float(window[1])])
            if len(y_win):
                lfp_y.extend([float(np.nanmin(y_win)), float(np.nanmax(y_win))])
            else:
                lfp_y.extend([float(np.nanmin(y_lfp)), float(np.nanmax(y_lfp))])

        t_vel, y_vel = get_trial_velocity_display(summary, trial_name)
        if t_vel is not None and y_vel is not None and len(t_vel) and y_vel.shape == t_vel.shape:
            vel_x.extend([float(np.min(t_vel)), float(np.max(t_vel))])
            vel_y.extend([float(np.nanmin(y_vel)), float(np.nanmax(y_vel))])

        t_mpta, y_mpta, s_mpta = get_trial_mean_pta_display(summary, trial_name)
        if t_mpta is not None and y_mpta is not None and len(t_mpta) and y_mpta.shape == t_mpta.shape:
            y_mpta = gevi_display(y_mpta)
            if s_mpta is not None:
                s_mpta = gevi_display(s_mpta)
            mean_pta_x.extend([float(np.min(t_mpta)), float(np.max(t_mpta))])
            mean_pta_y.extend([float(np.nanmin(y_mpta)), float(np.nanmax(y_mpta))])
            if s_mpta is not None and s_mpta.shape == t_mpta.shape:
                mean_pta_y.extend([float(np.nanmin(y_mpta - s_mpta)), float(np.nanmax(y_mpta + s_mpta))])

        for harmonic in parse_harmonic_selection(PLOT_SIGNAL_HILBERT_HARMONICS):
            source = signal_hilbert_section_key(harmonic)
            t_sig_h, y_sig_h = get_trial_hilbert_display(summary, trial_name, source)
            if t_sig_h is not None and y_sig_h is not None and len(t_sig_h) and y_sig_h.shape == t_sig_h.shape:
                sig_hil_x.extend([float(np.min(t_sig_h)), float(np.max(t_sig_h))])
                sig_hil_y.extend([float(np.nanmin(y_sig_h)), float(np.nanmax(y_sig_h))])

        t_lfp_h, y_lfp_h = get_trial_hilbert_display(summary, trial_name, "lfp_hilbert")
        if t_lfp_h is not None and y_lfp_h is not None and len(t_lfp_h) and y_lfp_h.shape == t_lfp_h.shape:
            lfp_hil_x.extend([float(np.min(t_lfp_h)), float(np.max(t_lfp_h))])
            lfp_hil_y.extend([float(np.nanmin(y_lfp_h)), float(np.nanmax(y_lfp_h))])

        t_pta, y_pta, _, _ = extract_first_pta_display_segment(summary, trial_name)
        if t_pta is not None and y_pta is not None and len(t_pta) and y_pta.shape == t_pta.shape:
            y_pta = gevi_display(y_pta)
            mode = single_pta_mode()
            if mode == "derivative":
                deriv = first_derivative_curve(t_pta, y_pta)
                if deriv is not None:
                    td, yd = deriv
                    pta_x.extend([float(np.nanmin(td)), float(np.nanmax(td))])
                    pta_y.extend([float(np.nanmin(yd)), float(np.nanmax(yd))])
            else:
                pta_x.extend([float(np.min(t_pta)), float(np.max(t_pta))])
                pta_y.extend([float(np.nanmin(y_pta)), float(np.nanmax(y_pta))])

        _, _, p_spec = trial_spectrogram_view(summary, trial_name, source="spectrogram")
        if p_spec.ndim == 2:
            finite = p_spec[np.isfinite(p_spec)]
            if finite.size:
                spectrogram_vals.append(finite)
        _, _, p_lfp_spec = trial_spectrogram_view(summary, trial_name, source="lfp_spectrogram")
        if p_lfp_spec.ndim == 2:
            finite = p_lfp_spec[np.isfinite(p_lfp_spec)]
            if finite.size:
                lfp_spectrogram_vals.append(finite)

        _, t_puls, M = build_single_trial_pulsogram(puls)
        if t_puls is not None and M is not None:
            M = gevi_display(M)
            t_ms = 1000.0 * t_puls
            m = (t_ms >= PULSOGRAM_TIME_RANGE_MS[0]) & (t_ms <= PULSOGRAM_TIME_RANGE_MS[1])
            if np.sum(m) >= 2:
                vals = M[:, m]
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    pulsogram_vals.extend([float(np.nanmin(finite)), float(np.nanmax(finite))])

    if full_x and full_y:
        limits["full_trace_x"] = (min(full_x), max(full_x))
        limits["full_trace_y"] = expand_limits(min(full_y), max(full_y), frac=0.06)
    if stim_x and stim_y:
        limits["stim_trace_x"] = (min(stim_x), max(stim_x))
        limits["stim_trace_y"] = expand_limits(min(stim_y), max(stim_y), frac=0.06)
    if lfp_x and lfp_y:
        limits["lfp_x"] = (min(lfp_x), max(lfp_x))
        limits["lfp_y"] = expand_limits(min(lfp_y), max(lfp_y), frac=0.06)
    if vel_x and vel_y:
        limits["velocity_x"] = (min(vel_x), max(vel_x))
        limits["velocity_y"] = expand_limits(min(vel_y), max(vel_y), frac=0.06)
    if mean_pta_x and mean_pta_y:
        limits["mean_pta_x"] = (min(mean_pta_x), max(mean_pta_x))
        limits["mean_pta_y"] = expand_limits(min(mean_pta_y), max(mean_pta_y), frac=0.08)
    if sig_hil_x and sig_hil_y:
        limits["signal_hilbert_x"] = (min(sig_hil_x), max(sig_hil_x))
        limits["signal_hilbert_y"] = expand_limits(min(sig_hil_y), max(sig_hil_y), frac=0.08)
    if lfp_hil_x and lfp_hil_y:
        limits["lfp_hilbert_x"] = (min(lfp_hil_x), max(lfp_hil_x))
        limits["lfp_hilbert_y"] = expand_limits(min(lfp_hil_y), max(lfp_hil_y), frac=0.08)
    if pta_x and pta_y:
        lo = min(-float(SINGLE_PTA_PRE_SEC), min(pta_x))
        hi = max(pta_x)
        limits["single_pta_x"] = (lo, hi)
        limits["single_pta_y"] = expand_limits(min(pta_y), max(pta_y), frac=0.08)
    if spectrogram_vals:
        all_vals = np.concatenate(spectrogram_vals)
        if SPECTROGRAM_VIEW == "relative":
            lo, hi = [float(v) for v in SPECTROGRAM_REL_PERCENTILES]
            q_lo, q_hi = np.percentile(all_vals, [lo, hi])
            limits["spectrogram_rel_span"] = max(abs(float(q_lo)), abs(float(q_hi)))
        else:
            lo, hi = [float(v) for v in SPECTROGRAM_ABS_PERCENTILES]
            q_lo, q_hi = np.percentile(all_vals, [lo, hi])
            limits["spectrogram_clim"] = (float(q_lo), float(q_hi))
    if lfp_spectrogram_vals:
        all_vals = np.concatenate(lfp_spectrogram_vals)
        if SPECTROGRAM_VIEW == "relative":
            lo, hi = [float(v) for v in SPECTROGRAM_REL_PERCENTILES]
            q_lo, q_hi = np.percentile(all_vals, [lo, hi])
            limits["lfp_spectrogram_rel_span"] = max(abs(float(q_lo)), abs(float(q_hi)))
        else:
            lo, hi = [float(v) for v in SPECTROGRAM_ABS_PERCENTILES]
            q_lo, q_hi = np.percentile(all_vals, [lo, hi])
            limits["lfp_spectrogram_clim"] = (float(q_lo), float(q_hi))
    if pulsogram_vals:
        vmax = max(abs(min(pulsogram_vals)), abs(max(pulsogram_vals)))
        limits["pulsogram_v"] = (-vmax, vmax)
    return limits


def expand_limits(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    if hi <= lo:
        pad = 1.0 if lo == 0 else 0.1 * abs(lo)
        return (lo - pad, hi + pad)
    pad = frac * (hi - lo)
    return (lo - pad, hi + pad)


def style_axis(ax, show_ylabel: bool, ylabel: str | None = None):
    ax.tick_params(labelsize=8)
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_label_coords(-0.36, 0.5)
    else:
        ax.set_ylabel("")


def add_row_label(ax, row_idx: int):
    label = chr(ord("a") + row_idx)
    ax.text(
        ROW_LABEL_X,
        ROW_LABEL_Y,
        label,
        transform=ax.transAxes,
        fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def plot_full_trace_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    train = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    proc = summary.get("trials", {}).get("processed_notched", {}).get(trial_name, {})

    t = np.asarray(train.get("t_full_s", proc.get("t", [])), dtype=float)
    y = np.asarray(train.get("signal_full", proc.get("F_notched", [])), dtype=float)
    pulse_times = np.asarray(train.get("pulse_times_s", []), dtype=float)

    if len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return

    y = gevi_display(y)
    ax.plot(t, y, color="tab:blue", lw=FULL_TRACE_LINEWIDTH)
    if len(pulse_times):
        ax.axvline(0.0, color="tab:red", ls="--", lw=0.9, alpha=0.9)
        ax.axvline(float(pulse_times[-1]), color="tab:orange", ls="--", lw=0.9, alpha=0.9)

    xlim = trial_panel_xlim(trial_name, "full_trace")
    ax.set_xlim(*(xlim or TRIAL_OVERVIEW_TIME_WINDOW_S))
    if "full_trace_y" in limits:
        ax.set_ylim(*limits["full_trace_y"])
    tighten_ylim_to_xlim(ax, t, y, xlim)
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, GEVI_YLABEL)


def plot_lfp_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, y = get_trial_lfp_display(summary, trial_name)
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]

    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return

    ax.plot(t, y, color="tab:green", lw=FULL_TRACE_LINEWIDTH)
    if len(pulse_times):
        ax.axvline(0.0, color="tab:red", ls="--", lw=0.9, alpha=0.9)
        ax.axvline(float(pulse_times[-1]), color="tab:orange", ls="--", lw=0.9, alpha=0.9)

    xlim = trial_panel_xlim(trial_name, "lfp")
    ax.set_xlim(*(xlim or TRIAL_OVERVIEW_TIME_WINDOW_S))
    if "lfp_y" in limits:
        ax.set_ylim(*limits["lfp_y"])
    tighten_ylim_to_xlim(ax, t, y, xlim)
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "LFP")


def plot_velocity_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, y = get_trial_velocity_display(summary, trial_name)
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]

    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return

    ax.plot(t, y, color="tab:brown", lw=FULL_TRACE_LINEWIDTH)
    if len(pulse_times):
        ax.axvline(0.0, color="tab:red", ls="--", lw=0.9, alpha=0.9)
        ax.axvline(float(pulse_times[-1]), color="tab:orange", ls="--", lw=0.9, alpha=0.9)

    if "velocity_x" in limits:
        ax.set_xlim(*limits["velocity_x"])
    if "velocity_y" in limits:
        ax.set_ylim(*limits["velocity_y"])
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "velocity (cm/s)")


def plot_stim_trace_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, y = get_trial_stim_display(summary, trial_name)
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]

    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return

    ax.plot(t, y, color="tab:purple", lw=FULL_TRACE_LINEWIDTH)
    if len(pulse_times):
        ax.axvline(0.0, color="tab:red", ls="--", lw=0.9, alpha=0.9)
        ax.axvline(float(pulse_times[-1]), color="tab:orange", ls="--", lw=0.9, alpha=0.9)

    xlim = trial_panel_xlim(trial_name, "stim_trace")
    ax.set_xlim(*(xlim or TRIAL_OVERVIEW_TIME_WINDOW_S))
    if "stim_trace_y" in limits:
        ax.set_ylim(*limits["stim_trace_y"])
    tighten_ylim_to_xlim(ax, t, y, xlim)
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "stim (V)")


def plot_single_pta_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    mode = single_pta_mode()
    t, y, second_t, pre_s = extract_first_pta_display_segment(summary, trial_name)
    if t is None or y is None:
        ax.axis("off")
        return
    if len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return

    y = gevi_display(y)
    deriv = first_derivative_curve(t, y) if mode in {"derivative", "both"} else None
    if mode == "derivative":
        if deriv is None:
            ax.axis("off")
            return
        td, yd = deriv
        ax.plot(td, yd, color="tab:green", lw=SINGLE_PTA_LINEWIDTH)
        ax.axhline(0, color="0.6", lw=0.7)
    else:
        ax.plot(t, y, color="black", lw=SINGLE_PTA_LINEWIDTH)
        if mode == "both" and deriv is not None:
            td, yd = deriv
            ax2 = ax.twinx()
            ax2.plot(td, yd, color="tab:green", lw=1.0, alpha=0.9)
            ax2.axhline(0, color="tab:green", lw=0.6, alpha=0.5)
            ax2.tick_params(axis="y", labelsize=7, labelcolor="tab:green")
    ax.axvline(0.0, color="tab:red", ls="--", lw=0.9)
    if np.isfinite(second_t) and float(np.min(t)) <= second_t <= float(np.max(t)):
        ax.axvline(second_t, color="tab:orange", ls="--", lw=0.9)

    ipi_s = get_trial_ipi_s(summary, trial_name)
    xlim = trial_panel_xlim(trial_name, "single_pta")
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(-float(pre_s), 3.0 * float(ipi_s))
    if "single_pta_y" in limits:
        ax.set_ylim(*limits["single_pta_y"])
    if mode == "derivative" and deriv is not None:
        tighten_ylim_to_xlim(ax, td, yd, xlim)
    else:
        tighten_ylim_to_xlim(ax, t, y, xlim)
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "First Pulse 1st")


def plot_mean_pta_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, y, s = get_trial_mean_pta_display(summary, trial_name)
    tr = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    pulse_times = np.asarray(tr.get("pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]

    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return

    y = gevi_display(y)
    if s is not None:
        s = gevi_display(s)
    ax.plot(t, y, color="black", lw=SINGLE_PTA_LINEWIDTH)
    if s is not None and s.shape == t.shape:
        ax.fill_between(t, y - s, y + s, color="black", alpha=0.2)
    ax.axvline(0.0, color="tab:red", ls="--", lw=0.9)
    if len(pulse_times) >= 2:
        second_t = float(pulse_times[1] - pulse_times[0])
        if float(np.min(t)) <= second_t <= float(np.max(t)):
            ax.axvline(second_t, color="tab:orange", ls="--", lw=0.9)

    if "mean_pta_x" in limits:
        ax.set_xlim(*limits["mean_pta_x"])
    if "mean_pta_y" in limits:
        ax.set_ylim(*limits["mean_pta_y"])
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, GEVI_YLABEL)


def plot_signal_hilbert_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool, harmonic: int = 1):
    source = signal_hilbert_section_key(harmonic)
    label = signal_hilbert_label(harmonic)
    t, y = get_trial_hilbert_display(summary, trial_name, source)
    sec = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {}).get(source, {})
    f_center = safe_float(sec.get("f_center_hz"))
    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return
    ax.plot(t, y, color="tab:blue", lw=1.0)
    ax.axvline(0.0, color="tab:red", ls="--", lw=0.9, alpha=0.9)
    if "signal_hilbert_x" in limits:
        ax.set_xlim(*limits["signal_hilbert_x"])
    if "signal_hilbert_y" in limits:
        ax.set_ylim(*limits["signal_hilbert_y"])
    ax.set_title(f"{trial_name} | {label}", fontsize=8, pad=4)
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "amp / baseline" if HILBERT_VIEW == "relative" else "amplitude")


def plot_lfp_hilbert_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, y = get_trial_hilbert_display(summary, trial_name, "lfp_hilbert")
    sec = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {}).get("lfp_hilbert", {})
    f_center = safe_float(sec.get("f_center_hz"))
    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return
    ax.plot(t, y, color="tab:green", lw=1.0)
    ax.axvline(0.0, color="tab:red", ls="--", lw=0.9, alpha=0.9)
    if "lfp_hilbert_x" in limits:
        ax.set_xlim(*limits["lfp_hilbert_x"])
    if "lfp_hilbert_y" in limits:
        ax.set_ylim(*limits["lfp_hilbert_y"])
    ax.set_title(f"{trial_name} | LFP Hilbert", fontsize=8, pad=4)
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "amp / baseline" if HILBERT_VIEW == "relative" else "amplitude")


def plot_spectrogram_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, f, p = trial_spectrogram_view(summary, trial_name, source="spectrogram")
    if len(t) == 0 or len(f) == 0 or p.shape != (len(f), len(t)):
        ax.axis("off")
        return

    cmap = SPECTROGRAM_REL_CMAP if SPECTROGRAM_VIEW == "relative" else SPECTROGRAM_ABS_CMAP
    norm, clim = spectrogram_display_args(limits, source="gevi")

    im = ax.imshow(
        p,
        origin="lower",
        aspect="auto",
        extent=[float(t[0]), float(t[-1]), float(f[0]), float(f[-1])],
        cmap=cmap,
        interpolation=SPECTROGRAM_INTERPOLATION,
        norm=norm,
    )
    if norm is None and clim is not None:
        im.set_clim(*clim)

    ax.set_ylim(0, min(float(SPEC_FMAX_HZ), float(np.max(f))))
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "frequency (Hz)")


def plot_lfp_spectrogram_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    t, f, p = trial_spectrogram_view(summary, trial_name, source="lfp_spectrogram")
    if len(t) == 0 or len(f) == 0 or p.shape != (len(f), len(t)):
        ax.axis("off")
        return

    cmap = SPECTROGRAM_REL_CMAP if SPECTROGRAM_VIEW == "relative" else SPECTROGRAM_ABS_CMAP
    norm, clim = spectrogram_display_args(limits, source="lfp")

    im = ax.imshow(
        p,
        origin="lower",
        aspect="auto",
        extent=[float(t[0]), float(t[-1]), float(f[0]), float(f[-1])],
        cmap=cmap,
        interpolation=SPECTROGRAM_INTERPOLATION,
        norm=norm,
    )
    if norm is None and clim is not None:
        im.set_clim(*clim)

    ax.set_ylim(0, min(float(SPEC_FMAX_HZ), float(np.max(f))))
    ax.set_xlabel("time (s)", fontsize=8)
    style_axis(ax, show_ylabel, "frequency (Hz)")


def plot_plv_histogram_trial_common(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool, section_key: str, label: str):
    tr = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    phases = get_trial_plv_phases(summary, trial_name, section_key=section_key)
    if len(phases) == 0:
        ax.axis("off")
        return
    center = safe_float(tr.get(section_key, {}).get("f_center_hz"))
    plv, pref = phases_to_plv(phases)
    counts, edges = np.histogram(phases, bins=np.linspace(-np.pi, np.pi, 25))
    rmax = max(1.0, float(np.max(counts)))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="tab:blue", alpha=0.45, edgecolor="white", linewidth=0.3)
    ax.annotate("", xy=(pref, plv * rmax), xytext=(0, 0), arrowprops=dict(color="crimson", lw=1.8, arrowstyle="->"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_yticklabels([])
    center_text = f" | {center:.1f} Hz" if np.isfinite(center) else ""
    ax.set_title(f"{trial_name} | {label}={plv:.3f}{center_text}", fontsize=8, pad=4)


def plot_plv_histogram_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    plot_plv_histogram_trial_common(ax, summary, trial_name, limits, show_ylabel, section_key="plv", label="PLV")


def plot_plv_h2_histogram_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    plot_plv_histogram_trial_common(ax, summary, trial_name, limits, show_ylabel, section_key="plv_h2", label="PLV H2")


def make_plv_harmonic_plotter(harmonic: int):
    def plotter(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
        plot_plv_histogram_trial_common(
            ax,
            summary,
            trial_name,
            limits,
            show_ylabel,
            section_key=plv_section_key(harmonic),
            label=plv_label(harmonic),
        )

    plotter.needs_polar = True
    return plotter


plot_plv_histogram_trial.needs_polar = True
plot_plv_h2_histogram_trial.needs_polar = True


def plot_pulsogram_trial(ax, summary: dict, trial_name: str, limits: dict, show_ylabel: bool):
    trial = summary.get("trials", {}).get("pulsogram", {}).get(trial_name, {})
    pulse_numbers, t_rel, M = build_single_trial_pulsogram(trial)
    if pulse_numbers is None or t_rel is None or M is None:
        ax.axis("off")
        return

    t_ms = 1000.0 * t_rel
    lo_ms, hi_ms = PULSOGRAM_TIME_RANGE_MS
    m = (t_ms >= lo_ms) & (t_ms <= hi_ms)
    if np.sum(m) < 2:
        ax.axis("off")
        return

    im_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "extent": [float(t_ms[m][0]), float(t_ms[m][-1]), int(pulse_numbers[0]), int(pulse_numbers[-1])],
        "cmap": "RdBu_r",
    }
    if "pulsogram_v" in limits:
        im_kwargs["vmin"], im_kwargs["vmax"] = limits["pulsogram_v"]
    ax.imshow(gevi_display(M[:, m]), **im_kwargs)
    ax.axvline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time (ms)", fontsize=8)
    style_axis(ax, show_ylabel, "pulse #")


def build_rows():
    rows = []
    if PLOT_FULL_TRACE:
        rows.append(("GEVI trace", plot_full_trace_trial))
    if PLOT_STIM_TRACE:
        rows.append(("stim (V)", plot_stim_trace_trial))
    if PLOT_LFP:
        rows.append(("LFP", plot_lfp_trial))
    if PLOT_VELOCITY:
        rows.append(("velocity (cm/s)", plot_velocity_trial))
    if PLOT_PULSE_WINDOWS:
        rows.append(("Pulse train", plot_mean_pta_trial))
    if PLOT_SIGNAL_HILBERT:
        for harmonic in parse_harmonic_selection(PLOT_SIGNAL_HILBERT_HARMONICS):
            rows.append((
                signal_hilbert_label(harmonic),
                lambda ax, summary, trial_name, limits, show_ylabel, harmonic=harmonic: plot_signal_hilbert_trial(
                    ax, summary, trial_name, limits, show_ylabel, harmonic
                ),
            ))
    if PLOT_LFP_HILBERT:
        rows.append(("LFP Hilbert", plot_lfp_hilbert_trial))
    mode = single_pta_mode()
    if mode != "off":
        label = "First Pulse 1st"
        rows.append((label, plot_single_pta_trial))
    if PLOT_SPECTROGRAM:
        rows.append(("Spectrogram", plot_spectrogram_trial))
    if PLOT_LFP_SPECTROGRAM:
        rows.append(("LFP Spectrogram", plot_lfp_spectrogram_trial))
    harmonic_selection = parse_harmonic_selection(PLOT_PLV_HISTOGRAMS)
    if PLOT_PLV_HISTOGRAM and 1 not in harmonic_selection:
        harmonic_selection = [1] + harmonic_selection
    for harmonic in harmonic_selection:
        rows.append((f"{plv_label(harmonic)} Histogram", make_plv_harmonic_plotter(harmonic)))
    if PLOT_PULSOGRAM:
        rows.append(("Pulsogram", plot_pulsogram_trial))
    return rows


def plot_all_trials(summary: dict, save_path: Path | None = None):
    trial_names = get_trial_names(summary)
    rows = build_rows()
    if not trial_names:
        print("[SKIP] no trial data found")
        return
    print_ephys_summary(summary, trial_names)
    if not rows:
        print("[SKIP] no active plot toggles")
        return

    limits = collect_row_limits(summary, trial_names)
    n_rows = len(rows)
    n_cols = len(trial_names)

    fig_w = max(18.0, 2.5 * n_cols)
    fig_h = max(10.0, 2.35 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for col, trial_name in enumerate(trial_names):
        axes[0, col].set_title(trial_name, fontsize=10, pad=8)

    for row_idx, (row_name, plot_fn) in enumerate(rows):
        for col_idx, trial_name in enumerate(trial_names):
            ax = axes[row_idx, col_idx]
            if getattr(plot_fn, "needs_polar", False):
                spec = ax.get_subplotspec()
                ax.remove()
                ax = fig.add_subplot(spec, projection="polar")
                axes[row_idx, col_idx] = ax
            plot_fn(ax, summary, trial_name, limits, show_ylabel=(col_idx == 0))
            if col_idx != 0:
                ax.tick_params(labelleft=False)
            if row_idx != n_rows - 1:
                ax.tick_params(labelbottom=True)
        add_row_label(axes[row_idx, 0], row_idx)
    plt.tight_layout(rect=[0.02, 0.03, 1.0, 0.99])

    if SAVE_FIGURE and save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot all trials in one block as a compact overview grid.")
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
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / f"{slugify_output_name(args.mouse, args.date, args.block)}_all_trials.png"
    plot_all_trials(summary, save_path=out_path)


if __name__ == "__main__":
    main()
