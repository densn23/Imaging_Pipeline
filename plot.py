from __future__ import annotations

import argparse
import math
import pickle
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from config import DATA_ANALYSIS_ROOT


FIGURES_DIR = DATA_ANALYSIS_ROOT / "figures"

MOUSE_NAME = "Jamie11"
SINGLE_DATE = "29-04-26"
SINGLE_BLOCK = "R2"


# -------------------------
# EXECUTION TOGGLES
# -------------------------
SHOW_FIGURE = True
SAVE_FIGURE = False
FIG_DPI = 300


# -------------------------
# PANEL TOGGLES
# -------------------------
PLOT_SINGLE_PTA = "derivative"  # False/off, True/"normal", "derivative", or "both"
PLOT_FULL_TRACE = True
PLOT_STIM_TRACE = False
PLOT_LFP = False
PLOT_VELOCITY = False
PLOT_CAMERA_FRAMES = False
PLOT_PULSE_WINDOWS = True
PLOT_FFT = False
PLOT_SPECTROGRAM = True
PLOT_LFP_SPECTROGRAM = False
PLOT_PLV_HISTOGRAMS = "1"  # 1=stim frequency, 2=2x stim, 3=3x stim; e.g. "1+2+3"
PLOT_BAND_SPECTRUM = False
PLOT_PULSOGRAM = True
PLOT_SIGNAL_HILBERT = False
PLOT_SIGNAL_HILBERT_HARMONICS = "1"  # e.g. "1", "2", "1+2+3", or "5+6"
PLOT_LFP_HILBERT = False


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
PULSOGRAM_TIME_RANGE_MS = (-100.0, 100.0)
LFP_DISPLAY_MAX_POINTS = 12000
USE_DECIMATED_LFP = False  # False = plot full 30 kHz LFP traces
STIM_TRACE_DISPLAY_MAX_POINTS = 12000
CAMERA_FRAME_DISPLAY_STRIDE = 100
SPECTROGRAM_VIEW = "relative"  # "absolute" or "relative"
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
GEVI_DISPLAY_SCALE = 100.0  # stored dF/F is fractional; plots show percent dF/F
GEVI_YLABEL = "dF/F"
GEVI_DERIV_YLABEL = "d(dF/F)/dt (%/s)"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def slugify_output_name(*parts) -> str:
    text = "_".join(str(part) for part in parts if part is not None and str(part).strip())
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "figure"


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


def gevi_display(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * float(GEVI_DISPLAY_SCALE)


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


def harmonic_frequency_text(harmonic: int) -> str:
    harmonic = int(harmonic)
    return "DBS frequency" if harmonic == 1 else f"{harmonic}x DBS frequency"


def hilbert_panel_title(harmonic: int) -> str:
    return f"{harmonic_frequency_text(harmonic)} amplitude"


def plv_panel_title(harmonic: int) -> str:
    return f"Phase locking at {harmonic_frequency_text(harmonic)}"


def db_from_linear(power_linear: np.ndarray) -> np.ndarray:
    power_linear = np.asarray(power_linear, dtype=float)
    if power_linear.size == 0:
        return np.asarray(power_linear, dtype=float)
    return 10.0 * np.log10(np.maximum(power_linear, 1e-30))


def spectrogram_linear_from_section(sec: dict, linear_key: str, db_key: str) -> np.ndarray:
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


def transform_spectrogram_for_display(p_show: np.ndarray, view: str, source: str = "gevi"):
    p_show = np.asarray(p_show, dtype=float)
    if p_show.ndim != 2:
        return p_show, "PSD (dB)", SPECTROGRAM_ABS_CMAP, None, None

    finite = p_show[np.isfinite(p_show)]
    if finite.size == 0:
        cmap = SPECTROGRAM_REL_CMAP if view == "relative" else SPECTROGRAM_ABS_CMAP
        label = "relative power (dB)" if view == "relative" else "PSD (dB)"
        return p_show, label, cmap, None, None

    if view == "relative":
        if SPECTROGRAM_SCALE_MODE == "manual":
            if source == "lfp":
                vmin, vmax = [float(v) for v in LFP_SPECTROGRAM_REL_DB_RANGE]
            else:
                vmin, vmax = [float(v) for v in GEVI_SPECTROGRAM_REL_DB_RANGE]
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                return p_show, "relative power (dB)", SPECTROGRAM_REL_CMAP, None, norm
        lo, hi = [float(v) for v in SPECTROGRAM_REL_PERCENTILES]
        v_lo, v_hi = np.percentile(finite, [lo, hi])
        span = float(max(abs(v_lo), abs(v_hi)))
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span) if span > 0 else None
        return p_show, "relative power (dB)", SPECTROGRAM_REL_CMAP, None, norm

    lo, hi = [float(v) for v in SPECTROGRAM_ABS_PERCENTILES]
    vmin, vmax = np.percentile(finite, [lo, hi])
    clim = (float(vmin), float(vmax)) if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin else None
    return p_show, "PSD (dB)", SPECTROGRAM_ABS_CMAP, clim, None


def summary_path_from_parts(mouse: str, date: str, block: str) -> Path:
    return DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block / f"{block}_summary.pkl"


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
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    tol = max(1e-12, 0.25 * float(np.nanmedian(np.diff(x))))
    keep = (x_ref >= x[0] - tol) & (x_ref <= x[-1] + tol)
    out = np.full_like(x_ref, np.nan, dtype=float)
    if np.any(keep):
        out[keep] = np.interp(np.clip(x_ref[keep], x[0], x[-1]), x, y)
    return out


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


def choose_grid(n_panels: int) -> tuple[int, int]:
    n_cols = min(3, max(1, int(math.ceil(math.sqrt(n_panels)))))
    n_rows = int(math.ceil(n_panels / n_cols))
    return n_rows, n_cols


def stim_end_time_s(summary: dict, trial_name: str | None = None) -> float:
    ephys = stim_trace_ephys_trials(summary)
    if trial_name is not None:
        names = [trial_name]
    else:
        names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]

    last_pulses = []
    for name in names:
        td_e = ephys.get(name, {})
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        if len(pulse_times):
            last_pulses.append(float(pulse_times[-1]))
    return float(np.nanmedian(last_pulses)) if last_pulses else np.nan


def stim_frequency_hz(summary: dict, trial_name: str | None = None) -> float:
    if trial_name is not None:
        tr = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
        f = safe_float(tr.get("f_stim_hz"))
        if np.isfinite(f):
            return f
    return safe_float(summary.get("summary", {}).get("train_pta", {}).get("metrics", {}).get("f_stim_hz_mean"))


def figure_title(summary: dict) -> str:
    mode_label = "block mean" if ONLY_TRIAL is None else f"trial {ONLY_TRIAL}"
    title = f"{summary.get('mouse')} | {summary.get('date')} {summary.get('block')} | {mode_label}"
    f_stim = stim_frequency_hz(summary, ONLY_TRIAL)
    if np.isfinite(f_stim):
        title += f" | {f_stim:.1f} Hz DBS"
    return title


def has_block_stim_data(summary: dict) -> bool:
    train = summary.get("summary", {}).get("train_pta", {})
    return bool(train.get("available")) and int(train.get("n_trials", 0)) > 0


def has_block_baseline_spectrogram(summary: dict) -> bool:
    sec = summary.get("summary", {}).get("processed_notched", {}).get("baseline_spectrogram", {})
    t = np.asarray(sec.get("time_s", []), dtype=float)
    f = np.asarray(sec.get("freq_hz", []), dtype=float)
    p = np.asarray(sec.get("power_db", []), dtype=float)
    return len(t) > 0 and len(f) > 0 and p.shape == (len(f), len(t))


def has_block_lfp_data(summary: dict) -> bool:
    sec = summary.get("summary", {}).get("lfp", {})
    t_key = "t_stim_s_display" if USE_DECIMATED_LFP else "t_stim_s_full"
    y_key = "mean_display" if USE_DECIMATED_LFP else "mean_full"
    t = np.asarray(sec.get(t_key, sec.get("t_stim_s_full", [])), dtype=float)
    y = np.asarray(sec.get(y_key, sec.get("mean_full", [])), dtype=float)
    return bool(sec.get("available")) and len(t) > 1 and y.shape == t.shape


def has_trial_lfp_data(summary: dict, trial_name: str | None) -> bool:
    if not trial_name:
        return False
    td_e = summary.get("trials", {}).get("ephys", {}).get(trial_name, {})
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    y = np.asarray(td_e.get("channels", {}).get("LFP", []), dtype=float)
    return len(t) > 1 and y.shape == t.shape


def has_block_stim_trace_data(summary: dict) -> bool:
    stim_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    ephys_trials = stim_trace_ephys_trials(summary)
    for name in stim_names:
        td_e = ephys_trials.get(name, {})
        t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
        y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
        if len(t) > 1 and y.shape == t.shape:
            return True
    return False


def has_trial_stim_trace_data(summary: dict, trial_name: str | None) -> bool:
    if not trial_name:
        return False
    td_e = stim_trace_ephys_trials(summary).get(trial_name, {})
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
    return len(t) > 1 and y.shape == t.shape


def stim_trial_names(summary: dict) -> list[str]:
    names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    if names:
        return names
    return sorted(stim_trace_ephys_trials(summary).keys())


def has_block_velocity_data(summary: dict) -> bool:
    ephys_trials = stim_trace_ephys_trials(summary)
    for name in stim_trial_names(summary):
        td_e = ephys_trials.get(name, {})
        t = np.asarray(td_e.get("vel_bin_t_s", []), dtype=float)
        y = np.asarray(td_e.get("vel_bin_cmps", []), dtype=float)
        if len(t) > 1 and y.shape == t.shape:
            return True
    return False


def has_trial_velocity_data(summary: dict, trial_name: str | None) -> bool:
    if not trial_name:
        return False
    td_e = stim_trace_ephys_trials(summary).get(trial_name, {})
    t = np.asarray(td_e.get("vel_bin_t_s", []), dtype=float)
    y = np.asarray(td_e.get("vel_bin_cmps", []), dtype=float)
    return len(t) > 1 and y.shape == t.shape


def has_block_camera_frame_data(summary: dict) -> bool:
    ephys_trials = stim_trace_ephys_trials(summary)
    for name in stim_trial_names(summary):
        t = np.asarray(ephys_trials.get(name, {}).get("cam_frame_times_stim_s", []), dtype=float)
        if len(t) > 1:
            return True
    return False


def has_trial_camera_frame_data(summary: dict, trial_name: str | None) -> bool:
    if not trial_name:
        return False
    t = np.asarray(stim_trace_ephys_trials(summary).get(trial_name, {}).get("cam_frame_times_stim_s", []), dtype=float)
    return len(t) > 1


def stim_trace_ephys_trials(summary: dict) -> dict:
    if summary.get("analysis") == "first_constant_train_only":
        src = summary.get("source_paths", {}).get("epoched_ephys")
        if src:
            path = Path(src)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        raw = pickle.load(f)
                    trials = raw.get("trials", {})
                    if isinstance(trials, dict):
                        return trials
                except Exception:
                    pass
    trials = summary.get("trials", {}).get("ephys", {})
    return trials if isinstance(trials, dict) else {}


def has_block_hilbert_data(summary: dict, source: str) -> bool:
    sec = summary.get("summary", {}).get("train_pta", {}).get(source, {})
    t = np.asarray(sec.get("time_s_display", sec.get("time_s_full", [])), dtype=float)
    if HILBERT_VIEW == "relative":
        y = np.asarray(sec.get("relative_mean_display", sec.get("relative_mean_full", [])), dtype=float)
    else:
        y = np.asarray(sec.get("amplitude_mean_display", sec.get("amplitude_mean_full", [])), dtype=float)
    return len(t) > 1 and y.shape == t.shape


def has_trial_hilbert_data(summary: dict, trial_name: str | None, source: str) -> bool:
    if not trial_name:
        return False
    sec = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {}).get(source, {})
    t = np.asarray(sec.get("time_s", []), dtype=float)
    y = np.asarray(sec.get("amplitude", []), dtype=float)
    return len(t) > 1 and y.shape == t.shape


def has_trial_stim_data(summary: dict, trial_name: str | None) -> bool:
    if not trial_name:
        return False
    return trial_name in summary.get("trials", {}).get("train_pta", {})


def phases_to_plv(phases: np.ndarray) -> tuple[float, float]:
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if len(phases) == 0:
        return np.nan, np.nan
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z))


def trial_plv_phases(tr: dict, section_key: str = "plv") -> np.ndarray:
    phases = np.asarray(tr.get(section_key, {}).get("phase_pulses_rad", []), dtype=float)
    return phases[np.isfinite(phases)]


def block_plv_phases(summary: dict, section_key: str = "plv") -> np.ndarray:
    sec = summary.get("summary", {}).get("train_pta", {}).get(section_key, {})
    phases = np.asarray(sec.get("phase_pulses_rad", []), dtype=float)
    phases = phases[np.isfinite(phases)]
    if len(phases):
        return phases

    out = []
    for tr in summary.get("trials", {}).get("train_pta", {}).values():
        ph = trial_plv_phases(tr, section_key=section_key)
        if len(ph):
            out.append(ph)
    return np.concatenate(out) if out else np.array([], dtype=float)


def has_block_plv_data(summary: dict, section_key: str = "plv") -> bool:
    return len(block_plv_phases(summary, section_key=section_key)) > 0


def has_trial_plv_data(summary: dict, trial_name: str | None, section_key: str = "plv") -> bool:
    if not trial_name:
        return False
    tr = summary.get("trials", {}).get("train_pta", {}).get(trial_name, {})
    return len(trial_plv_phases(tr, section_key=section_key)) > 0


def train_hilbert_baseline_meta(summary: dict, source: str) -> tuple[float, float, str]:
    sec = summary.get("summary", {}).get("train_pta", {}).get(source, {})
    baseline_start_s = safe_float(sec.get("baseline_start_s"))
    baseline_end_s = safe_float(sec.get("baseline_end_s"))
    baseline_stat = str(sec.get("baseline_stat", "median"))
    if not np.isfinite(baseline_end_s):
        baseline_end_s = -0.5
    return baseline_start_s, baseline_end_s, baseline_stat


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
    mode = single_pta_mode()
    second_times = []
    ipi_vals = []
    if ONLY_TRIAL is None:
        sec = summary["summary"]["single_pta"]
        stim_names = summary["trials"].get("stim_trial_names", [])
        display = sec.get("display", {})
        t = np.asarray(display.get("t_rel_s", []), dtype=float)
        y = np.asarray(display.get("mean", []), dtype=float)
        s = np.asarray(display.get("sd", []), dtype=float)
        second_t = safe_float(display.get("second_pulse_rel_s_mean"))
        if np.isfinite(second_t) and second_t > 0:
            second_times = [second_t]
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
        ipi_s = float(np.nanmedian(ipi_vals)) if ipi_vals else second_t
        if not np.isfinite(ipi_s) or ipi_s <= 0:
            ipi_s = 1.0 / 135.0
        x_hi = 3.0 * float(ipi_s)

        saved_display_ok = False
        if len(t) and y.shape == t.shape:
            finite_idx = np.flatnonzero(np.isfinite(y))
            if finite_idx.size:
                last_finite_t = float(t[finite_idx[-1]])
                # Some saved single-PTA displays only carry ~1 pulse period of real data.
                # If that happens, rebuild the intended 3-period view from the stored trial traces.
                saved_display_ok = last_finite_t >= (0.90 * float(x_hi))

        if saved_display_ok:
            pre_s = max(float(-np.min(t)), float(SINGLE_PTA_PRE_SEC))
            n_trials = int(sec.get("n_trials", 0))
            title = "First-pulse response"
        else:
            t, y, s, second_times, pre_s = rebuild_single_pta_display(summary, stim_names, x_hi)
            n_trials = int(len(stim_names))
            title = "First-pulse response"
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
        t, y, second_t, pre_s = extract_first_pta_display_segment(summary, ONLY_TRIAL, x_hi)
        s = None if t is None else np.full_like(y, np.nan)
        second_times = [second_t]
        title = "First-pulse response"

    if t is None or y is None or len(t) == 0 or y.shape != t.shape:
        if ONLY_TRIAL is None:
            sec = summary["summary"]["single_pta"]
            display = sec.get("display", {})
            t = np.asarray(display.get("t_rel_s", sec.get("t_rel_s", [])), dtype=float)
            y = np.asarray(display.get("mean", sec.get("mean", [])), dtype=float)
            s = np.asarray(display.get("sd", sec.get("spread", [])), dtype=float)
            n_trials = int(sec.get("n_trials", 0))
            title = "First-pulse response"
            second_t = safe_float(display.get("second_pulse_rel_s_mean"))
            second_times = [second_t] if np.isfinite(second_t) else []
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
            title = "First-pulse response"
            second_times = [safe_float(match.get("second_pulse_rel_s"))]
        if len(t) == 0 or y.shape != t.shape:
            ax.axis("off")
            return

    y = gevi_display(y)
    if s is not None:
        s = gevi_display(s)
    deriv = first_derivative_curve(t, y) if mode in {"derivative", "both"} else None
    if mode == "derivative":
        if deriv is None:
            ax.axis("off")
            return
        td, yd = deriv
        ax.plot(td, yd, color="tab:green", lw=2.0, label="derivative")
        ax.axhline(0, color="0.6", lw=0.8)
    else:
        if ONLY_TRIAL is None:
            ax.plot(t, y, color="black", lw=2.0, label="mean")
        else:
            ax.plot(t, y, color="black", lw=2.0)
        if s is not None and s.shape == t.shape:
            fill_label = "SD" if ONLY_TRIAL is None else None
            ax.fill_between(t, y - s, y + s, color="black", alpha=0.2, label=fill_label)
        if mode == "both" and deriv is not None:
            td, yd = deriv
            ax2 = ax.twinx()
            ax2.plot(td, yd, color="tab:green", lw=1.4)
            ax2.axhline(0, color="tab:green", lw=0.7, alpha=0.5)
            ax2.set_ylabel(GEVI_DERIV_YLABEL, color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.2, label="first pulse")
    if second_times:
        second_t = float(np.nanmedian(second_times))
        if np.isfinite(second_t) and np.min(t) <= second_t <= np.max(t):
            ax.axvline(second_t, color="tab:orange", ls="--", lw=1.2, label="second pulse")
    if not np.isfinite(pre_s) or pre_s <= 0:
        pre_s = float(SINGLE_PTA_PRE_SEC)
    ax.set_xlim(-float(pre_s), x_hi)
    ax.set_title("First-pulse derivative" if mode == "derivative" else title)
    ax.set_xlabel("time from first pulse (s)")
    ax.set_ylabel(GEVI_DERIV_YLABEL if mode == "derivative" else GEVI_YLABEL)
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
        x = gevi_display(x)
        s = gevi_display(s)
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
            ax.set_title("GEVI signal")
            ax.set_xlabel("time from stim onset (s)")
        else:
            ax.set_title("Baseline GEVI signal")
            ax.set_xlabel("time from trial start (s)")
        ax.set_ylabel(GEVI_YLABEL)
    else:
        trial_name, tr = choose_train_trial(summary)
        if tr is None:
            td = summary["trials"].get("processed_notched", {}).get(ONLY_TRIAL, {})
            t = np.asarray(td.get("t", []), dtype=float)
            x = np.asarray(td.get("F_notched", []), dtype=float)
            if len(t) == 0 or x.shape != t.shape:
                ax.axis("off")
                return
            x = gevi_display(x)
            ax.plot(t, x, color="tab:blue", lw=0.8)
            ax.set_title("GEVI signal")
            ax.set_xlabel("time from trial start (s)")
            ax.set_ylabel(GEVI_YLABEL)
            return
        t = np.asarray(tr.get("t_full_s", []), dtype=float)
        x = np.asarray(tr.get("signal_full", []), dtype=float)
        pulse_times = np.asarray(tr.get("pulse_times_s", []), dtype=float)
        if len(t) == 0 or x.shape != t.shape:
            ax.axis("off")
            return
        x = gevi_display(x)
        ax.plot(t, x, color="tab:blue", lw=0.8)
        for tp in pulse_times:
            ax.axvline(float(tp), color="tab:red", alpha=0.12, lw=0.7)
        if len(pulse_times):
            ax.axvline(0.0, color="tab:red", ls="--", lw=1.1, label="first pulse")
            stim_off = float(pulse_times[-1])
            ax.axvline(stim_off, color="tab:orange", ls="--", lw=1.1, label="last pulse")
        ax.set_title("GEVI signal")
        ax.set_xlabel("time from stim onset (s)")
        ax.set_ylabel(GEVI_YLABEL)
        if len(pulse_times):
            ax.legend(loc="best", fontsize=8)


def plot_lfp(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary.get("summary", {}).get("lfp", {})
        t_key = "t_stim_s_display" if USE_DECIMATED_LFP else "t_stim_s_full"
        mean_key = "mean_display" if USE_DECIMATED_LFP else "mean_full"
        sd_key = "sd_display" if USE_DECIMATED_LFP else "sd_full"
        t = np.asarray(sec.get(t_key, sec.get("t_stim_s_full", [])), dtype=float)
        y = np.asarray(sec.get(mean_key, sec.get("mean_full", [])), dtype=float)
        s = np.asarray(sec.get(sd_key, sec.get("sd_full", [])), dtype=float)
        if len(t) == 0 or y.shape != t.shape:
            ax.axis("off")
            return
        ax.plot(t, y, color="tab:green", lw=1.4)
        if s.shape == t.shape:
            ax.fill_between(t, y - s, y + s, color="tab:green", alpha=0.18)
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
            ax.axvline(float(np.nanmedian(last_pulses)), color="tab:orange", ls="--", lw=1.0, alpha=0.9)
        ax.set_title("LFP signal")
    else:
        td_e = summary.get("trials", {}).get("ephys", {}).get(ONLY_TRIAL, {})
        t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
        y = np.asarray(td_e.get("channels", {}).get("LFP", []), dtype=float)
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        if len(t) == 0 or y.shape != t.shape:
            ax.axis("off")
            return
        if USE_DECIMATED_LFP:
            t, [y] = decimate_curve_bundle(t, [y], max_points=LFP_DISPLAY_MAX_POINTS)
        ax.plot(t, y, color="tab:green", lw=0.9)
        if len(pulse_times):
            ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
            ax.axvline(float(pulse_times[-1]), color="tab:orange", ls="--", lw=1.0, alpha=0.9)
        ax.set_title("LFP signal")
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("LFP (a.u.)")


def plot_stim_trace(ax, summary: dict):
    ephys_trials = stim_trace_ephys_trials(summary)
    if ONLY_TRIAL is None:
        trial_name = None
        for name in [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]:
            td_e = ephys_trials.get(name, {})
            t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
            y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
            if len(t) >= 2 and y.shape == t.shape:
                trial_name = name
                break
        if trial_name is None:
            ax.axis("off")
            return
    else:
        trial_name = ONLY_TRIAL

    td_e = ephys_trials.get(trial_name, {})
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return
    ax.plot(t, y, color="tab:purple", lw=0.9)
    if len(pulse_times):
        ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
        ax.axvline(float(pulse_times[-1]), color="tab:orange", ls="--", lw=1.0, alpha=0.9)
    ax.set_title("DBS pulse train")
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("stim (V)")


def plot_velocity(ax, summary: dict):
    ephys_trials = stim_trace_ephys_trials(summary)
    if ONLY_TRIAL is None:
        rows = []
        times = []
        for name in stim_trial_names(summary):
            td_e = ephys_trials.get(name, {})
            t = np.asarray(td_e.get("vel_bin_t_s", []), dtype=float)
            y = np.asarray(td_e.get("vel_bin_cmps", []), dtype=float)
            good = np.isfinite(t) & np.isfinite(y)
            if np.sum(good) > 1:
                times.append(t[good])
                rows.append(y[good])
        t_common = build_common_axis_1d(times)
        if t_common is None or not rows:
            ax.axis("off")
            return
        Y = np.vstack([interpolate_curve(t_common, t, y) for t, y in zip(times, rows)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y_mean = np.nanmean(Y, axis=0)
            y_sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.full_like(y_mean, np.nan)
        ax.plot(t_common, y_mean, color="tab:brown", lw=1.5)
        if y_sd.shape == t_common.shape:
            ax.fill_between(t_common, y_mean - y_sd, y_mean + y_sd, color="tab:brown", alpha=0.18)
        t_plot = t_common
    else:
        td_e = ephys_trials.get(ONLY_TRIAL, {})
        t_plot = np.asarray(td_e.get("vel_bin_t_s", []), dtype=float)
        y = np.asarray(td_e.get("vel_bin_cmps", []), dtype=float)
        if len(t_plot) == 0 or y.shape != t_plot.shape:
            ax.axis("off")
            return
        ax.plot(t_plot, y, color="tab:brown", lw=1.0)

    ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
    stim_end = stim_end_time_s(summary, ONLY_TRIAL)
    if np.isfinite(stim_end) and float(np.nanmin(t_plot)) <= stim_end <= float(np.nanmax(t_plot)):
        ax.axvline(stim_end, color="tab:orange", ls="--", lw=1.0, alpha=0.9)
    ax.set_title("Wheel velocity")
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("velocity (cm/s)")


def plot_camera_frames(ax, summary: dict):
    ephys_trials = stim_trace_ephys_trials(summary)
    names = [ONLY_TRIAL] if ONLY_TRIAL is not None else stim_trial_names(summary)
    frame_rows = []
    row_labels = []
    for name in names:
        if name is None:
            continue
        t = np.asarray(ephys_trials.get(name, {}).get("cam_frame_times_stim_s", []), dtype=float)
        t = t[np.isfinite(t)]
        if len(t) <= 1:
            continue
        stride = max(1, int(CAMERA_FRAME_DISPLAY_STRIDE))
        frame_rows.append(t[::stride])
        row_labels.append(str(name))

    if not frame_rows:
        ax.axis("off")
        return

    offsets = np.arange(1, len(frame_rows) + 1)
    ax.eventplot(
        frame_rows,
        orientation="horizontal",
        lineoffsets=offsets,
        linelengths=0.75,
        colors="black",
        linewidths=0.5,
    )
    all_t = np.concatenate(frame_rows)
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
    stim_end = stim_end_time_s(summary, ONLY_TRIAL)
    if np.isfinite(stim_end) and float(np.nanmin(all_t)) <= stim_end <= float(np.nanmax(all_t)):
        ax.axvline(stim_end, color="tab:orange", ls="--", lw=1.0, alpha=0.9)
    ax.set_ylim(0.4, len(frame_rows) + 0.6)
    if len(frame_rows) <= 12:
        ax.set_yticks(offsets)
        ax.set_yticklabels(row_labels, fontsize=7)
    else:
        ax.set_ylabel("trial")
    ax.set_title("Camera frame timing")
    ax.set_xlabel("time from stim onset (s)")
    if len(frame_rows) <= 12:
        ax.set_ylabel("trial")


def plot_signal_hilbert(ax, summary: dict, harmonic: int = 1):
    source = signal_hilbert_section_key(harmonic)
    label = signal_hilbert_label(harmonic)
    if ONLY_TRIAL is None:
        sec = summary.get("summary", {}).get("train_pta", {}).get(source, {})
        t = np.asarray(sec.get("time_s_display", sec.get("time_s_full", [])), dtype=float)
        if HILBERT_VIEW == "relative":
            y = np.asarray(sec.get("relative_mean_display", sec.get("relative_mean_full", [])), dtype=float)
            s = np.asarray(sec.get("relative_sd_display", sec.get("relative_sd_full", [])), dtype=float)
        else:
            y = np.asarray(sec.get("amplitude_mean_display", sec.get("amplitude_mean_full", [])), dtype=float)
            s = np.asarray(sec.get("amplitude_sd_display", sec.get("amplitude_sd_full", [])), dtype=float)
        f_center = safe_float(sec.get("f_center_hz_mean"))
        title = hilbert_panel_title(harmonic)
    else:
        sec = summary.get("trials", {}).get("train_pta", {}).get(ONLY_TRIAL, {}).get(source, {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        y = np.asarray(sec.get("amplitude", []), dtype=float)
        if HILBERT_VIEW == "relative":
            baseline_start_s, baseline_end_s, baseline_stat = train_hilbert_baseline_meta(summary, source)
            if not np.isfinite(baseline_start_s):
                baseline_start_s = float(t[0]) if len(t) else np.nan
            y = relative_curve(t, y, baseline_start_s, baseline_end_s, baseline_stat)
        s = np.array([], dtype=float)
        f_center = safe_float(sec.get("f_center_hz"))
        title = hilbert_panel_title(harmonic)
    if len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return
    ax.plot(t, y, color="tab:blue", lw=1.5 if ONLY_TRIAL is None else 1.0)
    if s.shape == t.shape:
        ax.fill_between(t, y - s, y + s, color="tab:blue", alpha=0.18)
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
    stim_end = stim_end_time_s(summary, ONLY_TRIAL)
    if np.isfinite(stim_end) and float(np.nanmin(t)) <= stim_end <= float(np.nanmax(t)):
        ax.axvline(stim_end, color="tab:orange", ls="--", lw=1.0, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("amp / baseline" if HILBERT_VIEW == "relative" else "amplitude")


def plot_lfp_hilbert(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary.get("summary", {}).get("train_pta", {}).get("lfp_hilbert", {})
        t = np.asarray(sec.get("time_s_display", sec.get("time_s_full", [])), dtype=float)
        if HILBERT_VIEW == "relative":
            y = np.asarray(sec.get("relative_mean_display", sec.get("relative_mean_full", [])), dtype=float)
            s = np.asarray(sec.get("relative_sd_display", sec.get("relative_sd_full", [])), dtype=float)
        else:
            y = np.asarray(sec.get("amplitude_mean_display", sec.get("amplitude_mean_full", [])), dtype=float)
            s = np.asarray(sec.get("amplitude_sd_display", sec.get("amplitude_sd_full", [])), dtype=float)
        f_center = safe_float(sec.get("f_center_hz_mean"))
        title = "LFP DBS-frequency amplitude"
    else:
        sec = summary.get("trials", {}).get("train_pta", {}).get(ONLY_TRIAL, {}).get("lfp_hilbert", {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        y = np.asarray(sec.get("amplitude", []), dtype=float)
        if HILBERT_VIEW == "relative":
            baseline_start_s, baseline_end_s, baseline_stat = train_hilbert_baseline_meta(summary, "lfp_hilbert")
            if not np.isfinite(baseline_start_s):
                baseline_start_s = float(t[0]) if len(t) else np.nan
            y = relative_curve(t, y, baseline_start_s, baseline_end_s, baseline_stat)
        s = np.array([], dtype=float)
        f_center = safe_float(sec.get("f_center_hz"))
        title = "LFP DBS-frequency amplitude"
    if len(t) == 0 or y.shape != t.shape:
        ax.axis("off")
        return
    ax.plot(t, y, color="tab:green", lw=1.5 if ONLY_TRIAL is None else 1.0)
    if s.shape == t.shape:
        ax.fill_between(t, y - s, y + s, color="tab:green", alpha=0.18)
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.0, alpha=0.9)
    stim_end = stim_end_time_s(summary, ONLY_TRIAL)
    if np.isfinite(stim_end) and float(np.nanmin(t)) <= stim_end <= float(np.nanmax(t)):
        ax.axvline(stim_end, color="tab:orange", ls="--", lw=1.0, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("amp / baseline" if HILBERT_VIEW == "relative" else "amplitude")


def plot_pulse_windows(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary["summary"]["train_pta"]
        display = sec.get("display", {})
        t_rel = np.asarray(display.get("t_rel_s", []), dtype=float)
        mean_curve = np.asarray(display.get("mean", []), dtype=float)
        spread = np.asarray(display.get("sd", []), dtype=float)
        second_t = safe_float(display.get("second_pulse_rel_s_mean"))
        second_times = [second_t] if np.isfinite(second_t) else []
        if len(t_rel) == 0 or mean_curve.shape != t_rel.shape:
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
            t_rel = np.asarray(sec.get("t_rel_s", []), dtype=float)
            mean_curve = np.asarray(sec.get("mean_across_trials", []), dtype=float)
            spread = np.asarray(sec.get("sd_across_trials", []), dtype=float)
        if len(t_rel) == 0 or mean_curve.shape != t_rel.shape:
            ax.axis("off")
            return
        mean_curve = gevi_display(mean_curve)
        spread = gevi_display(spread)
        ax.plot(t_rel, mean_curve, color="black", lw=2.0, label="within-trial mean")
        if spread.shape == t_rel.shape:
            ax.fill_between(t_rel, mean_curve - spread, mean_curve + spread, color="black", alpha=0.2, label="SD")
        if second_times:
            second_t = float(np.nanmedian(second_times))
            if np.isfinite(second_t) and t_rel[0] <= second_t <= t_rel[-1]:
                ax.axvline(second_t, color="tab:orange", ls="--", lw=1.1, label="second pulse")
        title = "Pulse-triggered response"
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
        segs = gevi_display(segs)
        mean_curve = gevi_display(mean_curve)
        spread = gevi_display(spread)
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
        title = "Pulse-triggered response"
    ax.axvline(0.0, color="tab:red", ls="--", lw=1.1)
    ax.set_xlim(float(np.min(t_rel)), float(np.max(t_rel)))
    ax.set_title(title)
    ax.set_xlabel("time from pulse (s)")
    ax.set_ylabel(GEVI_YLABEL)
    ax.legend(loc="best", fontsize=8)


def plot_fft(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary["summary"]["train_pta"]["power_spectrum"]
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p = np.asarray(sec.get("psd_db_mean", []), dtype=float)
        s = np.asarray(sec.get("psd_db_sd", []), dtype=float)
        title = "Power spectrum"
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
        title = "Power spectrum"
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
    if has_block_stim_data(summary):
        sec = summary["summary"]["train_pta"]["spectrogram"]
        t = np.asarray(sec.get("time_s", []), dtype=float)
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p_linear = spectrogram_linear_from_section(sec, linear_key="power_linear_mean", db_key="power_db_mean")
        if SPECTROGRAM_VIEW == "relative":
            p_show = np.asarray(sec.get("relative_db_mean", []), dtype=float)
            if p_show.shape != (len(f), len(t)):
                baseline_start_s = safe_float(sec.get("baseline_start_s"))
                baseline_end_s = safe_float(sec.get("baseline_end_s"))
                if not np.isfinite(baseline_start_s):
                    baseline_start_s = float(t[0]) if len(t) else np.nan
                if not np.isfinite(baseline_end_s):
                    baseline_end_s = float(SPECTROGRAM_BASELINE_END_S)
                baseline_stat = str(sec.get("baseline_stat", SPECTROGRAM_BASELINE_STAT))
                p_show = relative_spectrogram_db(t, p_linear, baseline_start_s, baseline_end_s, baseline_stat)
        else:
            p_show = np.asarray(sec.get("power_db_mean", []), dtype=float)
            if p_show.shape != (len(f), len(t)):
                p_show = db_from_linear(p_linear)

        f_stim = safe_float(summary["summary"]["train_pta"]["metrics"].get("f_stim_hz_mean"))
        title = "GEVI spectrogram"
    else:
        sec = summary["summary"]["processed_notched"].get("baseline_spectrogram", {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p_linear = spectrogram_linear_from_section(sec, linear_key="power_linear", db_key="power_db")
        p_show = np.asarray(sec.get("power_db", []), dtype=float)
        if p_show.shape != (len(f), len(t)):
            p_show = db_from_linear(p_linear)
        title = "Baseline GEVI spectrogram"
        f_stim = np.nan

    if len(t) == 0 or len(f) == 0 or p_show.shape != (len(f), len(t)):
        ax.axis("off")
        return
    view = SPECTROGRAM_VIEW if has_block_stim_data(summary) else "absolute"
    p_show, cbar_label, cmap, clim, norm = transform_spectrogram_for_display(p_show, view, source="gevi")
    im = ax.imshow(
        p_show,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[float(t[0]), float(t[-1]), float(f[0]), float(f[-1])],
        interpolation=SPECTROGRAM_INTERPOLATION,
        norm=norm,
    )
    if norm is None and clim is not None:
        im.set_clim(*clim)
    if np.isfinite(f_stim) and np.min(f) <= f_stim <= np.max(f):
        ax.axhline(f_stim, color="tab:red", ls="--", lw=1.0, alpha=0.8)
    ax.set_ylim(0, min(float(SPEC_FMAX_HZ), float(np.max(f))))
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)


def plot_lfp_spectrogram(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary.get("summary", {}).get("train_pta", {}).get("lfp_spectrogram", {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p_linear = spectrogram_linear_from_section(sec, linear_key="power_linear_mean", db_key="power_db_mean")
        if SPECTROGRAM_VIEW == "relative":
            p_show = np.asarray(sec.get("relative_db_mean", []), dtype=float)
            if p_show.shape != (len(f), len(t)):
                baseline_start_s = safe_float(sec.get("baseline_start_s"))
                baseline_end_s = safe_float(sec.get("baseline_end_s"))
                if not np.isfinite(baseline_start_s):
                    baseline_start_s = float(t[0]) if len(t) else np.nan
                if not np.isfinite(baseline_end_s):
                    baseline_end_s = float(SPECTROGRAM_BASELINE_END_S)
                baseline_stat = str(sec.get("baseline_stat", SPECTROGRAM_BASELINE_STAT))
                p_show = relative_spectrogram_db(t, p_linear, baseline_start_s, baseline_end_s, baseline_stat)
        else:
            p_show = np.asarray(sec.get("power_db_mean", []), dtype=float)
            if p_show.shape != (len(f), len(t)):
                p_show = db_from_linear(p_linear)
        f_stim = safe_float(summary["summary"]["train_pta"]["metrics"].get("f_stim_hz_mean"))
        title = "LFP spectrogram"
    else:
        sec = summary.get("trials", {}).get("train_pta", {}).get(ONLY_TRIAL, {}).get("lfp_spectrogram", {})
        t = np.asarray(sec.get("time_s", []), dtype=float)
        f = np.asarray(sec.get("freq_hz", []), dtype=float)
        p_linear = spectrogram_linear_from_section(sec, linear_key="power_linear", db_key="power_db")
        if SPECTROGRAM_VIEW == "relative":
            baseline_start_s = float(t[0]) if len(t) else np.nan
            baseline_end_s = float(SPECTROGRAM_BASELINE_END_S)
            p_show = relative_spectrogram_db(t, p_linear, baseline_start_s, baseline_end_s, SPECTROGRAM_BASELINE_STAT)
        else:
            p_show = np.asarray(sec.get("power_db", []), dtype=float)
            if p_show.shape != (len(f), len(t)):
                p_show = db_from_linear(p_linear)
        f_stim = safe_float(summary.get("trials", {}).get("train_pta", {}).get(ONLY_TRIAL, {}).get("f_stim_hz"))
        title = "LFP spectrogram"

    if len(t) == 0 or len(f) == 0 or p_show.shape != (len(f), len(t)):
        ax.axis("off")
        return
    p_show, cbar_label, cmap, clim, norm = transform_spectrogram_for_display(p_show, SPECTROGRAM_VIEW, source="lfp")
    im = ax.imshow(
        p_show,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[float(t[0]), float(t[-1]), float(f[0]), float(f[-1])],
        interpolation=SPECTROGRAM_INTERPOLATION,
        norm=norm,
    )
    if norm is None and clim is not None:
        im.set_clim(*clim)
    if np.isfinite(f_stim) and np.min(f) <= f_stim <= np.max(f):
        ax.axhline(f_stim, color="tab:red", ls="--", lw=1.0, alpha=0.8)
    ax.set_ylim(0, min(float(SPEC_FMAX_HZ), float(np.max(f))))
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)


def plot_plv_histogram_common(ax, summary: dict, section_key: str, label: str):
    if ONLY_TRIAL is None:
        phases = block_plv_phases(summary, section_key=section_key)
        center = safe_float(summary.get("summary", {}).get("train_pta", {}).get(section_key, {}).get("f_center_hz_mean"))
    else:
        trial_name, tr = choose_train_trial(summary)
        if tr is None:
            ax.axis("off")
            return
        phases = trial_plv_phases(tr, section_key=section_key)
        center = safe_float(tr.get(section_key, {}).get("f_center_hz"))

    if len(phases) == 0:
        ax.axis("off")
        return

    harmonic = 1
    m = re.search(r"_h(\d+)$", section_key)
    if m:
        harmonic = int(m.group(1))
    title = plv_panel_title(harmonic)
    plv, pref = phases_to_plv(phases)
    counts, edges = np.histogram(phases, bins=np.linspace(-np.pi, np.pi, 37))
    rmax = max(1.0, float(np.max(counts)))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="tab:blue", alpha=0.45, edgecolor="white", linewidth=0.4)
    ax.annotate("", xy=(pref, plv * rmax), xytext=(0, 0), arrowprops=dict(color="crimson", lw=2.5, arrowstyle="->"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"PLV = {plv:.2f}", fontsize=11, pad=2)


def plot_plv_histogram(ax, summary: dict):
    plot_plv_histogram_common(ax, summary, section_key="plv", label="PLV")


def plot_plv_harmonic_histogram(ax, summary: dict, harmonic: int):
    plot_plv_histogram_common(
        ax,
        summary,
        section_key=plv_section_key(harmonic),
        label=plv_label(harmonic),
    )


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
    ax.set_title("Band-limited power")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(ylabel)


def plot_pulsogram(ax, summary: dict):
    if ONLY_TRIAL is None:
        sec = summary.get("summary", {}).get("pulsogram", {}).get("heatmap", {})
        pulse_numbers = np.asarray(sec.get("pulse_numbers", []), dtype=int)
        t_rel = np.asarray(sec.get("t_rel_s", []), dtype=float)
        M = np.asarray(sec.get("mean", []), dtype=float)
        if len(pulse_numbers) == 0 or len(t_rel) == 0 or M.shape != (len(pulse_numbers), len(t_rel)):
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
    M_show = gevi_display(M[:, m])
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
    ax.set_title("Pulse-by-pulse response")
    ax.set_xlabel("time from pulse (ms)")
    ax.set_ylabel("pulse number")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(GEVI_YLABEL)
    ax._pulsogram_im = im
    ax._pulsogram_max_abs = max_abs


def build_panels(summary: dict):
    panels = []
    stim_available = has_block_stim_data(summary) if ONLY_TRIAL is None else has_trial_stim_data(summary, ONLY_TRIAL)
    stim_trace_available = has_block_stim_trace_data(summary) if ONLY_TRIAL is None else has_trial_stim_trace_data(summary, ONLY_TRIAL)
    lfp_available = has_block_lfp_data(summary) if ONLY_TRIAL is None else has_trial_lfp_data(summary, ONLY_TRIAL)
    velocity_available = has_block_velocity_data(summary) if ONLY_TRIAL is None else has_trial_velocity_data(summary, ONLY_TRIAL)
    camera_frames_available = has_block_camera_frame_data(summary) if ONLY_TRIAL is None else has_trial_camera_frame_data(summary, ONLY_TRIAL)
    lfp_hilbert_available = has_block_hilbert_data(summary, "lfp_hilbert") if ONLY_TRIAL is None else has_trial_hilbert_data(summary, ONLY_TRIAL, "lfp_hilbert")
    baseline_spec_available = has_block_baseline_spectrogram(summary)
    averaged_spec_available = has_block_stim_data(summary) or baseline_spec_available
    if PLOT_FULL_TRACE:
        panels.append(("full_trace", plot_full_trace))
    if PLOT_STIM_TRACE and stim_trace_available:
        panels.append(("stim_trace", plot_stim_trace))
    if PLOT_LFP and lfp_available:
        panels.append(("lfp", plot_lfp))
    if PLOT_VELOCITY and velocity_available:
        panels.append(("velocity", plot_velocity))
    if PLOT_CAMERA_FRAMES and camera_frames_available:
        panels.append(("camera_frames", plot_camera_frames))
    if single_pta_mode() != "off" and stim_available:
        panels.append(("single_pta", plot_single_pta))
    if PLOT_PULSE_WINDOWS and stim_available:
        panels.append(("pulse_windows", plot_pulse_windows))
    if PLOT_FFT and stim_available:
        panels.append(("fft", plot_fft))
    if PLOT_SPECTROGRAM and averaged_spec_available:
        panels.append(("spectrogram", plot_spectrogram))
    if PLOT_LFP_SPECTROGRAM and stim_available:
        panels.append(("lfp_spectrogram", plot_lfp_spectrogram))
    for harmonic in parse_harmonic_selection(PLOT_PLV_HISTOGRAMS):
        section_key = plv_section_key(harmonic)
        available = has_block_plv_data(summary, section_key) if ONLY_TRIAL is None else has_trial_plv_data(summary, ONLY_TRIAL, section_key)
        if available:
            panels.append((
                f"plv_h{harmonic}_histogram",
                lambda ax, summary, harmonic=harmonic: plot_plv_harmonic_histogram(ax, summary, harmonic),
            ))
    if PLOT_SIGNAL_HILBERT:
        for harmonic in parse_harmonic_selection(PLOT_SIGNAL_HILBERT_HARMONICS):
            source = signal_hilbert_section_key(harmonic)
            available = has_block_hilbert_data(summary, source) if ONLY_TRIAL is None else has_trial_hilbert_data(summary, ONLY_TRIAL, source)
            if available:
                panels.append((source, lambda ax, summary, harmonic=harmonic: plot_signal_hilbert(ax, summary, harmonic)))
    if PLOT_LFP_HILBERT and lfp_hilbert_available:
        panels.append(("lfp_hilbert", plot_lfp_hilbert))
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    title = figure_title(summary)
    for i, (ax, (name, fn)) in enumerate(zip(axes, panels)):
        if name.startswith("plv_h") and name.endswith("_histogram"):
            spec = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(spec, projection="polar")
            axes[i] = ax
        fn(ax, summary)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.965], h_pad=2.8, w_pad=1.2)
    fig.subplots_adjust(hspace=0.55)

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
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / f"{slugify_output_name(args.mouse, args.date, args.block)}_plot.png"
    plot_summary(summary, save_path=out_path)


if __name__ == "__main__":
    main()
