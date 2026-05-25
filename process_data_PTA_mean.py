from pathlib import Path
import pickle
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt, spectrogram
from scipy.stats import mannwhitneyu
from config import DATA_ANALYSIS_ROOT

MOUSE_NAME = "Vinnie1"  # change to "Jamie5", etc.
SINGLE_DATE = "15-05-26"
SINGLE_BLOCK = "R13"


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
PLV_HARMONICS = "1+2+3+4+5+6"  # 1=stim frequency, 2=2x stim, 3=3x stim, etc.
SHOW_PLV_HISTOGRAMS = "1+2+3"       # e.g. "1", "2", "second", or "1+2+3"
SIGNAL_HILBERT_HARMONICS = "1+2+3+4+5+6"  # GEVI Hilbert amplitude bands to save

# -------------------------
# SETTINGS
# -------------------------
SIGNAL_MODE = "notched"  # "notched_or_bleach_or_raw", "notched", "bleach", "raw"
PRE_SEC = 0.010          # Plot zoom only
POST_SEC = 0.030         # Plot zoom only
EXTRACT_PRE_SEC = 0.010  # Used only when PULSE_WINDOW_MODE == "fixed"
EXTRACT_POST_SEC = 0.030 # Used only when PULSE_WINDOW_MODE == "fixed"
PULSE_WINDOW_MODE = "freq_aware"  # "fixed" or "freq_aware"
PULSE_WINDOW_SCALE = 1.0     # In freq_aware mode, use this many post-pulse periods on the right; left side is baseline only
PERIOD_FRACTION = 0.95       # Use this fraction of each pulse period to avoid bleed into the next pulse
BASELINE_MODE = "local_pre"  # "none", "local_pre", "global_pre_stim"
GLOBAL_BASELINE_STAT = "median"  # "median" or "mean"
GLOBAL_BASELINE_PRE_SEC = 0.5     # Used only when BASELINE_MODE == "global_pre_stim"
MIN_PULSES = 5
N_TO_PLOT = 3
NFFT_SPEC = 512
SPECTROGRAM_TARGET_WINDOW_SEC = 0.512
SPECTROGRAM_OVERLAP_FRAC = 0.95
SPECTROGRAM_WINDOW = "hann"
SHOW_PULSE_WINDOWS = True
SPREAD_MODE = "sd"  # "sem" or "sd"
SAVE_SPECTROGRAM = True
SAVE_LFP_SPECTROGRAM = True
SPECTROGRAM_CMAP = "RdBu_r"
SPECTROGRAM_MODE = "relative"  # "absolute" or "relative"
SPECTROGRAM_SCALE = "db"       # "db" or "linear"
SPECTROGRAM_BASELINE_END_S = -0.5
SPECTROGRAM_BASELINE_STAT = "median"  # "median" or "mean"
SPECTROGRAM_BASELINE_PRE_SEC = 5.0
SPECTROGRAM_DISPLAY_PERCENTILES = (5.0, 99.0)
SPECTROGRAM_INTERPOLATION = "bilinear"
SAVE_STIM_BAND_HILBERT = True
HILBERT_HALF_BAND_HZ = 2.0
HILBERT_BASELINE_END_S = -0.5
HILBERT_BASELINE_STAT = "median"  # "median" or "mean"
HILBERT_DISPLAY_MAX_POINTS = 20000
HILBERT_STATS_BASELINE_PRE_SEC = 5.0
HILBERT_STATS_BIN_SEC = 0.5
HILBERT_STATS_MIN_BASELINE_BINS = 4
HILBERT_STATS_MIN_STIM_BINS = 4
LATENCY_THRESHOLD_SD = 1.0
LATENCY_MAX_PEAKS = 3
LATENCY_TOP_SAMPLES = 2
ONLY_TRIAL = None  # e.g. "R1_1"


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
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(parse_harmonic_selection(item))
    else:
        text = str(value).strip().lower()
        if not text or text in {"false", "none", "off", "no"}:
            return []
        text = text.replace("harmonic", "").replace("h", "")
        parts = re.split(r"[+,\s;/]+", text)
        out = []
        for part in parts:
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


def trial_sort_key(name: str):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


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


def nansem(y: np.ndarray, axis=0):
    n = np.sum(np.isfinite(y), axis=axis)
    sd = np.nanstd(y, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = sd / np.sqrt(n)
    return se


def nansd(y: np.ndarray, axis=0):
    return np.nanstd(y, axis=axis, ddof=1)


def estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return np.nan
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return 1.0 / dt


def safe_float(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def decimate_curve_bundle(x: np.ndarray, arrays: list[np.ndarray], max_points: int = HILBERT_DISPLAY_MAX_POINTS):
    x = np.asarray(x, dtype=float)
    clean_arrays = [np.asarray(arr, dtype=float) for arr in arrays]
    if len(x) == 0:
        return np.array([], dtype=float), [np.array([], dtype=float) for _ in clean_arrays], 1

    stride = max(1, int(np.ceil(len(x) / max(2, int(max_points)))))
    if stride == 1:
        return x.copy(), [arr.copy() if arr.shape == x.shape else np.array([], dtype=float) for arr in clean_arrays], 1

    idx = np.arange(0, len(x), stride, dtype=int)
    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return x[idx], [arr[idx] if arr.shape == x.shape else np.array([], dtype=float) for arr in clean_arrays], int(stride)


def estimate_stim_freq_hz(pulse_times: np.ndarray) -> float:
    if len(pulse_times) < 2:
        return np.nan
    isi = np.diff(pulse_times)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    if len(isi) == 0:
        return np.nan
    return 1.0 / float(np.median(isi))


def estimate_ipi_s(pulse_times: np.ndarray) -> float:
    if len(pulse_times) < 2:
        return np.nan
    isi = np.diff(pulse_times)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    if len(isi) == 0:
        return np.nan
    return float(np.median(isi))


def resolve_pulse_window_s(pulse_times: np.ndarray) -> tuple[float, float]:
    if PULSE_WINDOW_MODE != "freq_aware":
        return float(EXTRACT_PRE_SEC), float(EXTRACT_POST_SEC)

    ipi = estimate_ipi_s(pulse_times)
    if not np.isfinite(ipi) or ipi <= 0:
        return float(EXTRACT_PRE_SEC), float(EXTRACT_POST_SEC)

    frac = min(1.0, max(0.0, float(PERIOD_FRACTION)))
    win_s = frac * ipi
    return float(win_s), float(win_s)


def build_freq_aware_folded_segment(
    t: np.ndarray,
    x: np.ndarray,
    pulse_times: np.ndarray,
    pulse_idx: int,
    rel_grid: np.ndarray,
    global_baseline: float | None = None,
):
    if len(rel_grid) < 4:
        return None

    n_periods = max(1, int(round(float(PULSE_WINDOW_SCALE))))
    if pulse_idx < 1 or (pulse_idx + n_periods) >= len(pulse_times):
        return None

    pre_mask = rel_grid < 0
    post_mask = rel_grid >= 0
    if not np.any(pre_mask) or not np.any(post_mask):
        return None

    pre_grid = rel_grid[pre_mask]
    post_grid = rel_grid[post_mask]
    pre_s = float(-pre_grid[0])
    post_s = float(post_grid[-1])
    if pre_s <= 0 or post_s <= 0:
        return None

    pre_alpha = (pre_grid + pre_s) / pre_s
    post_alpha = post_grid / post_s

    post_rows = []
    frac = min(1.0, max(0.0, float(PERIOD_FRACTION)))
    if frac <= 0:
        return None

    tp = float(pulse_times[pulse_idx])
    pre_start = tp - pre_s
    pre_end = tp
    if pre_start < t[0] or pre_end > t[-1]:
        return None

    y = np.empty(len(rel_grid), dtype=float)
    y[pre_mask] = np.interp(pre_start + pre_alpha * (pre_end - pre_start), t, x)

    if n_periods == 1:
        post_start = float(pulse_times[pulse_idx])
        post_end = float(pulse_times[pulse_idx + 1])
        if not (post_end > post_start):
            return None
        if post_start < t[0] or post_end > t[-1]:
            return None

        post_dur = post_end - post_start
        post_sample_end = post_start + frac * post_dur
        y[post_mask] = np.interp(post_start + post_alpha * (post_sample_end - post_start), t, x)
    else:
        for j in range(n_periods):
            post_start = float(pulse_times[pulse_idx + j])
            post_end = float(pulse_times[pulse_idx + j + 1])
            if not (post_end > post_start):
                return None
            if post_start < t[0] or post_end > t[-1]:
                return None

            post_dur = post_end - post_start
            post_sample_end = post_start + frac * post_dur
            post_rows.append(np.interp(post_start + post_alpha * (post_sample_end - post_start), t, x))

        y[post_mask] = np.mean(np.vstack(post_rows), axis=0)
    return baseline_correct_segment(y, rel_grid, BASELINE_MODE, global_baseline=global_baseline)


def baseline_correct_segment(seg: np.ndarray, rel_grid: np.ndarray, mode: str, global_baseline: float | None = None) -> np.ndarray:
    if mode == "none":
        return seg
    if mode == "global_pre_stim" and global_baseline is not None and np.isfinite(global_baseline):
        return seg - float(global_baseline)

    pre = seg[rel_grid < 0]
    if len(pre) == 0:
        return seg
    b = float(np.median(pre))
    return seg - b


def compute_global_pre_stim_baseline(t: np.ndarray, x: np.ndarray, pulse_times: np.ndarray) -> float | None:
    if len(pulse_times) == 0:
        return None
    t_first = float(pulse_times[0])
    pre = x[(t < t_first) & (t >= t_first - float(GLOBAL_BASELINE_PRE_SEC))]
    if len(pre) == 0:
        return None
    if GLOBAL_BASELINE_STAT == "mean":
        return float(np.mean(pre))
    return float(np.median(pre))


def build_rel_grid(t: np.ndarray, pre_s: float, post_s: float) -> np.ndarray | None:
    fs = estimate_fs(t)
    if not np.isfinite(fs):
        return None
    dt = 1.0 / fs
    n = int(np.floor((pre_s + post_s) / dt)) + 1
    if n < 4:
        return None
    return -pre_s + np.arange(n) * dt


def extract_all_pulse_segments(
    t: np.ndarray,
    x: np.ndarray,
    pulse_times: np.ndarray,
    pre_s: float,
    post_s: float,
    rel_grid: np.ndarray,
    global_baseline: float | None = None,
):
    segments = []
    used_pulse_idx = []
    for k, tp in enumerate(pulse_times):
        if PULSE_WINDOW_MODE == "freq_aware":
            y = build_freq_aware_folded_segment(
                t,
                x,
                pulse_times,
                k,
                rel_grid,
                global_baseline=global_baseline,
            )
            if y is None:
                continue
            segments.append(y)
            used_pulse_idx.append(int(k))
            continue

        t0 = float(tp - pre_s)
        t1 = float(tp + post_s)
        if t0 < t[0] or t1 > t[-1]:
            continue

        target_t = tp + rel_grid
        y = np.interp(target_t, t, x)
        y = baseline_correct_segment(y, rel_grid, BASELINE_MODE, global_baseline=global_baseline)
        segments.append(y)
        used_pulse_idx.append(int(k))

    if not segments:
        return None, []

    return np.vstack(segments), used_pulse_idx


def get_hilbert_band_hz(fs: float, f_center_hz: float | None = None) -> tuple[float, float] | None:
    if not np.isfinite(fs) or fs <= 0:
        return None
    f_center = float(f_center_hz) if f_center_hz is not None and np.isfinite(f_center_hz) else 135.0
    nyq = 0.5 * fs
    low = max(0.5, f_center - float(HILBERT_HALF_BAND_HZ))
    high = min(nyq * 0.95, f_center + float(HILBERT_HALF_BAND_HZ))
    if not (0 < low < high < nyq):
        return None
    return low, high


def spectral_peak_metrics(t: np.ndarray, x: np.ndarray, f_stim: float):
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0:
        return {}
    if len(x) < 16:
        return {}

    f, pxx = periodogram(x, fs=fs, window="hann", detrend="constant", scaling="density")
    p_db = 10.0 * np.log10(np.maximum(pxx, 1e-30))

    out = {"freq_hz": f, "psd_db": p_db}
    if not np.isfinite(f_stim) or f_stim <= 0:
        return out

    for mul in [1, 2, 3]:
        ft = mul * f_stim
        if ft >= 0.5 * fs:
            continue
        idx = int(np.argmin(np.abs(f - ft)))
        p_target = float(p_db[idx])

        band = (f >= ft - 5.0) & (f <= ft + 5.0)
        excl = (f >= ft - 0.5) & (f <= ft + 0.5)
        bg = p_db[band & ~excl]
        bg_med = float(np.median(bg)) if len(bg) else np.nan
        snr_like = p_target - bg_med if np.isfinite(bg_med) else np.nan

        out[f"f{mul}_target_hz"] = float(ft)
        out[f"f{mul}_power_db"] = p_target
        out[f"f{mul}_snr_like_db"] = float(snr_like) if np.isfinite(snr_like) else np.nan
    return out


def resolve_spectrogram_nperseg(fs: float, n_samples: int, window_sec: float | None = None) -> int:
    if window_sec is None or not np.isfinite(window_sec) or float(window_sec) <= 0:
        nperseg = max(int(NFFT_SPEC), int(round(float(fs) * float(SPECTROGRAM_TARGET_WINDOW_SEC))))
    else:
        nperseg = int(round(float(window_sec) * float(fs)))
    return max(32, min(int(nperseg), int(n_samples)))


def compute_spectrogram_data(t: np.ndarray, x: np.ndarray):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0 or len(x) < 32 or t.shape != x.shape:
        return {}

    nperseg = resolve_spectrogram_nperseg(fs, len(x))
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
    psg_db = 10.0 * np.log10(np.maximum(psg, 1e-30))
    t0 = float(t[0]) if len(t) else 0.0
    return {
        "freq_hz": np.asarray(fsg, dtype=np.float64),
        "time_s": np.asarray(tsg + t0, dtype=np.float64),
        "power_linear": np.asarray(psg, dtype=np.float64),
        "power_db": np.asarray(psg_db, dtype=np.float64),
        "window": SPECTROGRAM_WINDOW,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "window_sec": float(nperseg / fs) if np.isfinite(fs) and fs > 0 else np.nan,
    }


def compute_spectrogram_data_matched(
    t: np.ndarray,
    x: np.ndarray,
    window_sec: float | None = None,
    time_ref: np.ndarray | None = None,
    freq_ref: np.ndarray | None = None,
):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0 or len(x) < 32 or t.shape != x.shape:
        return {}

    nperseg = resolve_spectrogram_nperseg(fs, len(x), window_sec=window_sec)

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
    t_out = np.asarray(tsg + float(t[0]), dtype=np.float64)
    f_out = np.asarray(fsg, dtype=np.float64)
    p_out = np.asarray(psg, dtype=np.float64)

    if time_ref is not None and freq_ref is not None:
        time_ref = np.asarray(time_ref, dtype=float)
        freq_ref = np.asarray(freq_ref, dtype=float)
        if len(time_ref) >= 2 and len(freq_ref) >= 2 and p_out.shape == (len(f_out), len(t_out)):
            FF, TT = np.meshgrid(freq_ref, time_ref, indexing="ij")
            pts = np.column_stack([FF.ravel(), TT.ravel()])
            interp = RegularGridInterpolator((f_out, t_out), p_out, bounds_error=False, fill_value=np.nan)
            p_out = interp(pts).reshape(len(freq_ref), len(time_ref))
            t_out = np.asarray(time_ref, dtype=np.float64)
            f_out = np.asarray(freq_ref, dtype=np.float64)

    return {
        "freq_hz": np.asarray(f_out, dtype=np.float64),
        "time_s": np.asarray(t_out, dtype=np.float64),
        "power_linear": np.asarray(p_out, dtype=np.float64),
        "power_db": np.asarray(10.0 * np.log10(np.maximum(p_out, 1e-30)), dtype=np.float64),
        "window": SPECTROGRAM_WINDOW,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "window_sec": float(nperseg / fs) if np.isfinite(fs) and fs > 0 else np.nan,
    }


def choose_lfp(td_e: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    x = np.asarray(td_e.get("channels", {}).get("LFP", []), dtype=float)
    if len(t) == 0 or x.shape != t.shape:
        return None, None
    return t, x


def clip_trace_to_support(
    t: np.ndarray,
    x: np.ndarray,
    t_start: float | None,
    t_end: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(t) == 0 or x.shape != t.shape:
        return None, None
    if t_start is None or t_end is None or not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return t, x
    keep = np.isfinite(t) & (t >= float(t_start)) & (t <= float(t_end))
    if np.sum(keep) < 4:
        return None, None
    return t[keep], x[keep]

def compute_plv(t: np.ndarray, x: np.ndarray, pulse_times: np.ndarray, f_center_hz: float | None = None):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    pulse_times = np.asarray(pulse_times, dtype=float)
    
    if len(t) < 4 or x.shape != t.shape:
        return {}

    pulse_times = pulse_times[np.isfinite(pulse_times)]
    pulse_times = pulse_times[(pulse_times >= t[0]) & (pulse_times <= t[-1])]
    
    if len(pulse_times) < 2:
        return {}
    fs = estimate_fs(t)
    
    band = get_hilbert_band_hz(fs, f_center_hz=f_center_hz)
    if band is None:
        return {}
    
    nyq = 0.5 * fs
    low, high = band
    sos = butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
    xf = sosfiltfilt(sos, x)
    
    analytic = hilbert(xf)
    
    pulse_idx = np.searchsorted(t, pulse_times)
    pulse_idx = np.clip(pulse_idx, 1, len(t) - 1)
    
    left_idx = pulse_idx - 1
    right_idx = pulse_idx
    
    pulse_idx = np.where(
        np.abs(t[right_idx] - pulse_times) < np.abs(t[left_idx] - pulse_times),
        right_idx,
        left_idx,
    )
    
    phase_pulses = np.angle(analytic[pulse_idx])
    z = np.mean(np.exp(1j * phase_pulses))
    plv = np.abs(z)
    pref = np.angle(z)
    
    return {
        "plv": float(plv),
        "pref_phase_rad": float(pref),
        "phase_pulses_rad": np.asarray(phase_pulses, dtype=np.float64),
        "pulse_indices": np.asarray(pulse_idx, dtype=int),
        "pulse_times_s": np.asarray(pulse_times, dtype=np.float64),
        "band_hz": np.asarray(band, dtype=np.float64),
        "f_center_hz": float(f_center_hz) if f_center_hz is not None and np.isfinite(f_center_hz) else np.nan,
}



def compute_stim_band_hilbert(t: np.ndarray, x: np.ndarray, f_center_hz: float | None = None):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(t) < 4 or x.shape != t.shape:
        return {}
    fs = estimate_fs(t)
    band = get_hilbert_band_hz(fs, f_center_hz=f_center_hz)
    if band is None:
        return {
            "time_s": np.array([], dtype=np.float64),
            "amplitude": np.array([], dtype=np.float64),
            "band_hz": np.array([], dtype=np.float64),
            "f_center_hz": float(f_center_hz) if f_center_hz is not None and np.isfinite(f_center_hz) else np.nan,
            "display_time_s": np.array([], dtype=np.float64),
            "display_amplitude": np.array([], dtype=np.float64),
            "display_stride": 1,
        }

    nyq = 0.5 * fs
    low, high = band
    sos = butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
    xf = sosfiltfilt(sos, x)
    amp = np.abs(hilbert(xf))
    t_disp, [amp_disp], stride = decimate_curve_bundle(t, [amp], max_points=HILBERT_DISPLAY_MAX_POINTS)
    return {
        "time_s": np.asarray(t, dtype=np.float64),
        "amplitude": np.asarray(amp, dtype=np.float64),
        "band_hz": np.asarray(band, dtype=np.float64),
        "f_center_hz": float(f_center_hz) if f_center_hz is not None and np.isfinite(f_center_hz) else np.nan,
        "display_time_s": np.asarray(t_disp, dtype=np.float64),
        "display_amplitude": np.asarray(amp_disp, dtype=np.float64),
        "display_stride": int(stride),
    }


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


def binned_median_values(t: np.ndarray, y: np.ndarray, start_s: float, end_s: float, bin_s: float) -> np.ndarray:
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s or bin_s <= 0:
        return np.array([], dtype=float)
    edges = np.arange(float(start_s), float(end_s) + 0.5 * float(bin_s), float(bin_s), dtype=float)
    if len(edges) < 2 or edges[-1] < end_s:
        edges = np.append(edges, float(end_s))
    vals = []
    for left, right in zip(edges[:-1], edges[1:]):
        keep = (t >= left) & (t < right) & np.isfinite(y)
        if np.any(keep):
            vals.append(float(np.nanmedian(y[keep])))
    return np.asarray(vals, dtype=float)


def compute_hilbert_entrainment(signal_hilbert: dict, stim_end_s: float | None = None) -> dict:
    t = np.asarray(signal_hilbert.get("time_s", []), dtype=float)
    amp = np.asarray(signal_hilbert.get("amplitude", []), dtype=float)
    baseline_start_s = -float(HILBERT_STATS_BASELINE_PRE_SEC)
    out = {
        "baseline_start_s": baseline_start_s,
        "baseline_end_s": float(HILBERT_BASELINE_END_S),
        "stim_end_s": float(stim_end_s) if stim_end_s is not None and np.isfinite(stim_end_s) else np.nan,
        "baseline_median": np.nan,
        "stim_median": np.nan,
        "ratio": np.nan,
        "percent_change": np.nan,
        "baseline_n": 0,
        "stim_n": 0,
        "baseline_bin_median": np.nan,
        "stim_bin_median": np.nan,
        "baseline_vs_stim_u": np.nan,
        "baseline_vs_stim_p": np.nan,
        "baseline_n_bins": 0,
        "stim_n_bins": 0,
        "bin_sec": float(HILBERT_STATS_BIN_SEC),
        "baseline_bin_values": np.array([], dtype=np.float64),
        "stim_bin_values": np.array([], dtype=np.float64),
    }
    if len(t) == 0 or amp.shape != t.shape:
        return out

    base = amp[(t <= float(HILBERT_BASELINE_END_S)) & np.isfinite(amp)]
    stim_stop = float(stim_end_s) if stim_end_s is not None and np.isfinite(stim_end_s) else float(np.nanmax(t))
    stim = amp[(t >= 0) & (t <= stim_stop) & np.isfinite(amp)]
    out["baseline_n"] = int(len(base))
    out["stim_n"] = int(len(stim))
    if len(base) == 0 or len(stim) == 0:
        return out

    baseline_bins = binned_median_values(t, amp, baseline_start_s, float(HILBERT_BASELINE_END_S), float(HILBERT_STATS_BIN_SEC))
    stim_bins = binned_median_values(t, amp, 0.0, stim_stop, float(HILBERT_STATS_BIN_SEC))
    out["baseline_bin_values"] = baseline_bins.astype(np.float64)
    out["stim_bin_values"] = stim_bins.astype(np.float64)
    out["baseline_n_bins"] = int(len(baseline_bins))
    out["stim_n_bins"] = int(len(stim_bins))
    out["baseline_bin_median"] = float(np.nanmedian(baseline_bins)) if len(baseline_bins) else np.nan
    out["stim_bin_median"] = float(np.nanmedian(stim_bins)) if len(stim_bins) else np.nan
    if len(baseline_bins) >= int(HILBERT_STATS_MIN_BASELINE_BINS) and len(stim_bins) >= int(HILBERT_STATS_MIN_STIM_BINS):
        stat = mannwhitneyu(stim_bins, baseline_bins, alternative="two-sided")
        out["baseline_vs_stim_u"] = float(stat.statistic)
        out["baseline_vs_stim_p"] = float(stat.pvalue)

    base_med = float(np.nanmedian(base))
    stim_med = float(np.nanmedian(stim))
    ratio = stim_med / base_med if np.isfinite(base_med) and abs(base_med) > 1e-12 else np.nan
    out.update({
        "stim_end_s": stim_stop,
        "baseline_median": base_med,
        "stim_median": stim_med,
        "ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "percent_change": float((ratio - 1.0) * 100.0) if np.isfinite(ratio) else np.nan,
    })
    return out


def transform_spectrogram_for_display(tsg: np.ndarray, psg_db: np.ndarray) -> tuple[np.ndarray, str, tuple[float, float] | None, TwoSlopeNorm | None]:
    tsg = np.asarray(tsg, dtype=float)
    psg_db = np.asarray(psg_db, dtype=float)
    if psg_db.ndim != 2 or len(tsg) == 0 or psg_db.shape[1] != len(tsg):
        return psg_db, "PSD (dB)", None, None

    p_lin = np.power(10.0, psg_db / 10.0)
    p_show = np.asarray(psg_db, dtype=float)
    cbar_label = "PSD (dB)"
    norm = None
    if SPECTROGRAM_MODE == "relative":
        m_pre = (tsg < 0) & (tsg >= -float(SPECTROGRAM_BASELINE_PRE_SEC))
        if np.any(m_pre):
            baseline_lin = np.nanmean(p_lin[:, m_pre], axis=1, keepdims=True)
        else:
            baseline_lin = np.nanmean(p_lin, axis=1, keepdims=True)
        ratio = p_lin / np.maximum(baseline_lin, 1e-30)
        if SPECTROGRAM_SCALE == "linear":
            p_show = ratio
            cbar_label = "power / pre-stim baseline"
        else:
            p_show = 10.0 * np.log10(np.maximum(ratio, 1e-30))
            cbar_label = "relative PSD (dB re baseline)"
    elif SPECTROGRAM_SCALE == "linear":
        p_show = p_lin
        cbar_label = "PSD (linear)"

    clim = None
    finite = p_show[np.isfinite(p_show)]
    if finite.size:
        lo, hi = [float(v) for v in SPECTROGRAM_DISPLAY_PERCENTILES]
        vmin, vmax = np.percentile(finite, [lo, hi])
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            if SPECTROGRAM_MODE == "relative":
                center = 0.0 if SPECTROGRAM_SCALE == "db" else 1.0
                if float(vmin) < center < float(vmax):
                    norm = TwoSlopeNorm(vmin=float(vmin), vcenter=float(center), vmax=float(vmax))
                else:
                    clim = (float(vmin), float(vmax))
            else:
                clim = (float(vmin), float(vmax))

    return p_show, cbar_label, clim, norm


def build_trial_metrics(
    trial_name: str,
    n_total: int,
    n_used: int,
    f_stim: float,
    spec: dict,
    plv_by_harmonic: dict[int, dict],
    latency: dict | None = None,
    hilbert_entrainment: dict | None = None,
):
    plv = plv_by_harmonic.get(1, {})
    metrics = {
        "trial": trial_name,
        "n_pulses_total": int(n_total),
        "n_pulses_used": int(n_used),
        "f_stim_hz": float(f_stim) if np.isfinite(f_stim) else np.nan,
        "plv": float(plv.get("plv", np.nan)),
    }

    for harmonic, sec in sorted(plv_by_harmonic.items()):
        if int(harmonic) == 1:
            continue
        metrics[f"plv_h{int(harmonic)}"] = float(sec.get("plv", np.nan))
        metrics[f"f_center_h{int(harmonic)}_hz"] = float(sec.get("f_center_hz", np.nan))

    for mul in [1, 2, 3]:
        metrics[f"f{mul}_target_hz"] = float(spec.get(f"f{mul}_target_hz", np.nan))
        metrics[f"f{mul}_power_db"] = float(spec.get(f"f{mul}_power_db", np.nan))
        metrics[f"f{mul}_snr_like_db"] = float(spec.get(f"f{mul}_snr_like_db", np.nan))

    latency = latency or {}
    for i in range(1, LATENCY_MAX_PEAKS + 1):
        metrics[f"peak_{i}_latency_ms"] = float(latency.get(f"peak_{i}_latency_ms", np.nan))
        metrics[f"peak_{i}_amplitude"] = float(latency.get(f"peak_{i}_amplitude", np.nan))
    metrics["n_peak_events"] = int(latency.get("n_peak_events", 0) or 0)

    hilbert_entrainment = hilbert_entrainment or {}
    metrics["hilbert_amp_baseline_median"] = float(hilbert_entrainment.get("baseline_median", np.nan))
    metrics["hilbert_amp_stim_median"] = float(hilbert_entrainment.get("stim_median", np.nan))
    metrics["hilbert_amp_ratio"] = float(hilbert_entrainment.get("ratio", np.nan))
    metrics["hilbert_amp_percent_change"] = float(hilbert_entrainment.get("percent_change", np.nan))
    metrics["hilbert_amp_baseline_vs_stim_u"] = float(hilbert_entrainment.get("baseline_vs_stim_u", np.nan))
    metrics["hilbert_amp_baseline_vs_stim_p"] = float(hilbert_entrainment.get("baseline_vs_stim_p", np.nan))
    metrics["hilbert_amp_baseline_n_bins"] = int(hilbert_entrainment.get("baseline_n_bins", 0) or 0)
    metrics["hilbert_amp_stim_n_bins"] = int(hilbert_entrainment.get("stim_n_bins", 0) or 0)
    metrics["hilbert_amp_baseline_bin_median"] = float(hilbert_entrainment.get("baseline_bin_median", np.nan))
    metrics["hilbert_amp_stim_bin_median"] = float(hilbert_entrainment.get("stim_bin_median", np.nan))

    return metrics


def rebuild_display_pulse_windows_trial(tr: dict, post_periods: float = 3.0):
    t_full = np.asarray(tr.get("t_full_s", []), dtype=float)
    x_full = np.asarray(tr.get("signal_full", []), dtype=float)
    pulse_times = np.asarray(tr.get("pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(t_full) == 0 or x_full.shape != t_full.shape or len(pulse_times) < 2:
        return None, None, None, None

    diffs = np.diff(pulse_times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return None, None, None, None
    ipi_s = float(np.median(diffs))
    pre_s = float(ipi_s)
    post_s = float(post_periods) * float(ipi_s)

    dt = np.diff(t_full)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None, None, None, None
    dt = float(np.median(dt))
    t_plot = np.arange(-pre_s, post_s + 0.5 * dt, dt, dtype=float)

    rows = []
    for tp in pulse_times:
        t_rel = t_full - float(tp)
        keep = (t_rel >= -pre_s) & (t_rel <= post_s)
        if int(np.sum(keep)) < 4:
            continue
        tx = np.asarray(t_rel[keep], dtype=float)
        yx = np.asarray(x_full[keep], dtype=float)
        pre = yx[(tx < 0) & (tx >= -pre_s)]
        baseline = float(np.median(pre)) if len(pre) else 0.0
        rows.append(np.interp(t_plot, tx, yx - baseline, left=np.nan, right=np.nan))
    if not rows:
        return None, None, None, None

    segs = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(segs, axis=0)
        spread = np.nanstd(segs, axis=0, ddof=1) if segs.shape[0] >= 2 else np.full_like(mean, np.nan)
    return t_plot, segs, mean, spread


def analyze_trial(trial_name: str, td_img: dict, td_e: dict):
    x = choose_signal(td_img, SIGNAL_MODE)
    if x is None:
        return None, "missing_signal"

    t = np.asarray(td_img.get("t", []), dtype=float)
    x = np.asarray(x, dtype=float)
    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]
    if len(t) < 4:
        return None, "too_short_trace"

    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < MIN_PULSES:
        return None, "too_few_pulses"

    pre_s, post_s = resolve_pulse_window_s(pulse_times)
    rel_grid = build_rel_grid(t, pre_s, post_s)
    if rel_grid is None:
        return None, "invalid_time_axis"

    global_baseline = compute_global_pre_stim_baseline(t, x, pulse_times) if BASELINE_MODE == "global_pre_stim" else None
    segs, used_idx = extract_all_pulse_segments(t, x, pulse_times, pre_s, post_s, rel_grid, global_baseline=global_baseline)
    if segs is None or len(segs) < MIN_PULSES:
        return None, "too_few_valid_pulses"

    pta_mean = np.nanmean(segs, axis=0)
    pta_sem = nansem(segs, axis=0)
    pta_spread = nansd(segs, axis=0) if SPREAD_MODE == "sd" else pta_sem

    f_stim = estimate_stim_freq_hz(pulse_times)
    used_pulse_times = pulse_times[used_idx]
    cycle_s = estimate_ipi_s(used_pulse_times)
    if (not np.isfinite(cycle_s) or cycle_s <= 0) and np.isfinite(f_stim) and f_stim > 0:
        cycle_s = 1.0 / float(f_stim)
    latency = compute_peak_latency_events(rel_grid, pta_mean, cycle_s)
    t0 = float(pulse_times[0])
    t1 = float(pulse_times[-1])
    m = (t >= t0) & (t <= t1)
    x_stim = x[m]
    t_stim = t[m]
    spec = spectral_peak_metrics(t_stim, x_stim, f_stim) if len(x_stim) >= 16 else {}
    spectrogram_data = compute_spectrogram_data(t, x) if SAVE_SPECTROGRAM else {}
    lfp_spectrogram_data = {}
    if SAVE_LFP_SPECTROGRAM:
        t_lfp, x_lfp = choose_lfp(td_e)
        if t_lfp is not None and x_lfp is not None and len(t_lfp) >= 32:
            window_sec = safe_float(spectrogram_data.get("window_sec"))
            time_ref = np.asarray(spectrogram_data.get("time_s", []), dtype=float)
            freq_ref = np.asarray(spectrogram_data.get("freq_hz", []), dtype=float)
            if len(time_ref) >= 2 and len(freq_ref) >= 2:
                lfp_spectrogram_data = compute_spectrogram_data_matched(
                    t_lfp,
                    x_lfp,
                    window_sec=window_sec,
                    time_ref=time_ref,
                    freq_ref=freq_ref,
                )
            else:
                lfp_spectrogram_data = compute_spectrogram_data_matched(t_lfp, x_lfp, window_sec=window_sec)
                
    harmonics_to_compute = parse_harmonic_selection(PLV_HARMONICS)
    if 1 not in harmonics_to_compute:
        harmonics_to_compute = [1] + harmonics_to_compute
    plv_by_harmonic = {}
    for harmonic in harmonics_to_compute:
        f_center = float(harmonic) * f_stim if np.isfinite(f_stim) else np.nan
        plv_by_harmonic[int(harmonic)] = (
            compute_plv(t, x, used_pulse_times, f_center_hz=f_center)
            if np.isfinite(f_center)
            else {}
        )

    hilbert_harmonics_to_compute = parse_harmonic_selection(SIGNAL_HILBERT_HARMONICS)
    if 1 not in hilbert_harmonics_to_compute:
        hilbert_harmonics_to_compute = [1] + hilbert_harmonics_to_compute
    signal_hilbert_by_harmonic = {}
    if SAVE_STIM_BAND_HILBERT:
        for harmonic in hilbert_harmonics_to_compute:
            f_center = float(harmonic) * f_stim if np.isfinite(f_stim) else np.nan
            signal_hilbert_by_harmonic[int(harmonic)] = (
                compute_stim_band_hilbert(t, x, f_center_hz=f_center)
                if np.isfinite(f_center)
                else {}
            )

    signal_hilbert = signal_hilbert_by_harmonic.get(1, {})
    stim_end_s = float(np.nanmax(used_pulse_times)) if len(used_pulse_times) else np.nan
    hilbert_entrainment = compute_hilbert_entrainment(signal_hilbert, stim_end_s=stim_end_s)
    lfp_hilbert = {}
    if SAVE_STIM_BAND_HILBERT:
        t_lfp, x_lfp = choose_lfp(td_e)
        if t_lfp is not None and x_lfp is not None:
            t_lfp_clip, x_lfp_clip = clip_trace_to_support(t_lfp, x_lfp, float(t[0]), float(t[-1]))
            if t_lfp_clip is not None and x_lfp_clip is not None:
                lfp_hilbert = compute_stim_band_hilbert(t_lfp_clip, x_lfp_clip, f_center_hz=f_stim)

    result = {
        "trial": trial_name,
        "t_full_s": t.astype(np.float64),
        "signal_full": x.astype(np.float64),
        "pulse_times_s": pulse_times.astype(np.float64),
        "n_pulses_total": int(len(pulse_times)),
        "n_pulses_used": int(len(used_idx)),
        "used_pulse_indices": np.asarray(used_idx, dtype=int),
        "f_stim_hz": float(f_stim) if np.isfinite(f_stim) else np.nan,
        "t_rel_s": rel_grid.astype(np.float64),
        "window_pre_s": float(pre_s),
        "window_post_s": float(post_s),
        "pulse_segments": segs.astype(np.float64),
        "pta_mean": pta_mean.astype(np.float64),
        "pta_sem": pta_sem.astype(np.float64),
        "pta_spread": pta_spread.astype(np.float64),
        "spectral": spec,
        "spectrogram": spectrogram_data,
        "lfp_spectrogram": lfp_spectrogram_data,
        "signal_hilbert": signal_hilbert,
        "hilbert_entrainment": hilbert_entrainment,
        "lfp_hilbert": lfp_hilbert,
        "latency": latency,
    }
    for harmonic, sec in sorted(signal_hilbert_by_harmonic.items()):
        result[signal_hilbert_section_key(harmonic)] = sec
    for harmonic, sec in sorted(plv_by_harmonic.items()):
        result[plv_section_key(harmonic)] = sec
    result["metrics"] = build_trial_metrics(
        trial_name=trial_name,
        n_total=len(pulse_times),
        n_used=len(used_idx),
        f_stim=f_stim,
        spec=spec,
        plv_by_harmonic=plv_by_harmonic,
        latency=latency,
        hilbert_entrainment=hilbert_entrainment,
    )
    return result, None


def collect_plv_phases(trial_results: dict, section_key: str = "plv") -> np.ndarray:
    phases = []
    for name in sorted(trial_results.keys(), key=trial_sort_key):
        ph = np.asarray(trial_results[name].get(section_key, {}).get("phase_pulses_rad", []), dtype=float)
        ph = ph[np.isfinite(ph)]
        if len(ph):
            phases.append(ph)
    return np.concatenate(phases) if phases else np.array([], dtype=float)


def plot_plv_histogram_ax(ax, trial_results: dict, title: str, section_key: str = "plv", label: str = "PLV"):
    phases = collect_plv_phases(trial_results, section_key=section_key)
    if len(phases) == 0:
        ax.set_axis_off()
        ax.set_title(f"{title} | no PLV phases")
        return

    z = np.mean(np.exp(1j * phases))
    plv = float(np.abs(z))
    pref = float(np.angle(z))
    counts, edges = np.histogram(phases, bins=np.linspace(-np.pi, np.pi, 37))
    rmax = max(1.0, float(np.max(counts)))
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color="tab:blue",
        alpha=0.45,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.annotate("", xy=(pref, plv * rmax), xytext=(0, 0), arrowprops=dict(color="crimson", lw=2.5, arrowstyle="->"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"{title}\n{label}={plv:.3f}, n={len(phases)}")


def plot_trial_examples(block_label: str, trial_results: dict):
    names = sorted(trial_results.keys(), key=trial_sort_key)
    if not names:
        return
    idxs = np.linspace(0, len(names) - 1, min(N_TO_PLOT, len(names))).astype(int)
    names_plot = [names[i] for i in idxs]

    fig, axes = plt.subplots(len(names_plot), 4, figsize=(20, 3.8 * len(names_plot)))
    if len(names_plot) == 1:
        axes = np.array([axes])

    for r, name in enumerate(names_plot):
        tr = trial_results[name]
        ax0 = axes[r, 0]
        ax1 = axes[r, 1]
        ax2 = axes[r, 2]
        ax3 = axes[r, 3]

        t_full = tr["t_full_s"]
        x_full = tr["signal_full"]
        pulse_times = tr["pulse_times_s"]
        t_rel = tr["t_rel_s"]
        segs = tr["pulse_segments"]
        mean = tr["pta_mean"]
        sem = tr["pta_spread"] if "pta_spread" in tr else tr["pta_sem"]
        f_stim = tr["f_stim_hz"]
        spec = tr.get("spectral", {})
        f = np.asarray(spec.get("freq_hz", []), dtype=float)
        p = np.asarray(spec.get("psd_db", []), dtype=float)

        # (1) Full trace with pulse times marked.
        ax0.plot(t_full, x_full, color="tab:blue", lw=0.8)
        if len(pulse_times):
            for tp in pulse_times:
                ax0.axvline(float(tp), color="tab:red", alpha=0.12, lw=0.7)
        ax0.set_title(f"{name} | full trace")
        ax0.set_xlabel("time from stim onset (s)")
        ax0.set_ylabel("signal (a.u.)")

        # (2) Pulse-triggered windows within trial.
        display = rebuild_display_pulse_windows_trial(tr, post_periods=3.0)
        if display[0] is not None:
            t_rel_plot, segs_plot, mean_plot, sem_plot = display
        else:
            t_rel_plot, segs_plot, mean_plot, sem_plot = t_rel, segs, mean, sem
        if SHOW_PULSE_WINDOWS:
            for k in range(min(40, segs_plot.shape[0])):
                ax1.plot(t_rel_plot, segs_plot[k], color="tab:blue", alpha=0.15, lw=0.7)
        ax1.plot(t_rel_plot, mean_plot, color="black", lw=2.0, label="within-trial mean")
        spread_label = "SD" if SPREAD_MODE == "sd" else "SEM"
        ax1.fill_between(t_rel_plot, mean_plot - sem_plot, mean_plot + sem_plot, color="black", alpha=0.2, label=spread_label)
        ax1.axvline(0.0, color="tab:red", ls="--", lw=1.1, label="aligned pulse")
        if len(pulse_times) >= 2:
            last_pulse_rel = float(pulse_times[-1] - pulse_times[0])
            if t_rel_plot[0] <= last_pulse_rel <= t_rel_plot[-1]:
                ax1.axvline(last_pulse_rel, color="tab:orange", ls="--", lw=1.1, label="last pulse")
        ax1.set_title(f"{name} | pulse windows | pulses={tr['n_pulses_used']}")
        ax1.set_xlabel("time from pulse (s)")
        ax1.set_ylabel("signal (a.u.)")
        ax1.set_xlim(float(np.min(t_rel_plot)), float(np.max(t_rel_plot)))
        ax1.legend(loc="best", fontsize=8)

        # (3) FFT/periodogram in stimulation epoch.
        if len(f) and len(p):
            ax2.plot(f, p, color="tab:purple", lw=1.0)
            if np.isfinite(f_stim):
                nyq = float(np.max(f))
                for mul in [1, 2, 3]:
                    ft = mul * f_stim
                    if ft < nyq:
                        ax2.axvline(ft, color="tab:red", ls="--", alpha=0.6)
            ax2.set_xlim(0, min(250.0, np.max(f)))
            ax2.set_xlabel("frequency (Hz)")
            ax2.set_ylabel("PSD (dB)")
            ax2.set_title(f"{name} | FFT spectrum | f={f_stim:.1f} Hz")
        else:
            ax2.axis("off")

        # (4) Spectrogram over full trial.
        fs = estimate_fs(t_full)
        specgram = tr.get("spectrogram", {})
        fsg = np.asarray(specgram.get("freq_hz", []), dtype=float)
        tsg = np.asarray(specgram.get("time_s", []), dtype=float)
        psg_db = np.asarray(specgram.get("power_db", []), dtype=float)
        if len(fsg) and len(tsg) and psg_db.shape == (len(fsg), len(tsg)):
            p_show, cbar_label, clim, norm = transform_spectrogram_for_display(tsg, psg_db)
            im = ax3.imshow(
                p_show,
                origin="lower",
                aspect="auto",
                cmap=SPECTROGRAM_CMAP,
                extent=[float(tsg[0]), float(tsg[-1]), float(fsg[0]), float(fsg[-1])],
                interpolation=SPECTROGRAM_INTERPOLATION,
                norm=norm,
            )
            if norm is None and clim is not None:
                im.set_clim(*clim)
            if np.isfinite(f_stim):
                ax3.axhline(f_stim, color="tab:red", ls="--", lw=1.0, alpha=0.8)
            ax3.set_ylim(0, min(250.0, np.max(fsg)))
            ax3.set_xlabel("time from stim onset (s)")
            ax3.set_ylabel("frequency (Hz)")
            mode_label = "relative" if SPECTROGRAM_MODE == "relative" else "absolute"
            ax3.set_title(f"{name} | spectrogram ({mode_label}, {SPECTROGRAM_SCALE})")
            cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)
        else:
            ax3.axis("off")

    fig.suptitle(f"Pulse-Train PTA Prototype | {block_label}", y=0.995)
    plt.tight_layout()
    plt.show()

    for harmonic in parse_harmonic_selection(SHOW_PLV_HISTOGRAMS):
        label = plv_label(harmonic)
        fig2, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
        plot_plv_histogram_ax(
            ax,
            trial_results,
            f"{label} phase histogram | {block_label}",
            section_key=plv_section_key(harmonic),
            label=label,
        )
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

    trial_names = sorted(img_trials.keys(), key=trial_sort_key)
    results = {}
    fail_counts = {}

    for name in trial_names:
        if ONLY_TRIAL is not None and name != ONLY_TRIAL:
            continue
        if name not in e_trials:
            fail_counts["missing_ephys_trial"] = fail_counts.get("missing_ephys_trial", 0) + 1
            continue
        tr, err = analyze_trial(name, img_trials[name], e_trials[name])
        if tr is None:
            fail_counts[err] = fail_counts.get(err, 0) + 1
            continue
        results[name] = tr

    if not results:
        print(f"[SKIP] no valid trial-level train PTA results: {imaging_path}")
        if fail_counts:
            print(f"[INFO] fail reasons: {fail_counts}")
        return

    block_label = f"{img.get('date')} {img.get('block')}"
    print(f"[RUN] {block_label} | {imaging_path.name}")
    print(f"[INFO] valid trials: {len(results)} / {len(trial_names)}")
    if fail_counts:
        print(f"[INFO] skipped: {fail_counts}")

    for name in sorted(results.keys(), key=trial_sort_key):
        m = results[name]["metrics"]
        harmonic_parts = []
        for h in parse_harmonic_selection(PLV_HARMONICS):
            if h == 1 or f"plv_h{h}" not in m:
                continue
            value = m.get(f"plv_h{h}", np.nan)
            if np.isfinite(value):
                harmonic_parts.append(f"PLV_H{h}={value:.3f}")
        harmonic_text = " ".join(harmonic_parts)
        harmonic_text = f" | {harmonic_text}" if harmonic_text else ""
        print(
            f"[METRIC] {name} | f={m['f_stim_hz']:.2f} Hz | PLV={m['plv']:.3f}{harmonic_text} | "
            f"f1_snr={m['f1_snr_like_db']:.2f} dB | f2_snr={m['f2_snr_like_db']:.2f} dB | "
            f"f3_snr={m['f3_snr_like_db']:.2f} dB | pulses={m['n_pulses_used']}/{m['n_pulses_total']}"
        )

    if SHOW_PLOTS:
        plot_trial_examples(block_label, results)

    if SAVE_OUTPUT:
        out_path = imaging_path.parent / f"{imaging_path.stem}_pta_train.pkl"
        trial_metrics = {name: results[name]["metrics"] for name in sorted(results.keys(), key=trial_sort_key)}
        out = {
            "mouse": img.get("mouse"),
            "date": img.get("date"),
            "block": img.get("block"),
            "analysis": "pulse_train_pta_prototype",
            "settings": {
                "signal_mode": SIGNAL_MODE,
                "plot_pre_sec": float(PRE_SEC),
                "plot_post_sec": float(POST_SEC),
                "extract_pre_sec": float(EXTRACT_PRE_SEC),
                "extract_post_sec": float(EXTRACT_POST_SEC),
                "pulse_window_mode": PULSE_WINDOW_MODE,
                "pulse_window_scale": float(PULSE_WINDOW_SCALE),
                "period_fraction": float(PERIOD_FRACTION),
                "baseline_mode": BASELINE_MODE,
                "global_baseline_stat": GLOBAL_BASELINE_STAT,
                "global_baseline_pre_sec": float(GLOBAL_BASELINE_PRE_SEC),
                "min_pulses": int(MIN_PULSES),
                "show_pulse_windows": bool(SHOW_PULSE_WINDOWS),
                "plv_harmonics": PLV_HARMONICS,
                "show_plv_histograms": SHOW_PLV_HISTOGRAMS,
                "signal_hilbert_harmonics": SIGNAL_HILBERT_HARMONICS,
                "spread_mode": SPREAD_MODE,
                "save_spectrogram": bool(SAVE_SPECTROGRAM),
                "save_lfp_spectrogram": bool(SAVE_LFP_SPECTROGRAM),
                "spectrogram_window": SPECTROGRAM_WINDOW,
                "spectrogram_nperseg": int(NFFT_SPEC),
                "spectrogram_target_window_sec": float(SPECTROGRAM_TARGET_WINDOW_SEC),
                "spectrogram_overlap_frac": float(SPECTROGRAM_OVERLAP_FRAC),
                "spectrogram_relative_baseline_end_s": float(SPECTROGRAM_BASELINE_END_S),
                "spectrogram_relative_baseline_stat": SPECTROGRAM_BASELINE_STAT,
                "spectrogram_mode": SPECTROGRAM_MODE,
                "spectrogram_scale": SPECTROGRAM_SCALE,
                "spectrogram_baseline_pre_sec": float(SPECTROGRAM_BASELINE_PRE_SEC),
                "spectrogram_display_percentiles": [float(v) for v in SPECTROGRAM_DISPLAY_PERCENTILES],
                "spectrogram_interpolation": SPECTROGRAM_INTERPOLATION,
                "save_stim_band_hilbert": bool(SAVE_STIM_BAND_HILBERT),
                "hilbert_half_band_hz": float(HILBERT_HALF_BAND_HZ),
                "hilbert_relative_baseline_end_s": float(HILBERT_BASELINE_END_S),
                "hilbert_relative_baseline_stat": HILBERT_BASELINE_STAT,
                "hilbert_stats_baseline_start_s": -float(HILBERT_STATS_BASELINE_PRE_SEC),
                "hilbert_stats_baseline_end_s": float(HILBERT_BASELINE_END_S),
                "hilbert_stats_bin_sec": float(HILBERT_STATS_BIN_SEC),
                "hilbert_stats_min_baseline_bins": int(HILBERT_STATS_MIN_BASELINE_BINS),
                "hilbert_stats_min_stim_bins": int(HILBERT_STATS_MIN_STIM_BINS),
                "latency_threshold_sd": float(LATENCY_THRESHOLD_SD),
                "latency_max_peaks": int(LATENCY_MAX_PEAKS),
                "latency_top_samples": int(LATENCY_TOP_SAMPLES),
                "only_trial": ONLY_TRIAL,
            },
            "trial_results": results,
            "trial_metrics": trial_metrics,
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


