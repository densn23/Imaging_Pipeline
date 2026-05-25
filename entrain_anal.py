from pathlib import Path
import csv
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon


DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent
STIM_TABLE_CSV = DATA_ANALYSIS_ROOT / "tables" / "stim_table_all_jamie.csv"
FIGURES_DIR = DATA_ANALYSIS_ROOT / "figures"

# -------------------------
# Recording
# -------------------------
ANALYSIS_MODE = "condition"  # "block" or "condition"; condition supports amplitude="baseline"
MOUSE_NAME = "Jamie10, Jamie11, Vinnie1"
SINGLE_DATE = None
SINGLE_BLOCK = None
ONLY_TRIAL = None  # e.g. "R2_1"; None = all trials
RUN_BATCH = False  # True = process matching blocks; set SINGLE_BLOCK=None for all blocks in a date

# -------------------------
# Condition mode filters
# -------------------------
CONDITION_FREQUENCY_HZ = 135
CONDITION_AMPLITUDE_UA = 25
CONDITION_PULSE_WIDTH_US = 100
CONDITION_EXPOSURE_MS = None
CONDITION_STIMULATION_TIME_S = 10
CONDITION_PHASE = "Pre"
CONDITION_IMAGING_SIDE = "Left"
CONDITION_DATE = None
CONDITION_BLOCK = None
LOW_VELOCITY_PERCENTILE = 25.0
HIGH_VELOCITY_PERCENTILE = 75.0
MIN_TRIALS_PER_VELOCITY_GROUP = 2

# -------------------------
# Analysis
# -------------------------
SIGNAL_MODE = "notched"  # "notched", "bleach", or "raw"
BIN_SEC = 0.5
HILBERT_HALF_BAND_HZ = 2.0
ANALYZE_THETA_BAND = False
THETA_BAND_HZ = (4.0, 12.0)
BASELINE_END_S = -0.5
ANALYSIS_START_S = 0.0
ANALYSIS_END_S = None  # None = last DBS pulse
MOVEMENT_THRESHOLD_CM_S = 5.0
MOVING_BIN_FRACTION_CUTOFF = 0.5
MOVING_TRIAL_FRACTION_CUTOFF = 0.10
SPTA_PEAK_WINDOW_S = (0.0, 0.005)
VM_EARLY_WINDOW_S = (0.0, 1.0)
VM_LATE_WINDOW_SEC = 5.0
ANALYZE_VM_VELOCITY = False
ANALYZE_HILBERT_VELOCITY = True
ANALYZE_HILBERT_VM = True

# -------------------------
# Output
# -------------------------
SHOW_FIGURES = True
PRINT_TRIAL_TABLE = True
PRINT_BLOCK_TRIAL_METRICS = True
BLOCK_TRIAL_METRICS = (
    #hilbert_amp, plv, plv_pooled, spta_lat, mpta_lat, mpta_amp, "
    #spta_jit, mpta_jit, vm_early, vm_late, "
    #r_vm_velocity, 
    "r_hilbert_velocity, r_hilbert_vm"
)
CONDITION_PLOT_METRICS = (
    # "all" plots the full velocity figure.
    # Options: velocity_split, vm_velocity_split, hilbert_velocity_split,
    # r_vm_velocity, r_hilbert_velocity, r_hilbert_vm, r_theta_hilbert,
    # r_theta_velocity, r_theta_vm
    "velocity_split, r_hilbert_velocity, r_hilbert_vm"
)
SAVE_OUTPUT = False
SAVE_FIGURE = False
FIG_DPI = 300
SHOW_EPHYS_CHECK_SUMMARY = False
STIM_UA_PER_V = 10.0
GEVI_DISPLAY_SCALE = 100.0  # stored dF/F is fractional; plots show percent dF/F
GEVI_YLABEL = "dF/F"


NUMERIC_TOLERANCE = {
    "frequency_hz": 0.6,
    "amplitude_uA": 1.0,
    "pulse_width_us": 1.0,
    "exposure_ms": 0.2,
    "stimulation_time_s": 0.25,
}
BASELINE_AMPLITUDE_LABELS = {"baseline"}
IMAGING_SIDE_FLIP_MICE = {"vinnie1"}  # requested Left selects actual Right for Vinnie1, and vice versa


def processed_path_for(mouse: str, date: str, block: str) -> Path:
    if mouse is None or date is None or block is None:
        raise ValueError("Set MOUSE_NAME, SINGLE_DATE, and SINGLE_BLOCK for block mode, or use ANALYSIS_MODE='condition' / RUN_BATCH=True.")
    return (
        DATA_ANALYSIS_ROOT
        / mouse
        / "Imaging_Data"
        / date
        / block
        / f"{block}_traces_processed_notched.pkl"
    )


def ephys_path_for(mouse: str, date: str, block: str) -> Path:
    return (
        DATA_ANALYSIS_ROOT
        / mouse
        / "Open_Ephys"
        / date
        / block
        / f"{block}_epoched_ephys.pkl"
    )


def processed_path() -> Path:
    return processed_path_for(MOUSE_NAME, SINGLE_DATE, SINGLE_BLOCK)


def ephys_path() -> Path:
    return ephys_path_for(MOUSE_NAME, SINGLE_DATE, SINGLE_BLOCK)


def train_pta_path_for(mouse: str, date: str, block: str) -> Path:
    return processed_path_for(mouse, date, block).with_name(f"{block}_traces_processed_notched_pta_train.pkl")


def train_pta_path() -> Path:
    return train_pta_path_for(MOUSE_NAME, SINGLE_DATE, SINGLE_BLOCK)


def first_pta_path_for(mouse: str, date: str, block: str) -> Path:
    return processed_path_for(mouse, date, block).with_name(f"{block}_traces_processed_notched_pta_first_pulse.pkl")


def first_pta_path() -> Path:
    return first_pta_path_for(MOUSE_NAME, SINGLE_DATE, SINGLE_BLOCK)


def entrainment_output_path() -> Path:
    return processed_path().with_name(f"{SINGLE_BLOCK}_entrainment_analysis.pkl")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def iter_batch_blocks():
    imaging_root = DATA_ANALYSIS_ROOT / MOUSE_NAME / "Imaging_Data"
    if SINGLE_DATE is None:
        date_dirs = [p for p in sorted(imaging_root.iterdir()) if p.is_dir()]
    else:
        date_dirs = [imaging_root / SINGLE_DATE]

    for date_dir in date_dirs:
        if not date_dir.exists():
            continue
        if SINGLE_BLOCK is None:
            block_dirs = [p for p in sorted(date_dir.iterdir()) if p.is_dir() and p.name.startswith("R")]
        else:
            block_dirs = [date_dir / SINGLE_BLOCK]
        for block_dir in block_dirs:
            block = block_dir.name
            img_p = block_dir / f"{block}_traces_processed_notched.pkl"
            eph_p = DATA_ANALYSIS_ROOT / MOUSE_NAME / "Open_Ephys" / date_dir.name / block / f"{block}_epoched_ephys.pkl"
            if img_p.exists() and eph_p.exists():
                yield date_dir.name, block


def safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def gevi_display(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * float(GEVI_DISPLAY_SCALE)


def centered_fractional_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    center = float(np.nanmedian(x[np.isfinite(x)]))
    if np.isfinite(center) and abs(center) > 1e-12:
        return (x - center) / center
    return x - center


def median_float(values) -> float:
    arr = np.asarray([safe_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if len(arr) else np.nan


def trim_float(value, digits: int = 3) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "?"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_col(value, digits: int = 3) -> str:
    value = safe_float(value)
    return "?" if not np.isfinite(value) else f"{value:.{digits}g}"


def parse_mouse_names(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "all"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def selection_values(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = re.split(r"[,;+]", str(raw))
    return [str(value).strip() for value in values if str(value).strip()]


def numeric_match(actual: float, target, key: str) -> bool:
    if target is None:
        return True
    targets = target if isinstance(target, (list, tuple, set)) else selection_values(target)
    for item in targets:
        item_val = safe_float(item)
        if not np.isfinite(actual) or not np.isfinite(item_val):
            continue
        if abs(actual - item_val) <= NUMERIC_TOLERANCE.get(key, 0.0):
            return True
    return False


def text_match(actual, target) -> bool:
    if target is None:
        return True
    return str(actual or "").strip().lower() == str(target).strip().lower()


def normalize_text_label(value) -> str:
    return str(value or "").strip().lower()


def normalize_class_label(value) -> str:
    return re.sub(r"\s+", " ", normalize_text_label(value))


def normalize_mouse_label(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def flip_left_right_label(value) -> str:
    label = normalize_text_label(value)
    if label == "left":
        return "right"
    if label == "right":
        return "left"
    return label


def effective_imaging_side_for_mouse(requested_side, mouse_name) -> str:
    requested = normalize_text_label(requested_side)
    if normalize_mouse_label(mouse_name) in IMAGING_SIDE_FLIP_MICE:
        return flip_left_right_label(requested)
    return requested


def imaging_side_matches(row: dict[str, str], requested_side) -> bool:
    requested_values = selection_values(requested_side)
    if not requested_values:
        return True
    actual = normalize_text_label(row.get("imaging_side"))
    mouse = row.get("mouse")
    return any(actual == effective_imaging_side_for_mouse(requested, mouse) for requested in requested_values)


def is_baseline_condition_request() -> bool:
    return any(
        normalize_class_label(value) in BASELINE_AMPLITUDE_LABELS
        for value in selection_values(CONDITION_AMPLITUDE_UA)
    )


def row_is_true_baseline(row: dict[str, str]) -> bool:
    return (
        normalize_class_label(row.get("protocol")) == "baseline"
        and not np.isfinite(safe_float(row.get("frequency_hz")))
        and not np.isfinite(safe_float(row.get("amplitude_uA")))
        and not np.isfinite(safe_float(row.get("pulse_width_us")))
    )


def row_matches_condition(row: dict[str, str]) -> bool:
    mice = parse_mouse_names(MOUSE_NAME)
    if mice and normalize_mouse_label(row.get("mouse", "")) not in {normalize_mouse_label(mouse) for mouse in mice}:
        return False
    if CONDITION_DATE is not None and row.get("date") != CONDITION_DATE:
        return False
    if CONDITION_BLOCK is not None and row.get("block") != CONDITION_BLOCK:
        return False
    if not text_match(row.get("phase"), CONDITION_PHASE):
        return False
    if not imaging_side_matches(row, CONDITION_IMAGING_SIDE):
        return False
    if not numeric_match(safe_float(row.get("exposure_ms")), CONDITION_EXPOSURE_MS, "exposure_ms"):
        return False

    if is_baseline_condition_request():
        return row_is_true_baseline(row)

    if not numeric_match(safe_float(row.get("frequency_hz")), CONDITION_FREQUENCY_HZ, "frequency_hz"):
        return False
    if not numeric_match(safe_float(row.get("amplitude_uA")), CONDITION_AMPLITUDE_UA, "amplitude_uA"):
        return False
    if not numeric_match(safe_float(row.get("pulse_width_us")), CONDITION_PULSE_WIDTH_US, "pulse_width_us"):
        return False
    if not numeric_match(safe_float(row.get("stimulation_time_s")), CONDITION_STIMULATION_TIME_S, "stimulation_time_s"):
        return False
    return True


def iter_condition_blocks():
    seen = set()
    with STIM_TABLE_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row_matches_condition(row):
                continue
            key = (row.get("mouse", ""), row.get("date", ""), row.get("block", ""))
            if key in seen:
                continue
            seen.add(key)
            mouse, date, block = key
            img_p = processed_path_for(mouse, date, block)
            eph_p = ephys_path_for(mouse, date, block)
            if img_p.exists() and eph_p.exists():
                yield mouse, date, block


def condition_output_stem() -> str:
    parts = [
        str(MOUSE_NAME).replace(",", "_"),
        "condition",
        f"f{CONDITION_FREQUENCY_HZ}" if CONDITION_FREQUENCY_HZ is not None else "fany",
        f"amp{CONDITION_AMPLITUDE_UA}" if CONDITION_AMPLITUDE_UA is not None else "ampany",
        f"pw{CONDITION_PULSE_WIDTH_US}" if CONDITION_PULSE_WIDTH_US is not None else "pwany",
        f"exp{CONDITION_EXPOSURE_MS}" if CONDITION_EXPOSURE_MS is not None else "expany",
        f"stim{CONDITION_STIMULATION_TIME_S}" if CONDITION_STIMULATION_TIME_S is not None else "stimany",
    ]
    text = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def figure_output_path(stem: str) -> Path:
    FIGURES_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(stem)).strip("_") or "figure"
    return FIGURES_DIR / f"{safe}.png"


def save_figure(fig, stem: str) -> None:
    if not SAVE_FIGURE:
        return
    out_path = figure_output_path(stem)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[saved figure] {out_path}")


def pulse_train_metrics_from_times(pulse_times: np.ndarray) -> dict:
    pulse_times = np.asarray(pulse_times, dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < 2:
        return {"frequency_hz": np.nan, "stim_time_s": np.nan, "n_pulses": int(len(pulse_times))}
    isi = np.diff(pulse_times)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    freq = 1.0 / float(np.nanmedian(isi)) if len(isi) else np.nan
    return {
        "frequency_hz": freq,
        "stim_time_s": float(pulse_times[-1] - pulse_times[0]) if len(pulse_times) else np.nan,
        "n_pulses": int(len(pulse_times)),
    }


def stim_waveform_metrics_from_trace(t: np.ndarray, y: np.ndarray, pulse_times: np.ndarray) -> dict:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    pulse_times = np.asarray(pulse_times, dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(t) < 2 or y.shape != t.shape or len(pulse_times) == 0:
        return {"amp_v": np.nan, "pulse_width_us": np.nan}

    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return {"amp_v": np.nan, "pulse_width_us": np.nan}

    amps = []
    widths = []
    for tp in pulse_times[: min(30, len(pulse_times))]:
        keep = (t >= tp - 0.001) & (t <= tp + 0.002)
        if int(np.sum(keep)) < 3:
            continue
        tx = t[keep]
        yy = y[keep]
        base = np.nanmedian(yy[tx < tp]) if np.any(tx < tp) else np.nanmedian(yy)
        yy = yy - base
        peak_idx = int(np.nanargmax(np.abs(yy)))
        peak = float(yy[peak_idx])
        amps.append(abs(peak))
        half = 0.5 * abs(peak)
        above = np.abs(yy) >= half
        if not np.any(above):
            continue
        idx = np.where(above)[0]
        widths.append(float((idx[-1] - idx[0] + 1) * dt * 1e6))

    return {"amp_v": median_float(amps), "pulse_width_us": median_float(widths)}


def imaging_frame_interval_ms_from_trials(trials: dict) -> float:
    vals = []
    for td in trials.values():
        fps = safe_float(td.get("fps_hz"))
        if np.isfinite(fps) and fps > 0:
            vals.append(1000.0 / fps)
            continue
        t = np.asarray(td.get("t", []), dtype=float)
        t = t[np.isfinite(t)]
        if len(t) >= 2:
            vals.append(float(np.nanmedian(np.diff(t)) * 1000.0))
    return median_float(vals)


def ephys_check_for_block(mouse: str, date: str, block: str) -> dict:
    img_p = processed_path_for(mouse, date, block)
    eph_p = ephys_path_for(mouse, date, block)
    img = load_pickle(img_p) if img_p.exists() else {}
    eph = load_pickle(eph_p) if eph_p.exists() else {}
    eph_trials = eph.get("trials", {})
    stim_names = sorted(eph_trials.keys())

    freqs = []
    stim_times = []
    pulse_counts = []
    amps_v = []
    widths_us = []
    for name in stim_names:
        td_e = eph_trials.get(name, {})
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        train = pulse_train_metrics_from_times(pulse_times)
        if train["n_pulses"] < 2:
            continue
        freqs.append(train["frequency_hz"])
        stim_times.append(train["stim_time_s"])
        pulse_counts.append(train["n_pulses"])

        t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
        y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
        wave = stim_waveform_metrics_from_trace(t, y, pulse_times)
        amps_v.append(wave["amp_v"])
        widths_us.append(wave["pulse_width_us"])

    amp_v = median_float(amps_v)
    return {
        "n_trials": float(len([n for n in stim_names if len(np.asarray(eph_trials[n].get("stim_pulse_times_s", []))) >= 2])),
        "frequency_hz": median_float(freqs),
        "stim_time_s": median_float(stim_times),
        "amp_v": amp_v,
        "amp_uA_est": amp_v * STIM_UA_PER_V if np.isfinite(amp_v) else np.nan,
        "pulse_width_us": median_float(widths_us),
        "exposure_ms": imaging_frame_interval_ms_from_trials(img.get("trials", {})),
        "pulses_per_trial": median_float(pulse_counts),
    }


def print_ephys_check_summary(blocks: list[tuple[str, str, str]]) -> None:
    if not SHOW_EPHYS_CHECK_SUMMARY:
        return
    print("\nEphys/data-derived check:")
    print("  ampV comes from stim channel; amp~uA uses STIM_UA_PER_V.")
    print("  exp is derived from imaging frame interval/fps.")
    for mouse, date, block in blocks:
        check = ephys_check_for_block(mouse, date, block)
        print(
            f"  {mouse} | {date} | {block} | "
            f"freq={trim_float(check['frequency_hz'])} | "
            f"ampV={trim_float(check['amp_v'])} | "
            f"amp~uA={trim_float(check['amp_uA_est'])} | "
            f"PW={trim_float(check['pulse_width_us'])} | "
            f"exp={trim_float(check['exposure_ms'])} | "
            f"stim={trim_float(check['stim_time_s'])} | "
            f"n_trials={trim_float(check['n_trials'])} | "
            f"pulses/trial={trim_float(check['pulses_per_trial'])}"
        )


def choose_signal(trial: dict) -> np.ndarray | None:
    if SIGNAL_MODE == "raw":
        key = "F_raw"
    elif SIGNAL_MODE == "bleach":
        key = "F_bleach_corr"
    else:
        key = "F_notched"
    value = trial.get(key)
    return None if value is None else np.asarray(value, dtype=float)


def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / float(np.median(dt)) if len(dt) else np.nan


def estimate_stim_freq(pulse_times: np.ndarray) -> float:
    isi = np.diff(np.asarray(pulse_times, dtype=float))
    isi = isi[np.isfinite(isi) & (isi > 0)]
    return 1.0 / float(np.median(isi)) if len(isi) else np.nan


def stim_band_hilbert_amp(t: np.ndarray, x: np.ndarray, f_center_hz: float) -> np.ndarray:
    fs = estimate_fs(t)
    if not np.isfinite(fs) or not np.isfinite(f_center_hz):
        return np.full_like(x, np.nan, dtype=float)
    nyq = 0.5 * fs
    low = max(0.1, float(f_center_hz) - float(HILBERT_HALF_BAND_HZ))
    high = float(f_center_hz) + float(HILBERT_HALF_BAND_HZ)
    if high >= nyq or low <= 0:
        return np.full_like(x, np.nan, dtype=float)
    sos = butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
    xf = sosfiltfilt(sos, x)
    return np.abs(hilbert(xf))


def fixed_band_hilbert_amp(t: np.ndarray, x: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    fs = estimate_fs(t)
    if not np.isfinite(fs):
        return np.full_like(x, np.nan, dtype=float)
    low, high = [float(v) for v in band_hz]
    nyq = 0.5 * fs
    if low <= 0 or high <= low or high >= nyq:
        return np.full_like(x, np.nan, dtype=float)
    sos = butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
    xf = sosfiltfilt(sos, x)
    return np.abs(hilbert(xf))


def amp_baseline_stim_ratio(
    t: np.ndarray,
    amp: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[float, float, float]:
    baseline = amp[(t <= BASELINE_END_S) & np.isfinite(amp)]
    base_amp = float(np.nanmedian(baseline)) if len(baseline) else np.nan
    stim = amp[(t >= t_start) & (t <= t_end) & np.isfinite(amp)]
    stim_amp = float(np.nanmedian(stim)) if len(stim) else np.nan
    ratio = stim_amp / base_amp if np.isfinite(stim_amp) and np.isfinite(base_amp) and base_amp > 1e-12 else np.nan
    return safe_float(ratio), safe_float(base_amp), safe_float(stim_amp)


def binned_mean(t: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    for i in range(len(out)):
        keep = (t >= edges[i]) & (t < edges[i + 1]) & np.isfinite(y)
        if np.any(keep):
            out[i] = float(np.nanmean(y[keep]))
    return out


def binned_median(t: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    for i in range(len(out)):
        keep = (t >= edges[i]) & (t < edges[i + 1]) & np.isfinite(y)
        if np.any(keep):
            out[i] = float(np.nanmedian(y[keep]))
    return out


def binned_movement_fraction(
    t: np.ndarray,
    velocity_cmps: np.ndarray,
    edges: np.ndarray,
    threshold_cmps: float,
) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    speed = np.abs(np.asarray(velocity_cmps, dtype=float))
    for i in range(len(out)):
        keep = (t >= edges[i]) & (t < edges[i + 1]) & np.isfinite(speed)
        if np.any(keep):
            out[i] = float(np.mean(speed[keep] >= threshold_cmps))
    return out


def spearman_text(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 4:
        return np.nan, np.nan, int(np.sum(keep))
    r, p = spearmanr(x[keep], y[keep])
    return safe_float(r), safe_float(p), int(np.sum(keep))


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray([safe_float(v) for v in p_values], dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
    if not np.any(valid):
        return q

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order] * len(p_valid) / np.arange(1, len(p_valid) + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)

    q_valid = np.empty_like(p_valid)
    q_valid[order] = ranked
    q[valid] = q_valid
    return q


def add_fdr_p_values(metrics: dict, p_keys: list[str]) -> None:
    q_vals = bh_fdr([metrics.get(key) for key in p_keys])
    n_tests = int(np.sum(np.isfinite(q_vals)))
    for key, q in zip(p_keys, q_vals):
        metrics[f"{key}_fdr"] = safe_float(q)
    metrics["p_correction"] = "BH-FDR"
    metrics["p_correction_n_tests"] = n_tests


def phases_to_plv(phases: np.ndarray) -> tuple[float, float]:
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if len(phases) == 0:
        return np.nan, np.nan
    z = np.mean(np.exp(1j * phases))
    return safe_float(np.abs(z)), safe_float(np.angle(z))


def trial_plv(train_trial: dict | None) -> tuple[float, np.ndarray]:
    if not isinstance(train_trial, dict):
        return np.nan, np.array([], dtype=float)
    plv = safe_float(train_trial.get("metrics", {}).get("plv"))
    section = train_trial.get("plv", {})
    if isinstance(section, dict):
        if not np.isfinite(plv):
            plv = safe_float(section.get("plv"))
        phases = np.asarray(section.get("phase_pulses_rad", section.get("phase_samples_rad", [])), dtype=float)
        phases = phases[np.isfinite(phases)]
    else:
        phases = np.array([], dtype=float)
    return plv, phases


def spta_peak_latency_from_segment(first_segment: dict | None) -> tuple[float, float]:
    if not isinstance(first_segment, dict):
        return np.nan, np.nan
    t_rel = np.asarray(first_segment.get("t_rel_s", []), dtype=float)
    y = np.asarray(first_segment.get("signal", []), dtype=float)
    if len(t_rel) == 0 or y.shape != t_rel.shape:
        return np.nan, np.nan
    start_s, end_s = SPTA_PEAK_WINDOW_S
    keep = (t_rel >= float(start_s)) & (t_rel <= float(end_s)) & np.isfinite(y)
    if not np.any(keep):
        return np.nan, np.nan
    t_win = t_rel[keep]
    y_win = y[keep]
    n_top = min(2, len(y_win))
    top_idx = np.argsort(y_win)[-n_top:]
    return safe_float(np.nanmean(t_win[top_idx]) * 1000.0), safe_float(np.nanmax(y_win[top_idx]))


def mpta_peak_latency_from_trial(train_trial: dict | None) -> tuple[float, float]:
    if not isinstance(train_trial, dict):
        return np.nan, np.nan
    t_rel = np.asarray(train_trial.get("t_rel_s", []), dtype=float)
    y = np.asarray(train_trial.get("pta_mean", []), dtype=float)
    if len(t_rel) == 0 or y.shape != t_rel.shape:
        return np.nan, np.nan
    keep = (t_rel >= 0.0) & np.isfinite(y)
    if not np.any(keep):
        return np.nan, np.nan
    idx_all = np.where(keep)[0]
    idx = idx_all[int(np.nanargmax(y[idx_all]))]
    return safe_float(t_rel[idx] * 1000.0), safe_float(y[idx])


def pta_peak_amp(train_trial: dict | None) -> float:
    if not isinstance(train_trial, dict):
        return np.nan
    metric_amp = safe_float(train_trial.get("metrics", {}).get("peak_1_amplitude"))
    if np.isfinite(metric_amp):
        return metric_amp

    t_rel = np.asarray(train_trial.get("t_rel_s", []), dtype=float)
    y = np.asarray(train_trial.get("pta_mean", []), dtype=float)
    f_stim = safe_float(train_trial.get("f_stim_hz"))
    if len(t_rel) == 0 or y.shape != t_rel.shape or not np.isfinite(f_stim) or f_stim <= 0:
        return np.nan
    keep = (t_rel >= 0) & (t_rel <= 1.0 / f_stim) & np.isfinite(y)
    return float(np.nanmax(y[keep])) if np.any(keep) else np.nan


def analyze_trial(
    name: str,
    img_trial: dict,
    eph_trial: dict,
    train_trial: dict | None,
    first_segment: dict | None = None,
) -> dict | None:
    x = choose_signal(img_trial)
    t = np.asarray(img_trial.get("t", []), dtype=float)
    if x is None or len(t) < 4:
        return None
    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]

    pulse_times = np.asarray(eph_trial.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < 2:
        return None

    f_stim = estimate_stim_freq(pulse_times)
    t_end = float(ANALYSIS_END_S) if ANALYSIS_END_S is not None else float(pulse_times[-1])
    t_start = float(ANALYSIS_START_S)

    amp = stim_band_hilbert_amp(t, x, f_stim)
    stim_base_ratio, base_amp, stim_amp = amp_baseline_stim_ratio(t, amp, t_start, t_end)
    entrainment = amp / base_amp if np.isfinite(base_amp) and base_amp > 1e-12 else amp * np.nan

    if ANALYZE_THETA_BAND:
        theta_amp_trace = fixed_band_hilbert_amp(t, x, THETA_BAND_HZ)
        theta_ratio, theta_base_amp, theta_stim_amp = amp_baseline_stim_ratio(
            t, theta_amp_trace, t_start, t_end
        )
    else:
        theta_ratio = np.nan
        theta_base_amp = np.nan
        theta_stim_amp = np.nan

    x_baseline = x[(t < 0) & np.isfinite(x)]
    x_baseline_median = float(np.nanmedian(x_baseline)) if len(x_baseline) else 0.0
    x_rel = x - x_baseline_median
    early_start_s, early_end_s = VM_EARLY_WINDOW_S
    early_mask = (t >= float(early_start_s)) & (t <= float(early_end_s))
    early = x_rel[early_mask & np.isfinite(x_rel)]
    early_abs = x[early_mask & np.isfinite(x)]
    late_start_s = max(float(t_start), float(t_end) - float(VM_LATE_WINDOW_SEC))
    late_mask = (t >= late_start_s) & (t <= float(t_end))
    late = x_rel[late_mask & np.isfinite(x_rel)]
    late_abs = x[late_mask & np.isfinite(x)]

    edges = np.arange(t_start, t_end + float(BIN_SEC), float(BIN_SEC), dtype=float)
    if len(edges) < 3:
        return None
    centers = edges[:-1] + 0.5 * np.diff(edges)

    vel_t = np.asarray(eph_trial.get("vel_bin_t_s", []), dtype=float)
    vel = np.asarray(eph_trial.get("vel_bin_cmps", []), dtype=float)

    ent_bin = binned_mean(t, entrainment, edges)
    depol_bin = binned_mean(t, x_rel, edges)
    speed_bin = binned_mean(vel_t, np.abs(vel), edges)
    move_frac_bin = binned_movement_fraction(vel_t, vel, edges, MOVEMENT_THRESHOLD_CM_S)

    moving_bins = move_frac_bin >= MOVING_BIN_FRACTION_CUTOFF
    still_bins = move_frac_bin < MOVING_BIN_FRACTION_CUTOFF
    ent_moving = ent_bin[moving_bins & np.isfinite(ent_bin)]
    ent_still = ent_bin[still_bins & np.isfinite(ent_bin)]
    if len(ent_moving) >= 2 and len(ent_still) >= 2:
        movement_p = safe_float(mannwhitneyu(ent_moving, ent_still, alternative="two-sided").pvalue)
    else:
        movement_p = np.nan

    plv, plv_phases = trial_plv(train_trial)
    spta_lat_ms, spta_amp = spta_peak_latency_from_segment(first_segment)
    mpta_lat_ms, mpta_amp = mpta_peak_latency_from_trial(train_trial)

    return {
        "trial": name,
        "f_stim_hz": f_stim,
        "t": t,
        "x_rel": x_rel,
        "entrainment": entrainment,
        "vel_t": vel_t,
        "speed": np.abs(vel),
        "bin_centers": centers,
        "ent_bin": ent_bin,
        "depol_bin": depol_bin,
        "speed_bin": speed_bin,
        "move_frac_bin": move_frac_bin,
        "mean_ent": safe_float(np.nanmedian(ent_bin)),
        "hilbert_amp": safe_float(stim_base_ratio),
        "hilbert_baseline_amp": safe_float(base_amp),
        "hilbert_stim_amp": safe_float(stim_amp),
        "hilbert_stim_baseline_ratio": safe_float(stim_base_ratio),
        "theta_amp": safe_float(theta_ratio),
        "theta_baseline_amp": safe_float(theta_base_amp),
        "theta_stim_amp": safe_float(theta_stim_amp),
        "theta_stim_baseline_ratio": safe_float(theta_ratio),
        "mean_depol": safe_float(np.nanmedian(depol_bin)),
        "vm_early": safe_float(np.nanmedian(early)) if len(early) else np.nan,
        "vm_late": safe_float(np.nanmedian(late)) if len(late) else np.nan,
        "vm_baseline_abs": safe_float(x_baseline_median),
        "vm_early_abs": safe_float(np.nanmedian(early_abs)) if len(early_abs) else np.nan,
        "vm_late_abs": safe_float(np.nanmedian(late_abs)) if len(late_abs) else np.nan,
        "mean_speed": safe_float(np.nanmedian(speed_bin)),
        "movement_fraction": safe_float(np.nanmean(move_frac_bin)),
        "plv": safe_float(plv),
        "plv_phase_samples_rad": plv_phases,
        "stim_end_s": safe_float(t_end),
        "spta_lat": safe_float(spta_lat_ms),
        "spta_amp": safe_float(spta_amp),
        "mpta_lat": safe_float(mpta_lat_ms),
        "mpta_amp": safe_float(mpta_amp),
        "ent_moving": safe_float(np.nanmedian(ent_moving)) if len(ent_moving) else np.nan,
        "ent_still": safe_float(np.nanmedian(ent_still)) if len(ent_still) else np.nan,
        "movement_p": movement_p,
        "mpta_peak_amp": pta_peak_amp(train_trial),
    }


def print_corr(label: str, x: np.ndarray, y: np.ndarray) -> None:
    r, p, n = spearman_text(x, y)
    print(f"{label}: Spearman r={r:.3f}, p={p:.4g}, n={n}")


def compact_trial_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        rows.append({
            "mouse": r.get("mouse", MOUSE_NAME),
            "date": r.get("date", SINGLE_DATE),
            "block": r.get("block", SINGLE_BLOCK),
            "trial": r["trial"],
            "f_stim_hz": safe_float(r["f_stim_hz"]),
            "hilbert_amp": safe_float(r["hilbert_amp"]),
            "hilbert_amp_median": safe_float(r["hilbert_amp"]),
            "hilbert_baseline_amp": safe_float(r["hilbert_baseline_amp"]),
            "hilbert_stim_amp": safe_float(r["hilbert_stim_amp"]),
            "hilbert_stim_baseline_ratio": safe_float(r["hilbert_stim_baseline_ratio"]),
            "theta_amp": safe_float(r.get("theta_amp")),
            "theta_baseline_amp": safe_float(r.get("theta_baseline_amp")),
            "theta_stim_amp": safe_float(r.get("theta_stim_amp")),
            "theta_stim_baseline_ratio": safe_float(r.get("theta_stim_baseline_ratio")),
            "vm_median": safe_float(r["mean_depol"]),
            "vm_early": safe_float(r["vm_early"]),
            "vm_late": safe_float(r["vm_late"]),
            "vm_baseline_abs": safe_float(r["vm_baseline_abs"]),
            "vm_early_abs": safe_float(r["vm_early_abs"]),
            "vm_late_abs": safe_float(r["vm_late_abs"]),
            "velocity_median_cmps": safe_float(r["mean_speed"]),
            "movement_fraction": safe_float(r["movement_fraction"]),
            "plv": safe_float(r["plv"]),
            "spta_lat": safe_float(r["spta_lat"]),
            "spta_amp": safe_float(r["spta_amp"]),
            "mpta_lat": safe_float(r["mpta_lat"]),
            "mpta_amp": safe_float(r["mpta_amp"]),
            "mpta_peak_amp": safe_float(r["mpta_peak_amp"]),
        })
    return rows


def finite_values(values) -> np.ndarray:
    arr = np.asarray([safe_float(v) for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def latency_jitter_ms(values) -> float:
    vals = finite_values(values)
    return safe_float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else np.nan


def pooled_plv_from_results(results: list[dict]) -> tuple[float, int]:
    phase_sets = [
        np.asarray(r.get("plv_phase_samples_rad", []), dtype=float)
        for r in results
    ]
    phase_sets = [ph[np.isfinite(ph)] for ph in phase_sets if len(ph)]
    if not phase_sets:
        return np.nan, 0
    phases = np.concatenate(phase_sets)
    plv, _pref = phases_to_plv(phases)
    return plv, int(len(phases))


def paired_wilcoxon_summary(a, b) -> tuple[float, float, int]:
    arr_a = np.asarray([safe_float(v) for v in a], dtype=float)
    arr_b = np.asarray([safe_float(v) for v in b], dtype=float)
    keep = np.isfinite(arr_a) & np.isfinite(arr_b)
    if int(np.sum(keep)) < 2:
        return np.nan, np.nan, int(np.sum(keep))
    try:
        stat = wilcoxon(arr_a[keep], arr_b[keep], alternative="two-sided")
        return safe_float(stat.statistic), safe_float(stat.pvalue), int(np.sum(keep))
    except ValueError:
        return np.nan, np.nan, int(np.sum(keep))


def averaged_trace_vm_metrics(results: list[dict]) -> dict:
    traces = []
    dts = []
    stim_ends = []
    for r in results:
        t = np.asarray(r.get("t", []), dtype=float)
        x = np.asarray(r.get("x_rel", []), dtype=float)
        if len(t) < 2 or x.shape != t.shape:
            continue
        keep = np.isfinite(t) & np.isfinite(x)
        if int(np.sum(keep)) < 2:
            continue
        t = t[keep]
        x = x[keep]
        order = np.argsort(t)
        t = t[order]
        x = x[order]
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            dts.append(float(np.nanmedian(dt)))
        stim_end = safe_float(r.get("stim_end_s"))
        if np.isfinite(stim_end):
            stim_ends.append(stim_end)
        traces.append((t, x))

    if not traces or not dts:
        return {
            "vm_early": np.nan,
            "vm_late": np.nan,
            "vm_avg_trace_n_trials": 0,
            "vm_avg_trace_stim_end_s": np.nan,
        }

    dt = float(np.nanmedian(dts))
    stim_end = float(np.nanmedian(stim_ends)) if stim_ends else np.nan

    def window_median(lo: float, hi: float) -> float:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or dt <= 0:
            return np.nan
        grid = np.arange(float(lo), float(hi) + 0.5 * dt, dt, dtype=float)
        if len(grid) == 0:
            return np.nan
        curves = []
        for t, x in traces:
            y = np.interp(grid, t, x, left=np.nan, right=np.nan)
            curves.append(y)
        with np.errstate(invalid="ignore"):
            mean_trace = np.nanmean(np.vstack(curves), axis=0)
        mean_trace = mean_trace[np.isfinite(mean_trace)]
        return safe_float(np.nanmedian(mean_trace)) if len(mean_trace) else np.nan

    early_start_s, early_end_s = VM_EARLY_WINDOW_S
    late_start_s = max(float(ANALYSIS_START_S), stim_end - float(VM_LATE_WINDOW_SEC)) if np.isfinite(stim_end) else np.nan
    return {
        "vm_early": window_median(float(early_start_s), float(early_end_s)),
        "vm_late": window_median(late_start_s, stim_end),
        "vm_avg_trace_n_trials": int(len(traces)),
        "vm_avg_trace_stim_end_s": safe_float(stim_end),
    }


def compute_block_metrics(results: list[dict]) -> dict:
    hilbert_amp = np.asarray([r["hilbert_amp"] for r in results], dtype=float)
    hilbert_base = np.asarray([r["hilbert_baseline_amp"] for r in results], dtype=float)
    hilbert_stim = np.asarray([r["hilbert_stim_amp"] for r in results], dtype=float)
    hilbert_ratio = np.asarray([r["hilbert_stim_baseline_ratio"] for r in results], dtype=float)
    theta_amp = np.asarray([safe_float(r.get("theta_amp")) for r in results], dtype=float)
    theta_base = np.asarray([safe_float(r.get("theta_baseline_amp")) for r in results], dtype=float)
    theta_stim = np.asarray([safe_float(r.get("theta_stim_amp")) for r in results], dtype=float)
    mean_depol = np.asarray([r["mean_depol"] for r in results], dtype=float)
    vm_early = np.asarray([r["vm_early"] for r in results], dtype=float)
    vm_late = np.asarray([r["vm_late"] for r in results], dtype=float)
    vm_baseline_abs = np.asarray([r["vm_baseline_abs"] for r in results], dtype=float)
    vm_early_abs = np.asarray([r["vm_early_abs"] for r in results], dtype=float)
    vm_late_abs = np.asarray([r["vm_late_abs"] for r in results], dtype=float)
    mean_speed = np.asarray([r["mean_speed"] for r in results], dtype=float)
    movement_fraction = np.asarray([r["movement_fraction"] for r in results], dtype=float)
    plv = np.asarray([r["plv"] for r in results], dtype=float)
    spta_lat = np.asarray([r["spta_lat"] for r in results], dtype=float)
    mpta_lat = np.asarray([r["mpta_lat"] for r in results], dtype=float)
    mpta_amp = np.asarray([r["mpta_amp"] for r in results], dtype=float)
    plv_pooled, n_plv_pooled = pooled_plv_from_results(results)
    avg_vm = averaged_trace_vm_metrics(results)

    metrics = {
        "n_trials": int(len(results)),
        "hilbert_amp": safe_float(np.nanmedian(hilbert_amp)),
        "hilbert_amp_median": safe_float(np.nanmedian(hilbert_amp)),
        "hilbert_baseline_amp_median": safe_float(np.nanmedian(hilbert_base)),
        "hilbert_stim_amp_median": safe_float(np.nanmedian(hilbert_stim)),
        "hilbert_stim_baseline_ratio_median": safe_float(np.nanmedian(hilbert_ratio)),
        "hilbert_stim_baseline_percent_change_median": safe_float((np.nanmedian(hilbert_ratio) - 1.0) * 100.0),
        "theta_amp": safe_float(np.nanmedian(theta_amp)),
        "theta_amp_median": safe_float(np.nanmedian(theta_amp)),
        "theta_baseline_amp_median": safe_float(np.nanmedian(theta_base)),
        "theta_stim_amp_median": safe_float(np.nanmedian(theta_stim)),
        "plv": safe_float(np.nanmedian(plv)),
        "plv_median": safe_float(np.nanmedian(plv)),
        "plv_pooled": safe_float(plv_pooled),
        "n_plv_pooled_phases": int(n_plv_pooled),
        "spta_lat": safe_float(np.nanmedian(spta_lat)),
        "spta_lat_median": safe_float(np.nanmedian(spta_lat)),
        "spta_jit": latency_jitter_ms(spta_lat),
        "spta_n_latency_trials": int(len(finite_values(spta_lat))),
        "mpta_lat": safe_float(np.nanmedian(mpta_lat)),
        "mpta_lat_median": safe_float(np.nanmedian(mpta_lat)),
        "mpta_amp": safe_float(np.nanmedian(mpta_amp)),
        "mpta_amp_median": safe_float(np.nanmedian(mpta_amp)),
        "mpta_jit": latency_jitter_ms(mpta_lat),
        "mpta_n_latency_trials": int(len(finite_values(mpta_lat))),
        "vm_median": safe_float(np.nanmedian(mean_depol)),
        "vm_early": safe_float(avg_vm["vm_early"]),
        "vm_late": safe_float(avg_vm["vm_late"]),
        "vm_early_avg_trace": safe_float(avg_vm["vm_early"]),
        "vm_late_avg_trace": safe_float(avg_vm["vm_late"]),
        "vm_avg_trace_n_trials": int(avg_vm["vm_avg_trace_n_trials"]),
        "vm_avg_trace_stim_end_s": safe_float(avg_vm["vm_avg_trace_stim_end_s"]),
        "vm_early_trial_median": safe_float(np.nanmedian(vm_early)),
        "vm_late_trial_median": safe_float(np.nanmedian(vm_late)),
        "velocity_median_cmps": safe_float(np.nanmedian(mean_speed)),
        "movement_fraction_median": safe_float(np.nanmedian(movement_fraction)),
    }

    keep_h = np.isfinite(hilbert_base) & np.isfinite(hilbert_stim)
    metrics["n_hilbert_baseline_vs_stim"] = int(np.sum(keep_h))
    if int(np.sum(keep_h)) >= 2:
        try:
            stat = wilcoxon(hilbert_stim[keep_h], hilbert_base[keep_h], alternative="two-sided")
            metrics["hilbert_baseline_vs_stim_w"] = safe_float(stat.statistic)
            metrics["hilbert_baseline_vs_stim_p"] = safe_float(stat.pvalue)
        except ValueError:
            metrics["hilbert_baseline_vs_stim_w"] = np.nan
            metrics["hilbert_baseline_vs_stim_p"] = np.nan
    else:
        metrics["hilbert_baseline_vs_stim_w"] = np.nan
        metrics["hilbert_baseline_vs_stim_p"] = np.nan

    early_w, early_p, early_n = paired_wilcoxon_summary(vm_early_abs, vm_baseline_abs)
    late_w, late_p, late_n = paired_wilcoxon_summary(vm_late_abs, vm_baseline_abs)
    metrics.update({
        "vm_early_vs_baseline_w": early_w,
        "vm_early_vs_baseline_p": early_p,
        "n_vm_early_vs_baseline": early_n,
        "vm_late_vs_baseline_w": late_w,
        "vm_late_vs_baseline_p": late_p,
        "n_vm_late_vs_baseline": late_n,
    })

    if ANALYZE_VM_VELOCITY:
        r, p, n = spearman_text(mean_depol, mean_speed)
        metrics.update({"r_vm_velocity": r, "p_vm_velocity": p, "n_vm_velocity": n})
    if ANALYZE_HILBERT_VELOCITY:
        r, p, n = spearman_text(hilbert_amp, mean_speed)
        metrics.update({"r_hilbert_velocity": r, "p_hilbert_velocity": p, "n_hilbert_velocity": n})
    if ANALYZE_HILBERT_VM:
        r, p, n = spearman_text(hilbert_amp, mean_depol)
        metrics.update({"r_hilbert_vm": r, "p_hilbert_vm": p, "n_hilbert_vm": n})
    if ANALYZE_THETA_BAND:
        r, p, n = spearman_text(theta_amp, hilbert_amp)
        metrics.update({"r_theta_hilbert": r, "p_theta_hilbert": p, "n_theta_hilbert": n})
        r, p, n = spearman_text(theta_amp, mean_speed)
        metrics.update({"r_theta_velocity": r, "p_theta_velocity": p, "n_theta_velocity": n})
        r, p, n = spearman_text(theta_amp, mean_depol)
        metrics.update({"r_theta_vm": r, "p_theta_vm": p, "n_theta_vm": n})
    add_fdr_p_values(
        metrics,
        [
            key
            for key in [
                "p_vm_velocity",
                "p_hilbert_velocity",
                "p_hilbert_vm",
                "p_theta_hilbert",
                "p_theta_velocity",
                "p_theta_vm",
            ]
            if key in metrics
        ],
    )
    return metrics


def save_analysis(metrics: dict, trial_rows: list[dict]) -> None:
    out = {
        "mouse": MOUSE_NAME,
        "date": SINGLE_DATE,
        "block": SINGLE_BLOCK,
        "settings": {
            "signal_mode": SIGNAL_MODE,
            "bin_sec": float(BIN_SEC),
            "hilbert_half_band_hz": float(HILBERT_HALF_BAND_HZ),
            "analyze_theta_band": bool(ANALYZE_THETA_BAND),
            "theta_band_hz": [float(v) for v in THETA_BAND_HZ],
            "baseline_end_s": float(BASELINE_END_S),
            "analysis_start_s": float(ANALYSIS_START_S),
            "analysis_end_s": ANALYSIS_END_S,
            "movement_threshold_cm_s": float(MOVEMENT_THRESHOLD_CM_S),
            "spta_peak_window_s": [float(v) for v in SPTA_PEAK_WINDOW_S],
            "vm_early_window_s": [float(v) for v in VM_EARLY_WINDOW_S],
            "vm_late_window_sec": float(VM_LATE_WINDOW_SEC),
        },
        "metrics": metrics,
        "trials": trial_rows,
    }
    out_path = entrainment_output_path()
    with out_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {out_path}")


def metric_title(metrics: dict, r_key: str, p_key: str) -> str:
    r = safe_float(metrics.get(r_key))
    p = safe_float(metrics.get(p_key))
    q = safe_float(metrics.get(f"{p_key}_fdr"))
    if np.isfinite(r) and np.isfinite(p):
        if np.isfinite(q):
            return f"r={r:.3f}, p={p:.4g}, q={q:.4g}"
        return f"r={r:.3f}, p={p:.4g}"
    return "not enough trials"


def p_with_fdr_text(metrics: dict, p_key: str) -> str:
    p = safe_float(metrics.get(p_key))
    q = safe_float(metrics.get(f"{p_key}_fdr"))
    if np.isfinite(p) and np.isfinite(q):
        return f"p={p:.4g}, q={q:.4g}"
    if np.isfinite(p):
        return f"p={p:.4g}"
    return "p=?"


def format_median_iqr(values, digits: int = 3) -> str:
    vals = finite_values(values)
    if len(vals) == 0:
        return "n=0"
    q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
    return f"median={med:.{digits}g}, IQR={q1:.{digits}g}-{q3:.{digits}g}, n={len(vals)}"


def print_selected_metric_summary(metrics: dict, trial_rows: list[dict]) -> None:
    if not PRINT_BLOCK_TRIAL_METRICS:
        return
    requested = selection_values(BLOCK_TRIAL_METRICS)
    if not requested:
        return

    row_metric_keys = {
        "hilbert_amp": "hilbert_amp",
        "theta_amp": "theta_amp",
        "plv": "plv",
        "spta_lat": "spta_lat",
        "mpta_lat": "mpta_lat",
        "mpta_amp": "mpta_amp",
    }
    jitter_keys = {
        "spta_jit": ("spta_jit", "spta_n_latency_trials"),
        "mpta_jit": ("mpta_jit", "mpta_n_latency_trials"),
    }
    corr_keys = {
        "r_vm_velocity": ("r_vm_velocity", "p_vm_velocity", "n_vm_velocity"),
        "r_hilbert_velocity": ("r_hilbert_velocity", "p_hilbert_velocity", "n_hilbert_velocity"),
        "r_hilbert_vm": ("r_hilbert_vm", "p_hilbert_vm", "n_hilbert_vm"),
        "r_theta_hilbert": ("r_theta_hilbert", "p_theta_hilbert", "n_theta_hilbert"),
        "r_theta_velocity": ("r_theta_velocity", "p_theta_velocity", "n_theta_velocity"),
        "r_theta_vm": ("r_theta_vm", "p_theta_vm", "n_theta_vm"),
    }

    print("\nSelected trial/block metrics:")
    for metric in requested:
        key = metric.strip()
        if key in row_metric_keys:
            row_key = row_metric_keys[key]
            vals = [row.get(row_key) for row in trial_rows]
            extra = ""
            if key == "hilbert_amp":
                extra = f", baseline vs stim {p_with_fdr_text(metrics, 'hilbert_baseline_vs_stim_p')}"
            print(f"  {key}: {format_median_iqr(vals)}{extra}")
        elif key == "plv_pooled":
            print(
                f"  plv_pooled: {fmt_col(metrics.get('plv_pooled'))}, "
                f"n_phases={int(metrics.get('n_plv_pooled_phases', 0))}"
            )
        elif key == "vm_early":
            print(
                f"  vm_early: {fmt_col(metrics.get('vm_early'))} "
                f"(averaged trace, {VM_EARLY_WINDOW_S[0]:g}-{VM_EARLY_WINDOW_S[1]:g} s), "
                f"vs baseline {p_with_fdr_text(metrics, 'vm_early_vs_baseline_p')}, "
                f"n={int(metrics.get('n_vm_early_vs_baseline', 0))}"
            )
        elif key == "vm_late":
            print(
                f"  vm_late: {fmt_col(metrics.get('vm_late'))} "
                f"(averaged trace, last {VM_LATE_WINDOW_SEC:g} s), "
                f"vs baseline {p_with_fdr_text(metrics, 'vm_late_vs_baseline_p')}, "
                f"n={int(metrics.get('n_vm_late_vs_baseline', 0))}"
            )
        elif key in jitter_keys:
            value_key, n_key = jitter_keys[key]
            print(f"  {key}: {fmt_col(metrics.get(value_key))} ms, n={int(metrics.get(n_key, 0))}")
        elif key in corr_keys:
            r_key, p_key, n_key = corr_keys[key]
            print(
                f"  {key}: r={fmt_col(metrics.get(r_key))}, "
                f"{p_with_fdr_text(metrics, p_key)}, n={int(metrics.get(n_key, 0))}"
            )
        else:
            value = safe_float(metrics.get(key))
            print(f"  {key}: {fmt_col(value)}")


def add_panel_letters(axes, start: int = 0) -> None:
    for i, axis in enumerate(np.ravel(axes)):
        axis.text(
            -0.16,
            1.08,
            f"{chr(97 + start + i)}",
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def velocity_split_metrics(trial_rows: list[dict]) -> dict:
    velocity = np.asarray([safe_float(r["velocity_median_cmps"]) for r in trial_rows], dtype=float)
    vm = np.asarray([safe_float(r["vm_median"]) for r in trial_rows], dtype=float)
    hilbert_amp = np.asarray([safe_float(r["hilbert_amp"]) for r in trial_rows], dtype=float)
    theta_amp = np.asarray([safe_float(r.get("theta_amp")) for r in trial_rows], dtype=float)
    valid = np.isfinite(velocity) & np.isfinite(vm) & np.isfinite(hilbert_amp)
    velocity = velocity[valid]
    vm = vm[valid]
    hilbert_amp = hilbert_amp[valid]
    theta_amp = theta_amp[valid]

    fdr_p_keys = [
        "vm_high_vs_low_p",
        "hilbert_high_vs_low_p",
        "p_vm_velocity",
        "p_hilbert_velocity",
        "p_hilbert_vm",
    ]
    if ANALYZE_THETA_BAND:
        fdr_p_keys.extend(["p_theta_hilbert", "p_theta_velocity", "p_theta_vm"])

    out = {
        "n_trials": int(len(velocity)),
        "velocity_low_percentile": float(LOW_VELOCITY_PERCENTILE),
        "velocity_high_percentile": float(HIGH_VELOCITY_PERCENTILE),
        "velocity_low_cut_cmps": np.nan,
        "velocity_high_cut_cmps": np.nan,
        "n_low_velocity_trials": 0,
        "n_high_velocity_trials": 0,
        "vm_low_median": np.nan,
        "vm_high_median": np.nan,
        "vm_high_minus_low": np.nan,
        "vm_high_vs_low_u": np.nan,
        "vm_high_vs_low_p": np.nan,
        "hilbert_low_median": np.nan,
        "hilbert_high_median": np.nan,
        "hilbert_high_minus_low": np.nan,
        "hilbert_high_vs_low_u": np.nan,
        "hilbert_high_vs_low_p": np.nan,
        "r_vm_velocity": np.nan,
        "p_vm_velocity": np.nan,
        "n_vm_velocity": int(len(velocity)),
        "r_hilbert_velocity": np.nan,
        "p_hilbert_velocity": np.nan,
        "n_hilbert_velocity": int(len(velocity)),
        "r_hilbert_vm": np.nan,
        "p_hilbert_vm": np.nan,
        "n_hilbert_vm": int(len(velocity)),
        "r_theta_hilbert": np.nan,
        "p_theta_hilbert": np.nan,
        "n_theta_hilbert": int(np.sum(np.isfinite(theta_amp) & np.isfinite(hilbert_amp))),
        "r_theta_velocity": np.nan,
        "p_theta_velocity": np.nan,
        "n_theta_velocity": int(np.sum(np.isfinite(theta_amp) & np.isfinite(velocity))),
        "r_theta_vm": np.nan,
        "p_theta_vm": np.nan,
        "n_theta_vm": int(np.sum(np.isfinite(theta_amp) & np.isfinite(vm))),
    }
    if len(velocity) < 4:
        return out

    low_cut = float(np.nanpercentile(velocity, LOW_VELOCITY_PERCENTILE))
    high_cut = float(np.nanpercentile(velocity, HIGH_VELOCITY_PERCENTILE))
    low = velocity <= low_cut
    high = velocity >= high_cut
    out["velocity_low_cut_cmps"] = low_cut
    out["velocity_high_cut_cmps"] = high_cut

    r, p, _ = spearman_text(vm, velocity)
    out["r_vm_velocity"] = r
    out["p_vm_velocity"] = p
    r, p, _ = spearman_text(hilbert_amp, velocity)
    out["r_hilbert_velocity"] = r
    out["p_hilbert_velocity"] = p
    r, p, _ = spearman_text(hilbert_amp, vm)
    out["r_hilbert_vm"] = r
    out["p_hilbert_vm"] = p
    if ANALYZE_THETA_BAND:
        r, p, n = spearman_text(theta_amp, hilbert_amp)
        out["r_theta_hilbert"] = r
        out["p_theta_hilbert"] = p
        out["n_theta_hilbert"] = n
        r, p, n = spearman_text(theta_amp, velocity)
        out["r_theta_velocity"] = r
        out["p_theta_velocity"] = p
        out["n_theta_velocity"] = n
        r, p, n = spearman_text(theta_amp, vm)
        out["r_theta_vm"] = r
        out["p_theta_vm"] = p
        out["n_theta_vm"] = n

    if high_cut <= low_cut:
        add_fdr_p_values(out, fdr_p_keys)
        return out
    out["n_low_velocity_trials"] = int(np.sum(low))
    out["n_high_velocity_trials"] = int(np.sum(high))

    if np.sum(low) < MIN_TRIALS_PER_VELOCITY_GROUP or np.sum(high) < MIN_TRIALS_PER_VELOCITY_GROUP:
        add_fdr_p_values(out, fdr_p_keys)
        return out

    vm_low = vm[low]
    vm_high = vm[high]
    hilbert_low = hilbert_amp[low]
    hilbert_high = hilbert_amp[high]
    out["vm_low_median"] = safe_float(np.nanmedian(vm_low))
    out["vm_high_median"] = safe_float(np.nanmedian(vm_high))
    out["vm_high_minus_low"] = out["vm_high_median"] - out["vm_low_median"]
    out["hilbert_low_median"] = safe_float(np.nanmedian(hilbert_low))
    out["hilbert_high_median"] = safe_float(np.nanmedian(hilbert_high))
    out["hilbert_high_minus_low"] = out["hilbert_high_median"] - out["hilbert_low_median"]

    stat = mannwhitneyu(vm_high, vm_low, alternative="two-sided")
    out["vm_high_vs_low_u"] = float(stat.statistic)
    out["vm_high_vs_low_p"] = float(stat.pvalue)
    stat = mannwhitneyu(hilbert_high, hilbert_low, alternative="two-sided")
    out["hilbert_high_vs_low_u"] = float(stat.statistic)
    out["hilbert_high_vs_low_p"] = float(stat.pvalue)
    add_fdr_p_values(out, fdr_p_keys)
    return out


def group_arrays_for_plot(trial_rows: list[dict], metrics: dict):
    velocity = np.asarray([safe_float(r["velocity_median_cmps"]) for r in trial_rows], dtype=float)
    vm = np.asarray([safe_float(r["vm_median"]) for r in trial_rows], dtype=float)
    hilbert_amp = np.asarray([safe_float(r["hilbert_amp"]) for r in trial_rows], dtype=float)
    theta_amp = np.asarray([safe_float(r.get("theta_amp")) for r in trial_rows], dtype=float)
    low_cut = safe_float(metrics.get("velocity_low_cut_cmps"))
    high_cut = safe_float(metrics.get("velocity_high_cut_cmps"))
    low = np.isfinite(velocity) & np.isfinite(low_cut) & (velocity <= low_cut)
    high = np.isfinite(velocity) & np.isfinite(high_cut) & (velocity >= high_cut)
    middle = np.isfinite(velocity) & ~(low | high)
    return velocity, vm, hilbert_amp, theta_amp, low, middle, high


def selected_condition_plot_metrics() -> list[str]:
    requested = [normalize_class_label(item) for item in selection_values(CONDITION_PLOT_METRICS)]
    if not requested or "all" in requested:
        return [
            "velocity_split",
            "vm_velocity_split",
            "hilbert_velocity_split",
            "r_vm_velocity",
            "r_hilbert_velocity",
            "r_hilbert_vm",
        ]
    return requested


def plot_condition_results(trial_rows: list[dict], metrics: dict) -> None:
    velocity, vm, hilbert_amp, theta_amp, low, middle, high = group_arrays_for_plot(trial_rows, metrics)
    vm_plot = gevi_display(vm)
    requested = selected_condition_plot_metrics()
    panels = []

    def add_panel(key: str, title: str, draw_func) -> None:
        if key in requested:
            panels.append((title, draw_func))

    def draw_velocity_split(axis) -> None:
        axis.hist(velocity[np.isfinite(velocity)], bins=20, color="0.7", edgecolor="white")
        if np.isfinite(metrics.get("velocity_low_cut_cmps", np.nan)):
            axis.axvline(metrics["velocity_low_cut_cmps"], color="tab:blue", ls="--", label="low cut")
        if np.isfinite(metrics.get("velocity_high_cut_cmps", np.nan)):
            axis.axvline(metrics["velocity_high_cut_cmps"], color="tab:red", ls="--", label="high cut")
        axis.set_xlabel("trial velocity (cm/s)")
        axis.set_ylabel("n trials")
        axis.legend(loc="best", fontsize=8)

    def draw_vm_velocity_split(axis) -> None:
        axis.boxplot([vm_plot[low], vm_plot[high]], labels=["low vel", "high vel"], showfliers=False)
        axis.scatter(np.ones(np.sum(low)), vm_plot[low], s=18, alpha=0.55, color="tab:blue")
        axis.scatter(np.ones(np.sum(high)) * 2, vm_plot[high], s=18, alpha=0.55, color="tab:red")
        axis.set_ylabel(GEVI_YLABEL)

    def draw_hilbert_velocity_split(axis) -> None:
        axis.boxplot([hilbert_amp[low], hilbert_amp[high]], labels=["low vel", "high vel"], showfliers=False)
        axis.scatter(np.ones(np.sum(low)), hilbert_amp[low], s=18, alpha=0.55, color="tab:blue")
        axis.scatter(np.ones(np.sum(high)) * 2, hilbert_amp[high], s=18, alpha=0.55, color="tab:red")
        axis.set_ylabel("trial Hilbert amp / baseline")

    def draw_vm_velocity(axis) -> None:
        axis.scatter(velocity[middle], vm_plot[middle], s=30, alpha=0.35, color="0.5", label="middle")
        axis.scatter(velocity[low], vm_plot[low], s=45, alpha=0.8, color="tab:blue", label="low")
        axis.scatter(velocity[high], vm_plot[high], s=45, alpha=0.8, color="tab:red", label="high")
        axis.set_xlabel("trial velocity (cm/s)")
        axis.set_ylabel(GEVI_YLABEL)
        axis.legend(loc="best", fontsize=8)

    def draw_hilbert_velocity(axis) -> None:
        groups = [hilbert_amp[low], hilbert_amp[high]]
        positions = np.arange(1, 3, dtype=float)
        axis.boxplot(groups, labels=["low vel", "high vel"], showfliers=False)
        axis.scatter(np.full(np.sum(low), positions[0]), hilbert_amp[low], s=18, alpha=0.55, color="tab:blue")
        axis.scatter(np.full(np.sum(high), positions[1]), hilbert_amp[high], s=18, alpha=0.55, color="tab:red")
        axis.set_ylabel("trial Hilbert amp / baseline")

    def draw_hilbert_vm(axis) -> None:
        valid_vm = np.isfinite(vm_plot) & np.isfinite(hilbert_amp)
        if int(np.sum(valid_vm)) < 4:
            axis.text(0.5, 0.5, "not enough trials", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        vm_low_cut = float(np.nanpercentile(vm_plot[valid_vm], LOW_VELOCITY_PERCENTILE))
        vm_high_cut = float(np.nanpercentile(vm_plot[valid_vm], HIGH_VELOCITY_PERCENTILE))
        vm_low = valid_vm & (vm_plot <= vm_low_cut)
        vm_high = valid_vm & (vm_plot >= vm_high_cut)
        groups = [hilbert_amp[vm_low], hilbert_amp[vm_high]]
        positions = np.arange(1, 3, dtype=float)
        axis.boxplot(groups, labels=["low Vm", "high Vm"], showfliers=False)
        axis.scatter(np.full(np.sum(vm_low), positions[0]), hilbert_amp[vm_low], s=18, alpha=0.55, color="tab:blue")
        axis.scatter(np.full(np.sum(vm_high), positions[1]), hilbert_amp[vm_high], s=18, alpha=0.55, color="tab:red")
        axis.set_ylabel("trial Hilbert amp / baseline")

    def draw_theta_hilbert(axis) -> None:
        if not ANALYZE_THETA_BAND:
            axis.text(0.5, 0.5, "set ANALYZE_THETA_BAND = True", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        valid = np.isfinite(theta_amp) & np.isfinite(hilbert_amp)
        if int(np.sum(valid)) < 4:
            axis.text(0.5, 0.5, "not enough trials", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        axis.scatter(theta_amp[valid & middle], hilbert_amp[valid & middle], s=30, alpha=0.35, color="0.5", label="middle")
        axis.scatter(theta_amp[valid & low], hilbert_amp[valid & low], s=45, alpha=0.8, color="tab:blue", label="low vel")
        axis.scatter(theta_amp[valid & high], hilbert_amp[valid & high], s=45, alpha=0.8, color="tab:red", label="high vel")
        axis.set_xlabel("theta Hilbert amp / baseline")
        axis.set_ylabel("DBS Hilbert amp / baseline")
        axis.legend(loc="best", fontsize=8)

    def draw_theta_velocity(axis) -> None:
        if not ANALYZE_THETA_BAND:
            axis.text(0.5, 0.5, "set ANALYZE_THETA_BAND = True", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        valid = np.isfinite(theta_amp) & np.isfinite(velocity)
        if int(np.sum(valid)) < 4:
            axis.text(0.5, 0.5, "not enough trials", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        axis.scatter(velocity[valid & middle], theta_amp[valid & middle], s=30, alpha=0.35, color="0.5", label="middle")
        axis.scatter(velocity[valid & low], theta_amp[valid & low], s=45, alpha=0.8, color="tab:blue", label="low")
        axis.scatter(velocity[valid & high], theta_amp[valid & high], s=45, alpha=0.8, color="tab:red", label="high")
        axis.set_xlabel("trial velocity (cm/s)")
        axis.set_ylabel("theta Hilbert amp / baseline")
        axis.legend(loc="best", fontsize=8)

    def draw_theta_vm(axis) -> None:
        if not ANALYZE_THETA_BAND:
            axis.text(0.5, 0.5, "set ANALYZE_THETA_BAND = True", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        valid = np.isfinite(theta_amp) & np.isfinite(vm_plot)
        if int(np.sum(valid)) < 4:
            axis.text(0.5, 0.5, "not enough trials", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            return
        axis.scatter(vm_plot[valid & middle], theta_amp[valid & middle], s=30, alpha=0.35, color="0.5", label="middle")
        axis.scatter(vm_plot[valid & low], theta_amp[valid & low], s=45, alpha=0.8, color="tab:blue", label="low vel")
        axis.scatter(vm_plot[valid & high], theta_amp[valid & high], s=45, alpha=0.8, color="tab:red", label="high vel")
        axis.set_xlabel(GEVI_YLABEL)
        axis.set_ylabel("theta Hilbert amp / baseline")
        axis.legend(loc="best", fontsize=8)

    add_panel("velocity_split", "Trial velocity distribution", draw_velocity_split)
    add_panel("vm_velocity_split", "Vm by velocity group", draw_vm_velocity_split)
    if "vm" in requested or "vm_median" in requested:
        panels.append(("Vm by velocity group", draw_vm_velocity_split))
    add_panel("hilbert_velocity_split", "Hilbert amplitude by velocity group", draw_hilbert_velocity_split)
    if "hilbert_amp" in requested:
        panels.append(("Hilbert amplitude by velocity group", draw_hilbert_velocity_split))
    add_panel("r_vm_velocity", "Vm vs velocity", draw_vm_velocity)
    add_panel("r_hilbert_velocity", "Hilbert amplitude vs velocity", draw_hilbert_velocity)
    add_panel("r_hilbert_vm", "Hilbert amplitude vs Vm", draw_hilbert_vm)
    add_panel("r_theta_hilbert", "Theta vs DBS Hilbert", draw_theta_hilbert)
    if "theta_band" in requested:
        panels.append(("Theta vs DBS Hilbert", draw_theta_hilbert))
    add_panel("r_theta_velocity", "Theta vs velocity", draw_theta_velocity)
    add_panel("r_theta_vm", "Theta vs Vm", draw_theta_vm)

    if not panels:
        print("No recognized CONDITION_PLOT_METRICS for the condition velocity figure.")
        return

    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows), constrained_layout=True)
    axes = np.ravel(np.asarray(axes))
    for axis, (title, draw_func) in zip(axes, panels):
        draw_func(axis)
        axis.set_title(title)
    for axis in axes[n_panels:]:
        axis.set_visible(False)

    add_panel_letters(axes[:n_panels])
    save_figure(fig, f"{condition_output_stem()}_condition_velocity")
    plt.show()


def plot_results(results: list[dict], metrics: dict) -> None:
    hilbert_amp = np.asarray([r["hilbert_amp"] for r in results], dtype=float)
    mean_depol = gevi_display([r["mean_depol"] for r in results])
    mean_speed = np.asarray([r["mean_speed"] for r in results], dtype=float)
    movement_fraction = np.asarray([r["movement_fraction"] for r in results], dtype=float)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    sc0 = ax[0].scatter(mean_speed, mean_depol, c=movement_fraction, s=70, cmap="viridis")
    ax[0].set_xlabel("trial velocity (cm/s)")
    ax[0].set_ylabel(GEVI_YLABEL)
    ax[0].set_title("Vm vs velocity")

    ax[1].scatter(mean_speed, hilbert_amp, c=movement_fraction, s=70, cmap="viridis")
    ax[1].set_xlabel("trial velocity (cm/s)")
    ax[1].set_ylabel("trial Hilbert amp / baseline")
    ax[1].set_title("Hilbert amplitude vs velocity")

    ax[2].scatter(mean_depol, hilbert_amp, c=movement_fraction, s=70, cmap="viridis")
    ax[2].set_xlabel(GEVI_YLABEL)
    ax[2].set_ylabel("trial Hilbert amp / baseline")
    ax[2].set_title("Hilbert amplitude vs Vm")

    fig.colorbar(sc0, ax=ax, label="movement fraction", shrink=0.85, pad=0.02)

    add_panel_letters(ax)
    fig.suptitle(f"{MOUSE_NAME} | {SINGLE_DATE} {SINGLE_BLOCK}")

    save_figure(fig, f"{MOUSE_NAME}_{SINGLE_DATE}_{SINGLE_BLOCK}_block_velocity")
    plt.show()


def run_current_block(show_figures: bool | None = None) -> None:
    img = load_pickle(processed_path())
    eph = load_pickle(ephys_path())
    train = load_pickle(train_pta_path()) if train_pta_path().exists() else {}
    first = load_pickle(first_pta_path()) if first_pta_path().exists() else {}

    img_trials = img.get("trials", {})
    eph_trials = eph.get("trials", {})
    train_trials = train.get("trial_results", {})
    first_segments = {seg.get("trial"): seg for seg in first.get("segments", []) if isinstance(seg, dict)}

    names = sorted(set(img_trials) & set(eph_trials))
    if ONLY_TRIAL is not None:
        names = [n for n in names if n == ONLY_TRIAL]

    results = []
    for name in names:
        result = analyze_trial(name, img_trials[name], eph_trials[name], train_trials.get(name), first_segments.get(name))
        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit("No usable trials found.")

    print(f"\n{MOUSE_NAME} | {SINGLE_DATE} {SINGLE_BLOCK} | trials={len(results)}")

    metrics = compute_block_metrics(results)
    trial_rows = compact_trial_rows(results)

    for label, r_key, p_key, n_key in [
        ("Vm vs velocity", "r_vm_velocity", "p_vm_velocity", "n_vm_velocity"),
        ("Hilbert amp vs velocity", "r_hilbert_velocity", "p_hilbert_velocity", "n_hilbert_velocity"),
        ("Hilbert amp vs Vm", "r_hilbert_vm", "p_hilbert_vm", "n_hilbert_vm"),
        ("Theta amp vs Hilbert amp", "r_theta_hilbert", "p_theta_hilbert", "n_theta_hilbert"),
        ("Theta amp vs velocity", "r_theta_velocity", "p_theta_velocity", "n_theta_velocity"),
        ("Theta amp vs Vm", "r_theta_vm", "p_theta_vm", "n_theta_vm"),
    ]:
        if r_key not in metrics:
            continue
        q_key = f"{p_key}_fdr"
        q_text = f", q={metrics[q_key]:.4g}" if np.isfinite(safe_float(metrics.get(q_key))) else ""
        print(f"{label}: r={metrics[r_key]:.3f}, p={metrics[p_key]:.4g}{q_text}, n={metrics[n_key]}")

    print(
        "Hilbert amp baseline vs stim: "
        f"baseline={fmt_col(metrics['hilbert_baseline_amp_median'])}, "
        f"stim={fmt_col(metrics['hilbert_stim_amp_median'])}, "
        f"ratio={fmt_col(metrics['hilbert_stim_baseline_ratio_median'])}, "
        f"change={fmt_col(metrics['hilbert_stim_baseline_percent_change_median'])}%, "
        f"Wilcoxon p={fmt_col(metrics['hilbert_baseline_vs_stim_p'], digits=4)}, "
        f"n={metrics['n_hilbert_baseline_vs_stim']}"
    )
    print(
        "Vm early/late vs baseline: "
        f"early={fmt_col(metrics['vm_early'])}, "
        f"Wilcoxon p={fmt_col(metrics['vm_early_vs_baseline_p'], digits=4)}, "
        f"n={metrics['n_vm_early_vs_baseline']}; "
        f"late={fmt_col(metrics['vm_late'])}, "
        f"Wilcoxon p={fmt_col(metrics['vm_late_vs_baseline_p'], digits=4)}, "
        f"n={metrics['n_vm_late_vs_baseline']}"
    )
    print_selected_metric_summary(metrics, trial_rows)

    if PRINT_TRIAL_TABLE:
        print("\nTrial overview")
        print(
            f"{'trial':<10} {'Hilbert':>9} {'PLV':>7} {'sPTA':>7} {'mPTA':>7} "
            f"{'VmEarly':>9} {'VmLate':>9} {'Vm':>10} {'velocity':>10}"
        )
        for r in trial_rows:
            print(
                f"{r['trial']:<10} "
                f"{fmt_col(r['hilbert_amp']):>9} "
                f"{fmt_col(r['plv']):>7} "
                f"{fmt_col(r['spta_lat']):>7} "
                f"{fmt_col(r['mpta_lat']):>7} "
                f"{fmt_col(r['vm_early']):>9} "
                f"{fmt_col(r['vm_late']):>9} "
                f"{fmt_col(r['vm_median']):>10} "
                f"{fmt_col(r['velocity_median_cmps']):>10}"
            )

    if SAVE_OUTPUT:
        save_analysis(metrics, trial_rows)

    print_ephys_check_summary([(MOUSE_NAME, SINGLE_DATE, SINGLE_BLOCK)])

    do_show = SHOW_FIGURES if show_figures is None else bool(show_figures)
    if do_show:
        plot_results(results, metrics)


def load_block_results(mouse: str, date: str, block: str) -> list[dict]:
    img_p = processed_path_for(mouse, date, block)
    eph_p = ephys_path_for(mouse, date, block)
    train_p = train_pta_path_for(mouse, date, block)
    first_p = first_pta_path_for(mouse, date, block)
    img = load_pickle(img_p)
    eph = load_pickle(eph_p)
    train = load_pickle(train_p) if train_p.exists() else {}
    first = load_pickle(first_p) if first_p.exists() else {}

    img_trials = img.get("trials", {})
    eph_trials = eph.get("trials", {})
    train_trials = train.get("trial_results", {})
    first_segments = {seg.get("trial"): seg for seg in first.get("segments", []) if isinstance(seg, dict)}
    names = sorted(set(img_trials) & set(eph_trials))
    out = []
    for name in names:
        result = analyze_trial(name, img_trials[name], eph_trials[name], train_trials.get(name), first_segments.get(name))
        if result is None:
            continue
        result["mouse"] = mouse
        result["date"] = date
        result["block"] = block
        out.append(result)
    return out


def condition_output_path() -> Path:
    return DATA_ANALYSIS_ROOT / "tables" / f"{condition_output_stem()}_entrain_condition.pkl"


def save_condition_analysis(metrics: dict, trial_rows: list[dict], blocks_used: list[tuple[str, str, str]]) -> None:
    out = {
        "analysis": "condition_velocity_split",
        "settings": {
            "mouse_name": MOUSE_NAME,
            "frequency_hz": CONDITION_FREQUENCY_HZ,
            "amplitude_uA": CONDITION_AMPLITUDE_UA,
            "pulse_width_us": CONDITION_PULSE_WIDTH_US,
            "exposure_ms": CONDITION_EXPOSURE_MS,
            "stimulation_time_s": CONDITION_STIMULATION_TIME_S,
            "phase": CONDITION_PHASE,
            "imaging_side": CONDITION_IMAGING_SIDE,
            "date": CONDITION_DATE,
            "block": CONDITION_BLOCK,
            "low_velocity_percentile": float(LOW_VELOCITY_PERCENTILE),
            "high_velocity_percentile": float(HIGH_VELOCITY_PERCENTILE),
            "signal_mode": SIGNAL_MODE,
            "bin_sec": float(BIN_SEC),
            "hilbert_half_band_hz": float(HILBERT_HALF_BAND_HZ),
            "analyze_theta_band": bool(ANALYZE_THETA_BAND),
            "theta_band_hz": [float(v) for v in THETA_BAND_HZ],
            "spta_peak_window_s": [float(v) for v in SPTA_PEAK_WINDOW_S],
            "vm_early_window_s": [float(v) for v in VM_EARLY_WINDOW_S],
            "vm_late_window_sec": float(VM_LATE_WINDOW_SEC),
        },
        "blocks_used": blocks_used,
        "metrics": metrics,
        "trials": trial_rows,
    }
    out_path = condition_output_path()
    with out_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {out_path}")


def baseline_bins_for_trial(
    name: str,
    img_trial: dict,
    eph_trial: dict,
    mouse: str,
    date: str,
    block: str,
) -> list[dict]:
    x = choose_signal(img_trial)
    t = np.asarray(img_trial.get("t", []), dtype=float)
    if x is None or len(t) < 4:
        return []
    n = min(len(t), len(x))
    t = t[:n]
    x = np.asarray(x[:n], dtype=float)

    vel_t = np.asarray(eph_trial.get("vel_bin_t_s", []), dtype=float)
    vel = np.asarray(eph_trial.get("vel_bin_cmps", []), dtype=float)
    if len(vel_t) < 4 or vel.shape != vel_t.shape:
        return []

    keep_x = np.isfinite(t) & np.isfinite(x)
    keep_v = np.isfinite(vel_t) & np.isfinite(vel)
    if int(np.sum(keep_x)) < 4 or int(np.sum(keep_v)) < 4:
        return []

    t = t[keep_x]
    x = x[keep_x]
    vel_t = vel_t[keep_v]
    vel = vel[keep_v]

    x_rel = centered_fractional_signal(x)
    t_start = max(float(np.nanmin(t)), float(np.nanmin(vel_t)))
    t_end = min(float(np.nanmax(t)), float(np.nanmax(vel_t)))
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start + float(BIN_SEC):
        return []

    edges = np.arange(t_start, t_end + float(BIN_SEC), float(BIN_SEC), dtype=float)
    if len(edges) < 3:
        return []
    centers = edges[:-1] + 0.5 * np.diff(edges)

    vm_bin = binned_median(t, x_rel, edges)
    speed_bin = binned_median(vel_t, np.abs(vel), edges)
    move_frac_bin = binned_movement_fraction(vel_t, vel, edges, MOVEMENT_THRESHOLD_CM_S)

    rows = []
    for i, center in enumerate(centers):
        vm = safe_float(vm_bin[i])
        speed = safe_float(speed_bin[i])
        if not np.isfinite(vm) or not np.isfinite(speed):
            continue
        rows.append({
            "mouse": mouse,
            "date": date,
            "block": block,
            "trial": name,
            "bin_index": int(i),
            "bin_center_s": safe_float(center),
            "vm_median": vm,
            "velocity_median_cmps": speed,
            "movement_fraction": safe_float(move_frac_bin[i]),
        })
    return rows


def load_baseline_block_bins(mouse: str, date: str, block: str) -> list[dict]:
    img = load_pickle(processed_path_for(mouse, date, block))
    eph = load_pickle(ephys_path_for(mouse, date, block))
    img_trials = img.get("trials", {})
    eph_trials = eph.get("trials", {})
    rows = []
    for name in sorted(set(img_trials) & set(eph_trials)):
        rows.extend(baseline_bins_for_trial(name, img_trials[name], eph_trials[name], mouse, date, block))
    return rows


def baseline_velocity_metrics(bin_rows: list[dict]) -> dict:
    velocity = np.asarray([safe_float(r["velocity_median_cmps"]) for r in bin_rows], dtype=float)
    vm = np.asarray([safe_float(r["vm_median"]) for r in bin_rows], dtype=float)
    valid = np.isfinite(velocity) & np.isfinite(vm)
    velocity = velocity[valid]
    vm = vm[valid]

    out = {
        "n_bins": int(len(velocity)),
        "n_trials": int(len({r.get("trial") for r in bin_rows})),
        "velocity_low_percentile": float(LOW_VELOCITY_PERCENTILE),
        "velocity_high_percentile": float(HIGH_VELOCITY_PERCENTILE),
        "velocity_low_cut_cmps": np.nan,
        "velocity_high_cut_cmps": np.nan,
        "velocity_split_mode": "percentile",
        "n_low_velocity_bins": 0,
        "n_high_velocity_bins": 0,
        "vm_low_median": np.nan,
        "vm_high_median": np.nan,
        "vm_high_minus_low": np.nan,
        "vm_high_vs_low_u": np.nan,
        "vm_high_vs_low_p": np.nan,
        "vm_high_minus_low_trial_median": np.nan,
        "vm_high_vs_low_trial_w": np.nan,
        "vm_high_vs_low_trial_p": np.nan,
        "n_velocity_split_trials": 0,
        "r_vm_velocity": np.nan,
        "p_vm_velocity": np.nan,
        "n_vm_velocity": int(len(velocity)),
    }
    if len(velocity) < 4:
        add_fdr_p_values(out, ["vm_high_vs_low_trial_p", "vm_high_vs_low_p", "p_vm_velocity"])
        return out

    low_cut = float(np.nanpercentile(velocity, LOW_VELOCITY_PERCENTILE))
    high_cut = float(np.nanpercentile(velocity, HIGH_VELOCITY_PERCENTILE))
    if high_cut <= low_cut:
        low = velocity <= low_cut
        high = velocity > low_cut
        out["velocity_split_mode"] = "positive_vs_zero"
    else:
        low = velocity <= low_cut
        high = velocity >= high_cut
    out["velocity_low_cut_cmps"] = low_cut
    out["velocity_high_cut_cmps"] = high_cut
    out["n_low_velocity_bins"] = int(np.sum(low))
    out["n_high_velocity_bins"] = int(np.sum(high))

    r, p, n = spearman_text(vm, velocity)
    out["r_vm_velocity"] = r
    out["p_vm_velocity"] = p
    out["n_vm_velocity"] = n

    if np.sum(low) >= MIN_TRIALS_PER_VELOCITY_GROUP and np.sum(high) >= MIN_TRIALS_PER_VELOCITY_GROUP:
        vm_low = vm[low]
        vm_high = vm[high]
        out["vm_low_median"] = safe_float(np.nanmedian(vm_low))
        out["vm_high_median"] = safe_float(np.nanmedian(vm_high))
        out["vm_high_minus_low"] = out["vm_high_median"] - out["vm_low_median"]
        stat = mannwhitneyu(vm_high, vm_low, alternative="two-sided")
        out["vm_high_vs_low_u"] = safe_float(stat.statistic)
        out["vm_high_vs_low_p"] = safe_float(stat.pvalue)

    trial_diffs = []
    for key in sorted({(r.get("mouse"), r.get("date"), r.get("block"), r.get("trial")) for r in bin_rows}):
        trial_rows = [
            r for r in bin_rows
            if (r.get("mouse"), r.get("date"), r.get("block"), r.get("trial")) == key
        ]
        trial_velocity = np.asarray([safe_float(r["velocity_median_cmps"]) for r in trial_rows], dtype=float)
        trial_vm = np.asarray([safe_float(r["vm_median"]) for r in trial_rows], dtype=float)
        valid_trial = np.isfinite(trial_velocity) & np.isfinite(trial_vm)
        trial_velocity = trial_velocity[valid_trial]
        trial_vm = trial_vm[valid_trial]
        if out.get("velocity_split_mode") == "positive_vs_zero":
            trial_low = trial_velocity <= low_cut
            trial_high = trial_velocity > low_cut
        else:
            trial_low = trial_velocity <= low_cut
            trial_high = trial_velocity >= high_cut
        if np.sum(trial_low) < 1 or np.sum(trial_high) < 1:
            continue
        trial_diffs.append(float(np.nanmedian(trial_vm[trial_high]) - np.nanmedian(trial_vm[trial_low])))
    trial_diffs = finite_values(trial_diffs)
    out["n_velocity_split_trials"] = int(len(trial_diffs))
    if len(trial_diffs):
        out["vm_high_minus_low_trial_median"] = safe_float(np.nanmedian(trial_diffs))
    if len(trial_diffs) >= 2:
        try:
            stat = wilcoxon(trial_diffs, alternative="two-sided")
            out["vm_high_vs_low_trial_w"] = safe_float(stat.statistic)
            out["vm_high_vs_low_trial_p"] = safe_float(stat.pvalue)
        except ValueError:
            out["vm_high_vs_low_trial_w"] = np.nan
            out["vm_high_vs_low_trial_p"] = np.nan

    add_fdr_p_values(out, ["vm_high_vs_low_trial_p", "vm_high_vs_low_p", "p_vm_velocity"])
    return out


def baseline_arrays_for_plot(bin_rows: list[dict], metrics: dict):
    velocity = np.asarray([safe_float(r["velocity_median_cmps"]) for r in bin_rows], dtype=float)
    vm = np.asarray([safe_float(r["vm_median"]) for r in bin_rows], dtype=float)
    low_cut = safe_float(metrics.get("velocity_low_cut_cmps"))
    high_cut = safe_float(metrics.get("velocity_high_cut_cmps"))
    low = np.isfinite(velocity) & np.isfinite(low_cut) & (velocity <= low_cut)
    if metrics.get("velocity_split_mode") == "positive_vs_zero":
        high = np.isfinite(velocity) & np.isfinite(low_cut) & (velocity > low_cut)
    else:
        high = np.isfinite(velocity) & np.isfinite(high_cut) & (velocity >= high_cut)
    middle = np.isfinite(velocity) & ~(low | high)
    return velocity, vm, low, middle, high


def plot_baseline_condition_results(bin_rows: list[dict], metrics: dict) -> None:
    velocity, vm, low, middle, high = baseline_arrays_for_plot(bin_rows, metrics)
    vm_plot = gevi_display(vm)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    ax[0].hist(velocity[np.isfinite(velocity)], bins=25, color="0.7", edgecolor="white")
    if np.isfinite(metrics.get("velocity_low_cut_cmps", np.nan)):
        ax[0].axvline(metrics["velocity_low_cut_cmps"], color="tab:blue", ls="--", label="low cut")
    if metrics.get("velocity_split_mode") != "positive_vs_zero" and np.isfinite(metrics.get("velocity_high_cut_cmps", np.nan)):
        ax[0].axvline(metrics["velocity_high_cut_cmps"], color="tab:red", ls="--", label="high cut")
    ax[0].set_xlabel("bin velocity (cm/s)")
    ax[0].set_ylabel("n bins")
    ax[0].set_title("Velocity split")
    ax[0].legend(loc="best", fontsize=8)

    ax[1].boxplot([vm_plot[low], vm_plot[high]], labels=["low vel", "high vel"], showfliers=False)
    ax[1].scatter(np.ones(np.sum(low)), vm_plot[low], s=14, alpha=0.35, color="tab:blue")
    ax[1].scatter(np.ones(np.sum(high)) * 2, vm_plot[high], s=14, alpha=0.35, color="tab:red")
    ax[1].set_ylabel(GEVI_YLABEL)
    ax[1].set_title("Vm by velocity group")

    ax[2].scatter(velocity[middle], vm_plot[middle], s=14, alpha=0.22, color="0.5", label="middle")
    ax[2].scatter(velocity[low], vm_plot[low], s=18, alpha=0.45, color="tab:blue", label="low")
    ax[2].scatter(velocity[high], vm_plot[high], s=18, alpha=0.45, color="tab:red", label="high")
    ax[2].set_xlabel("bin velocity (cm/s)")
    ax[2].set_ylabel(GEVI_YLABEL)
    ax[2].set_title("Vm vs velocity")
    ax[2].legend(loc="best", fontsize=8)

    add_panel_letters(ax)
    save_figure(fig, f"{condition_output_stem()}_baseline_velocity")
    plt.show()


def save_baseline_condition_analysis(metrics: dict, bin_rows: list[dict], blocks_used: list[tuple[str, str, str]]) -> None:
    out = {
        "analysis": "baseline_velocity_split",
        "settings": {
            "mouse_name": MOUSE_NAME,
            "amplitude_uA": CONDITION_AMPLITUDE_UA,
            "exposure_ms": CONDITION_EXPOSURE_MS,
            "phase": CONDITION_PHASE,
            "imaging_side": CONDITION_IMAGING_SIDE,
            "date": CONDITION_DATE,
            "block": CONDITION_BLOCK,
            "low_velocity_percentile": float(LOW_VELOCITY_PERCENTILE),
            "high_velocity_percentile": float(HIGH_VELOCITY_PERCENTILE),
            "movement_threshold_cm_s": float(MOVEMENT_THRESHOLD_CM_S),
            "signal_mode": SIGNAL_MODE,
            "bin_sec": float(BIN_SEC),
        },
        "blocks_used": blocks_used,
        "metrics": metrics,
        "bins": bin_rows,
    }
    out_path = condition_output_path()
    with out_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {out_path}")


def run_baseline_condition_mode() -> None:
    blocks = list(iter_condition_blocks())
    if not blocks:
        raise SystemExit("No true baseline blocks with processed imaging and ephys files were found.")

    bin_rows = []
    for mouse, date, block in blocks:
        bin_rows.extend(load_baseline_block_bins(mouse, date, block))
    if not bin_rows:
        raise SystemExit("No usable baseline bins found in matching blocks.")

    metrics = baseline_velocity_metrics(bin_rows)
    print(f"\nBaseline condition mode | blocks={len(blocks)} | trials={metrics['n_trials']} | bins={metrics['n_bins']}")
    print("Filters:")
    print(
        f"  mouse={MOUSE_NAME}, amp=baseline, exp={CONDITION_EXPOSURE_MS}, "
        f"phase={CONDITION_PHASE}, side={CONDITION_IMAGING_SIDE}, bin={BIN_SEC:g} s"
    )
    print(
        f"Velocity split: low <= {metrics['velocity_low_cut_cmps']:.3g} cm/s "
        f"(n={metrics['n_low_velocity_bins']} bins), "
        + (
            f"high > {metrics['velocity_low_cut_cmps']:.3g} cm/s "
            if metrics.get("velocity_split_mode") == "positive_vs_zero"
            else f"high >= {metrics['velocity_high_cut_cmps']:.3g} cm/s "
        )
        + f"(n={metrics['n_high_velocity_bins']} bins)"
    )
    print(
        f"Vm high-low trial-paired: {metrics['vm_high_minus_low_trial_median']:.3g}, "
        f"{p_with_fdr_text(metrics, 'vm_high_vs_low_trial_p')}, "
        f"n={metrics['n_velocity_split_trials']} trials"
    )
    print(f"Vm vs velocity bins: r={metrics['r_vm_velocity']:.3f}, {p_with_fdr_text(metrics, 'p_vm_velocity')}, n={metrics['n_vm_velocity']} bins")

    if PRINT_TRIAL_TABLE:
        print("\nIncluded baseline blocks")
        for mouse, date, block in blocks:
            print(f"  {mouse} | {date} | {block}")

    print_ephys_check_summary(blocks)

    if SAVE_OUTPUT:
        save_baseline_condition_analysis(metrics, bin_rows, blocks)

    if SHOW_FIGURES:
        plot_baseline_condition_results(bin_rows, metrics)


def run_condition_mode() -> None:
    if is_baseline_condition_request():
        run_baseline_condition_mode()
        return

    blocks = list(iter_condition_blocks())
    if not blocks:
        raise SystemExit("No matching condition blocks with processed imaging and ephys files were found.")

    results = []
    for mouse, date, block in blocks:
        results.extend(load_block_results(mouse, date, block))
    if not results:
        raise SystemExit("No usable trials found in matching condition blocks.")

    trial_rows = compact_trial_rows(results)
    metrics = compute_block_metrics(results)
    metrics.update(velocity_split_metrics(trial_rows))

    print(f"\nCondition mode | blocks={len(blocks)} | trials={metrics['n_trials']}")
    print("Filters:")
    print(
        f"  mouse={MOUSE_NAME}, f={CONDITION_FREQUENCY_HZ}, amp={CONDITION_AMPLITUDE_UA}, "
        f"PW={CONDITION_PULSE_WIDTH_US}, exp={CONDITION_EXPOSURE_MS}, stim={CONDITION_STIMULATION_TIME_S}"
    )
    print(
        f"Velocity split: low <= {metrics['velocity_low_cut_cmps']:.3g} cm/s "
        f"(n={metrics['n_low_velocity_trials']}), high >= {metrics['velocity_high_cut_cmps']:.3g} cm/s "
        f"(n={metrics['n_high_velocity_trials']})"
    )
    print(f"Vm high-low: {metrics['vm_high_minus_low']:.3g}, {p_with_fdr_text(metrics, 'vm_high_vs_low_p')}")
    print(f"Hilbert high-low: {metrics['hilbert_high_minus_low']:.3g}, {p_with_fdr_text(metrics, 'hilbert_high_vs_low_p')}")
    print(f"Vm vs velocity: r={metrics['r_vm_velocity']:.3f}, {p_with_fdr_text(metrics, 'p_vm_velocity')}")
    print(f"Hilbert vs velocity: r={metrics['r_hilbert_velocity']:.3f}, {p_with_fdr_text(metrics, 'p_hilbert_velocity')}")
    print(f"Hilbert vs Vm: r={metrics['r_hilbert_vm']:.3f}, {p_with_fdr_text(metrics, 'p_hilbert_vm')}")
    if ANALYZE_THETA_BAND:
        print(
            f"Theta vs Hilbert: r={metrics['r_theta_hilbert']:.3f}, "
            f"{p_with_fdr_text(metrics, 'p_theta_hilbert')}"
        )
        print(
            f"Theta vs velocity: r={metrics['r_theta_velocity']:.3f}, "
            f"{p_with_fdr_text(metrics, 'p_theta_velocity')}"
        )
        print(
            f"Theta vs Vm: r={metrics['r_theta_vm']:.3f}, "
            f"{p_with_fdr_text(metrics, 'p_theta_vm')}"
        )
    print_selected_metric_summary(metrics, trial_rows)

    if PRINT_TRIAL_TABLE:
        print("\nIncluded blocks")
        for mouse, date, block in blocks:
            print(f"  {mouse} | {date} | {block}")

    print_ephys_check_summary(blocks)

    if SAVE_OUTPUT:
        save_condition_analysis(metrics, trial_rows, blocks)

    if SHOW_FIGURES:
        plot_condition_results(trial_rows, metrics)


def main() -> None:
    if ANALYSIS_MODE in {"condition", "baseline"}:
        run_condition_mode()
        return

    if not RUN_BATCH:
        run_current_block()
        return

    global SINGLE_DATE, SINGLE_BLOCK
    original_date = SINGLE_DATE
    original_block = SINGLE_BLOCK
    try:
        for date_name, block_name in list(iter_batch_blocks()):
            SINGLE_DATE = date_name
            SINGLE_BLOCK = block_name
            run_current_block(show_figures=False)
    finally:
        SINGLE_DATE = original_date
        SINGLE_BLOCK = original_block


if __name__ == "__main__":
    main()
