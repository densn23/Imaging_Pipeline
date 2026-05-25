from __future__ import annotations

import argparse
import csv
import pickle
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import plot as plot_summary_module
import process_data_PTA as first_pta_module
import process_data_PTA_mean as train_pta_module
import process_data_pulsogram as pulsogram_module
import process_data_filter as filter_module
import summarize as summarize_module
import table as table_module
from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = "Jamie10"
SINGLE_DATE = None
SINGLE_BLOCK = None
SELECTION_DATE = None
SELECTION_BLOCK = None


# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = False
USE_LOGBOOK_SELECTION = True
SAVE_OUTPUT = False
OVERWRITE = False
SHOW_PLOTS = True
SHOW_INCLUDED_BLOCKS = True
PLOT_STIM_TRACE = True


# -------------------------
# LOGBOOK SELECTION
# -------------------------
FREQUENCY_HZ = 40
PULSEWIDTH_DIRECTION = "Down"  # "up", "down", or None
EXCLUDE_BLOCKS = None  # e.g. "23-12-25 R6, Jamie10/07-01-26/R2"
AVERAGING_MODE = "equal_blocks"
FIXED_SELECTION_WINDOW_SEC = 1.0


# -------------------------
# PANELS (selection-mode averaging)
# -------------------------
PLOT_FULL_TRACE = True
PLOT_SINGLE_PTA = True
PLOT_MEAN_PTA = True
PLOT_LFP = False
PLOT_SPECTROGRAM = True
PLOT_LFP_SPECTROGRAM = False
PLOT_PULSOGRAM = True
PLOT_SIGNAL_HILBERT = False
PLOT_LFP_HILBERT = False


# -------------------------
# OUTPUT
# -------------------------
OUTPUT_SUFFIX = "_first_constant_train_summary.pkl"
ONLY_TRIAL = None
STIM_TABLE_CSV = DATA_ANALYSIS_ROOT / "tables" / "stim_table_all_jamie.csv"


# -------------------------
# FIRST-TRAIN DETECTION
# -------------------------
PRE_STIM_KEEP_SEC = 5.0
REFERENCE_PULSES = 10
MIN_TOTAL_PULSES = 8
MIN_KEPT_PULSES = 5
CONFIRM_WINDOW_PULSES = 5
CONFIRM_MIN_CHANGED = 3
MAX_PULSE_ANALYSIS_SEC = 0.004
PULSE_PRE_BASELINE_SEC = 0.0005
PULSE_WIDTH_REL_HEIGHT = 0.50
AMPLITUDE_CHANGE_REL_TOL = 0.05
WIDTH_CHANGE_REL_TOL = 0.10
IPI_CHANGE_REL_TOL = 0.05
PULSE_SPECTROGRAM_NPERSEG = 128
PULSE_SPECTROGRAM_OVERLAP_FRAC = 0.95


BLOCK_RE = re.compile(r"^R\d+$")


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


def parse_exclude_blocks(raw: str | None) -> list[tuple[str | None, str | None, str | None]]:
    if not raw:
        return []
    specs: list[tuple[str | None, str | None, str | None]] = []
    for raw_item in str(raw).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "/" in item:
            parts = [part.strip() for part in item.split("/") if part.strip()]
        else:
            parts = [part.strip() for part in item.split() if part.strip()]
        if len(parts) >= 3:
            specs.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            specs.append((None, parts[0], parts[1]))
        elif len(parts) == 1:
            specs.append((None, None, parts[0]))
    return specs


def row_is_excluded(row: dict[str, str], exclude_specs: list[tuple[str | None, str | None, str | None]]) -> bool:
    mouse = row.get("mouse", "")
    date = row.get("date", "")
    block = row.get("block", "")
    for mouse_ex, date_ex, block_ex in exclude_specs:
        if mouse_ex is not None and mouse_ex != mouse:
            continue
        if date_ex is not None and date_ex != date:
            continue
        if block_ex is not None and block_ex != block:
            continue
        return True
    return False


def normalize_text_label(value: Any) -> str:
    return str(value or "").strip().lower()


def pulsewidth_protocol_label(direction: str | None) -> str | None:
    if direction is None:
        return None
    low = str(direction).strip().lower()
    if low not in {"up", "down"}:
        raise ValueError("PULSEWIDTH_DIRECTION must be 'up', 'down', or None.")
    return f"pulsewidth analysis ramp {low}"


def load_selected_logbook_rows(mouse_names: list[str]) -> list[dict[str, str]]:
    exclude_specs = parse_exclude_blocks(EXCLUDE_BLOCKS)
    protocol_target = pulsewidth_protocol_label(PULSEWIDTH_DIRECTION)
    rows: list[dict[str, str]] = []
    with STIM_TABLE_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if mouse_names and row.get("mouse", "") not in mouse_names:
                continue
            if SELECTION_DATE is not None and row.get("date") != SELECTION_DATE:
                continue
            if SELECTION_BLOCK is not None and row.get("block") != SELECTION_BLOCK:
                continue
            if row_is_excluded(row, exclude_specs):
                continue
            protocol_label = normalize_text_label(row.get("protocol"))
            if protocol_target is not None:
                if protocol_label != protocol_target:
                    continue
            elif not protocol_label.startswith("pulsewidth analysis"):
                continue
            if FREQUENCY_HZ is not None:
                row_freq = safe_float(row.get("frequency_hz"))
                if not np.isfinite(row_freq) or not np.isfinite(FREQUENCY_HZ) or abs(row_freq - float(FREQUENCY_HZ)) > 0.6:
                    continue
            rows.append(row)
    rows.sort(
        key=lambda r: (
            table_module.mouse_key(r.get("mouse", "")),
            table_module.date_key(r.get("date", "")),
            table_module.natural_block_key(r.get("block", "")),
        )
    )
    return rows


def make_memory_blockspec(row: dict[str, str], summary_dict: dict, output_path: Path) -> table_module.BlockSpec:
    spec = table_module.BlockSpec(row=row, summary_path=output_path)
    spec._summary_cache = summary_dict
    return spec


def configure_table_plot_module(protocol_target: str | None) -> None:
    protocol_label = protocol_target if protocol_target is not None else "pulsewidth analysis"
    table_module.MOUSE_NAME = MOUSE_NAME
    table_module.FREQUENCY_HZ = FREQUENCY_HZ
    table_module.AMPLITUDE_UA = None
    table_module.PULSE_WIDTH_US = None
    table_module.EXPOSURE_MS = None
    table_module.STIMULATION_TIME_S = None
    table_module.PHASE = None
    table_module.IMAGING_SIDE = None
    table_module.DATE = SELECTION_DATE
    table_module.BLOCK = SELECTION_BLOCK
    table_module.PROTOCOL_CONTAINS = protocol_label
    table_module.AVERAGING_MODE = AVERAGING_MODE
    table_module.SAVE_FIGURE = False
    table_module.SHOW_FIGURE = SHOW_PLOTS
    table_module.SHOW_INCLUDED_BLOCKS = False
    table_module.PLOT_FULL_TRACE = PLOT_FULL_TRACE
    table_module.PLOT_STIM_TRACE = PLOT_STIM_TRACE
    table_module.PLOT_SINGLE_PTA = PLOT_SINGLE_PTA
    table_module.PLOT_MEAN_PTA = PLOT_MEAN_PTA
    table_module.PLOT_LFP = PLOT_LFP
    table_module.PLOT_SPECTROGRAM = PLOT_SPECTROGRAM
    table_module.PLOT_LFP_SPECTROGRAM = PLOT_LFP_SPECTROGRAM
    table_module.PLOT_PULSOGRAM = PLOT_PULSOGRAM
    table_module.PLOT_SIGNAL_HILBERT = PLOT_SIGNAL_HILBERT
    table_module.PLOT_LFP_HILBERT = PLOT_LFP_HILBERT


def trial_sort_key(name: str):
    return train_pta_module.trial_sort_key(name)


def safe_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def get_block_paths(mouse: str, date: str, block: str) -> dict[str, Path]:
    img_block = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block
    eph_block = DATA_ANALYSIS_ROOT / mouse / "Open_Ephys" / date / block
    return {
        "processed_notched": img_block / f"{block}_traces_processed_notched.pkl",
        "epoched_ephys": eph_block / f"{block}_epoched_ephys.pkl",
        "output": img_block / f"{block}{OUTPUT_SUFFIX}",
    }


def pulse_feature_table(td_e: dict) -> dict | None:
    stim = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    pulse_idx = np.asarray(td_e.get("stim_pulse_samples_in_trial", []), dtype=int)
    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    fs = safe_float(td_e.get("sample_rate"))

    if len(stim) < 2 or t.shape != stim.shape or len(pulse_idx) != len(pulse_times):
        return None
    if not np.isfinite(fs) or fs <= 0 or len(pulse_idx) == 0:
        return None

    pre_samp = max(1, int(round(float(PULSE_PRE_BASELINE_SEC) * fs)))
    max_win_samp = max(3, int(round(float(MAX_PULSE_ANALYSIS_SEC) * fs)))

    amps = np.full(len(pulse_idx), np.nan, dtype=float)
    widths_s = np.full(len(pulse_idx), np.nan, dtype=float)
    ipi_prev_s = np.full(len(pulse_idx), np.nan, dtype=float)

    for i, sample_idx in enumerate(pulse_idx):
        j0 = int(sample_idx)
        if j0 < 0 or j0 >= len(stim):
            continue

        next_idx = int(pulse_idx[i + 1]) if i + 1 < len(pulse_idx) else min(len(stim), j0 + max_win_samp)
        j1 = min(len(stim), max(j0 + 1, min(next_idx, j0 + max_win_samp)))

        seg = stim[j0:j1]
        pre = stim[max(0, j0 - pre_samp):j0]
        baseline = float(np.median(pre)) if len(pre) else 0.0
        env = np.abs(seg - baseline)
        if env.size == 0:
            continue

        amp = float(np.max(env))
        amps[i] = amp
        if np.isfinite(amp) and amp > 0:
            thr = float(PULSE_WIDTH_REL_HEIGHT) * amp
            active = np.flatnonzero(env >= thr)
            if active.size:
                widths_s[i] = float((active[-1] - active[0] + 1) / fs)

        if i >= 1:
            ipi_prev_s[i] = float(pulse_times[i] - pulse_times[i - 1])

    return {
        "pulse_times_s": pulse_times.astype(np.float64),
        "pulse_sample_idx": pulse_idx.astype(int),
        "amplitude": amps.astype(np.float64),
        "width_s": widths_s.astype(np.float64),
        "ipi_prev_s": ipi_prev_s.astype(np.float64),
        "sample_rate_hz": float(fs),
    }


def rel_change_mask(values: np.ndarray, ref_value: float, rel_tol: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros(values.shape, dtype=bool)
    if not np.isfinite(ref_value):
        return out
    scale = max(abs(float(ref_value)), 1e-12)
    out[np.isfinite(values)] = np.abs(values[np.isfinite(values)] - float(ref_value)) > float(rel_tol) * scale
    return out


def detect_first_change(features: dict) -> dict | None:
    pulse_times = np.asarray(features.get("pulse_times_s", []), dtype=float)
    amp = np.asarray(features.get("amplitude", []), dtype=float)
    width_s = np.asarray(features.get("width_s", []), dtype=float)
    ipi_prev_s = np.asarray(features.get("ipi_prev_s", []), dtype=float)

    n_pulses = len(pulse_times)
    if n_pulses < max(int(MIN_TOTAL_PULSES), 2):
        return None

    ref_n = min(int(REFERENCE_PULSES), n_pulses)
    if ref_n < 2:
        return None

    ref_amp = safe_float(np.nanmedian(amp[:ref_n]))
    ref_width_s = safe_float(np.nanmedian(width_s[:ref_n]))
    ref_ipi_s = safe_float(np.nanmedian(ipi_prev_s[1:ref_n])) if ref_n >= 2 else np.nan

    amp_changed = rel_change_mask(amp, ref_amp, AMPLITUDE_CHANGE_REL_TOL)
    width_changed = rel_change_mask(width_s, ref_width_s, WIDTH_CHANGE_REL_TOL)
    ipi_changed = rel_change_mask(ipi_prev_s, ref_ipi_s, IPI_CHANGE_REL_TOL)

    # For this special workflow, we want the end of the first stable pulse train.
    # Slow amplitude drift inside a train should not cut the segment early.
    # So the actual cut is driven by consistent width / pulse-period change.
    trigger_changed = width_changed | ipi_changed
    trigger_changed[:ref_n] = False
    any_changed = amp_changed | width_changed | ipi_changed
    any_changed[:ref_n] = False

    change_idx = None
    change_reasons: list[str] = []
    for i in range(ref_n, n_pulses):
        j_hi = min(n_pulses, i + int(CONFIRM_WINDOW_PULSES))
        window = trigger_changed[i:j_hi]
        if int(np.count_nonzero(window)) < int(CONFIRM_MIN_CHANGED):
            continue
        first_changed_rel = np.flatnonzero(window)
        if first_changed_rel.size == 0:
            continue
        change_idx = int(i + first_changed_rel[0])
        if width_changed[change_idx]:
            change_reasons.append("width")
        if ipi_changed[change_idx]:
            change_reasons.append("frequency")
        if amp_changed[change_idx]:
            change_reasons.append("amplitude")
        break

    change_detected = change_idx is not None
    if change_detected:
        cut_time_s = float(pulse_times[change_idx])
        kept_pulses = int(change_idx)
    else:
        cut_time_s = np.nan
        kept_pulses = int(n_pulses)

    ref_freq_hz = float(1.0 / ref_ipi_s) if np.isfinite(ref_ipi_s) and ref_ipi_s > 0 else np.nan
    change_amp = float(amp[change_idx]) if change_detected and np.isfinite(amp[change_idx]) else np.nan
    change_width_s = float(width_s[change_idx]) if change_detected and np.isfinite(width_s[change_idx]) else np.nan
    change_ipi_s = float(ipi_prev_s[change_idx]) if change_detected and np.isfinite(ipi_prev_s[change_idx]) else np.nan
    change_freq_hz = float(1.0 / change_ipi_s) if np.isfinite(change_ipi_s) and change_ipi_s > 0 else np.nan

    return {
        "change_detected": bool(change_detected),
        "change_pulse_index": int(change_idx) if change_detected else None,
        "change_time_s": float(cut_time_s) if np.isfinite(cut_time_s) else np.nan,
        "change_reasons": change_reasons,
        "reference": {
            "n_reference_pulses": int(ref_n),
            "amplitude": float(ref_amp) if np.isfinite(ref_amp) else np.nan,
            "width_s": float(ref_width_s) if np.isfinite(ref_width_s) else np.nan,
            "ipi_s": float(ref_ipi_s) if np.isfinite(ref_ipi_s) else np.nan,
            "frequency_hz": float(ref_freq_hz) if np.isfinite(ref_freq_hz) else np.nan,
        },
        "change_values": {
            "amplitude": float(change_amp) if np.isfinite(change_amp) else np.nan,
            "width_s": float(change_width_s) if np.isfinite(change_width_s) else np.nan,
            "ipi_s": float(change_ipi_s) if np.isfinite(change_ipi_s) else np.nan,
            "frequency_hz": float(change_freq_hz) if np.isfinite(change_freq_hz) else np.nan,
        },
        "n_pulses_total": int(n_pulses),
        "n_pulses_kept": int(kept_pulses),
        "pulse_features": {
            "pulse_times_s": pulse_times.astype(np.float64),
            "amplitude": amp.astype(np.float64),
            "width_s": width_s.astype(np.float64),
            "ipi_prev_s": ipi_prev_s.astype(np.float64),
            "amplitude_changed": amp_changed.astype(bool),
            "width_changed": width_changed.astype(bool),
            "frequency_changed": ipi_changed.astype(bool),
            "any_changed": any_changed.astype(bool),
        },
    }


def format_param_value(value: float, unit: str, scale: float = 1.0, decimals: int = 3) -> str:
    if not np.isfinite(safe_float(value)):
        return "nan"
    return f"{scale * float(value):.{decimals}f} {unit}".strip()


def print_detected_parameters(truncation_meta: dict) -> None:
    if not truncation_meta:
        return

    print("[INFO] detected first-train parameters:")
    for name in sorted(truncation_meta.keys(), key=trial_sort_key):
        meta = truncation_meta[name]
        ref = dict(meta.get("reference", {}))
        change = dict(meta.get("change_values", {}))
        reasons = list(meta.get("change_reasons", []))
        reason_text = ", ".join(reasons) if reasons else "none detected"
        cut_s = safe_float(meta.get("cut_time_s"))
        n_keep = int(meta.get("n_pulses_kept", 0))
        n_total = int(meta.get("n_pulses_total", 0))
        print(
            f"  {name} | stim_on=0.000 s | kept={n_keep}/{n_total} pulses | "
            f"cut={format_param_value(cut_s, 's', 1.0, 3)} | change={reason_text}"
        )
        print(
            "    ref: "
            f"f={format_param_value(ref.get('frequency_hz'), 'Hz', 1.0, 2)} | "
            f"width={format_param_value(ref.get('width_s'), 'us', 1e6, 1)} | "
            f"amp={format_param_value(ref.get('amplitude'), 'V', 1.0, 3)}"
        )
        if meta.get("change_detected", False):
            print(
                "    first changed pulse: "
                f"f={format_param_value(change.get('frequency_hz'), 'Hz', 1.0, 2)} | "
                f"width={format_param_value(change.get('width_s'), 'us', 1e6, 1)} | "
                f"amp={format_param_value(change.get('amplitude'), 'V', 1.0, 3)}"
            )


def truncate_imaging_trial(td_img: dict, cut_time_s: float | None) -> dict | None:
    t = np.asarray(td_img.get("t", []), dtype=float)
    if len(t) < 4:
        return None

    if cut_time_s is None or not np.isfinite(cut_time_s):
        keep = t >= -float(PRE_STIM_KEEP_SEC)
    else:
        keep = (t >= -float(PRE_STIM_KEEP_SEC)) & (t < float(cut_time_s))
    if int(np.sum(keep)) < 4:
        return None

    td_out = dict(td_img)
    td_out["t"] = np.asarray(t[keep], dtype=np.float64)

    for key in ["F_raw", "F_bleach_corr", "dff", "F_notched"]:
        arr = td_img.get(key)
        if arr is None:
            td_out[key] = None
            continue
        arr_np = np.asarray(arr)
        if arr_np.ndim == 1 and len(arr_np) == len(t):
            td_out[key] = arr_np[keep].copy()

    td_out["spectral_post_notch"] = {}
    td_out["first_constant_train_cut_s"] = float(cut_time_s) if cut_time_s is not None and np.isfinite(cut_time_s) else np.nan
    return td_out


def truncate_ephys_trial(td_e: dict, cut_time_s: float | None, n_pulses_kept: int) -> dict | None:
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    if len(t) < 4:
        return None

    if cut_time_s is None or not np.isfinite(cut_time_s):
        keep = t >= -float(PRE_STIM_KEEP_SEC)
    else:
        keep = (t >= -float(PRE_STIM_KEEP_SEC)) & (t < float(cut_time_s))
    if int(np.sum(keep)) < 4:
        return None

    td_out = dict(td_e)
    td_out["t_stim_s"] = np.asarray(t[keep], dtype=np.float64)

    channels_out = {}
    for key, arr in dict(td_e.get("channels", {})).items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 1 and len(arr_np) == len(t):
            channels_out[key] = arr_np[keep].copy()
        else:
            channels_out[key] = arr
    td_out["channels"] = channels_out

    cam_times = np.asarray(td_e.get("cam_frame_times_stim_s", []), dtype=float)
    if cut_time_s is None or not np.isfinite(cut_time_s):
        td_out["cam_frame_times_stim_s"] = np.asarray(cam_times[cam_times >= -float(PRE_STIM_KEEP_SEC)], dtype=np.float64)
    else:
        td_out["cam_frame_times_stim_s"] = np.asarray(
            cam_times[(cam_times >= -float(PRE_STIM_KEEP_SEC)) & (cam_times < float(cut_time_s))],
            dtype=np.float64,
        )

    pulse_samples = np.asarray(td_e.get("stim_pulse_samples_in_trial", []), dtype=int)
    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    n_keep = min(int(n_pulses_kept), len(pulse_samples), len(pulse_times))
    td_out["stim_pulse_samples_in_trial"] = pulse_samples[:n_keep].astype(int)
    td_out["stim_pulse_times_s"] = pulse_times[:n_keep].astype(np.float64)

    if n_keep >= 2:
        td_out["median_ipi_ms"] = float(1000.0 * np.median(np.diff(td_out["stim_pulse_times_s"])))
    else:
        td_out["median_ipi_ms"] = np.nan

    stim_on_frame_idx = td_e.get("stim_on_frame_idx", -1)
    td_out["stim_on_frame_idx"] = int(stim_on_frame_idx) if np.isfinite(safe_float(stim_on_frame_idx)) else -1
    if len(td_out["cam_frame_times_stim_s"]) and n_keep:
        last_pulse_s = float(td_out["stim_pulse_times_s"][-1])
        stim_off_frame_idx = int(np.searchsorted(td_out["cam_frame_times_stim_s"], last_pulse_s, side="right") - 1)
        td_out["stim_off_frame_idx"] = max(stim_off_frame_idx, td_out["stim_on_frame_idx"])
        td_out["stim_off_s_block"] = safe_float(td_e.get("t0_block_s")) + last_pulse_s
    else:
        td_out["stim_off_frame_idx"] = -1
        td_out["stim_off_s_block"] = np.nan

    td_out["first_constant_train_cut_s"] = float(cut_time_s) if cut_time_s is not None and np.isfinite(cut_time_s) else np.nan
    return td_out


def build_truncated_trials(
    processed_notched: dict,
    ephys: dict,
    fixed_cut_time_s: float | None = None,
) -> tuple[dict, dict, dict, dict]:
    img_trials = processed_notched.get("trials", {})
    e_trials = ephys.get("trials", {})

    truncated_img_trials = {}
    truncated_ephys_trials = {}
    truncation_meta = {}
    skipped = {}

    for name in sorted(img_trials.keys(), key=trial_sort_key):
        if ONLY_TRIAL is not None and name != ONLY_TRIAL:
            continue
        if name not in e_trials:
            skipped[name] = {"reason": "missing_ephys_trial"}
            continue

        features = pulse_feature_table(e_trials[name])
        if features is None:
            skipped[name] = {"reason": "invalid_stim_channel"}
            continue

        use_fixed_window = fixed_cut_time_s is not None and np.isfinite(safe_float(fixed_cut_time_s)) and float(fixed_cut_time_s) > 0
        if use_fixed_window:
            pulse_times_all = np.asarray(features.get("pulse_times_s", []), dtype=float)
            n_keep = int(np.count_nonzero(np.isfinite(pulse_times_all) & (pulse_times_all < float(fixed_cut_time_s))))
            detected = {
                "change_detected": True,
                "change_pulse_index": int(n_keep) if n_keep > 0 else None,
                "change_time_s": float(fixed_cut_time_s),
                "change_reasons": ["fixed_window"],
                "reference": {
                    "n_reference_pulses": min(int(REFERENCE_PULSES), int(len(pulse_times_all))),
                    "amplitude": np.nan,
                    "width_s": np.nan,
                    "ipi_s": np.nan,
                    "frequency_hz": np.nan,
                },
                "change_values": {
                    "amplitude": np.nan,
                    "width_s": np.nan,
                    "ipi_s": np.nan,
                    "frequency_hz": np.nan,
                },
                "n_pulses_total": int(len(pulse_times_all)),
                "n_pulses_kept": int(n_keep),
                "pulse_features": features,
            }
        else:
            detected = detect_first_change(features)
            if detected is None:
                skipped[name] = {"reason": "too_few_pulses"}
                continue

        n_keep = int(detected.get("n_pulses_kept", 0))
        cut_time_s = safe_float(detected.get("change_time_s"))
        if not detected.get("change_detected", False):
            cut_time_s = np.nan

        if not detected.get("change_detected", False):
            n_keep = int(detected.get("n_pulses_total", 0))
        if n_keep < int(MIN_KEPT_PULSES):
            skipped[name] = {
                "reason": "first_train_too_short",
                "n_pulses_kept": int(n_keep),
                "n_pulses_total": int(detected.get("n_pulses_total", 0)),
            }
            continue

        td_img_trunc = truncate_imaging_trial(img_trials[name], cut_time_s)
        td_e_trunc = truncate_ephys_trial(e_trials[name], cut_time_s, n_keep)
        if td_img_trunc is None or td_e_trunc is None:
            skipped[name] = {"reason": "truncate_failed"}
            continue

        truncated_img_trials[name] = td_img_trunc
        truncated_ephys_trials[name] = td_e_trunc
        truncation_meta[name] = {
            **detected,
            "cut_time_s": float(cut_time_s) if np.isfinite(cut_time_s) else np.nan,
            "analysis_mode": (
                "fixed_window"
                if use_fixed_window
                else ("first_change" if detected.get("change_detected", False) else "full_trial_no_change")
            ),
        }

    return truncated_img_trials, truncated_ephys_trials, truncation_meta, skipped


def populate_truncated_post_notch_spectral(trials: dict) -> tuple[dict, dict]:
    out_trials = {}
    for name, td in trials.items():
        td_out = dict(td)
        x = td_out.get("F_notched")
        fs = filter_module.get_fs_hz(td_out)
        if x is None or not np.isfinite(fs) or fs <= 0:
            td_out["spectral_post_notch"] = {}
        else:
            td_out["spectral_post_notch"] = filter_module.compute_post_notch_trial_spectral(np.asarray(x, dtype=float), float(fs))
        out_trials[name] = td_out
    return out_trials, filter_module.summarize_post_notch_spectral(out_trials)


def build_first_pta_output(mouse: str, date: str, block: str, img_trials: dict, e_trials: dict) -> tuple[dict | None, dict]:
    segments = []
    fail_counts = {}
    names = sorted(img_trials.keys(), key=trial_sort_key)

    for name in names:
        seg, err = first_pta_module.extract_first_pulse_segment(
            name,
            img_trials,
            e_trials,
            first_pta_module.PLOT_PRE_SEC,
            first_pta_module.PLOT_POST_SEC,
            first_pta_module.BASELINE_PRE_SEC,
        )
        if seg is None:
            fail_counts[err] = fail_counts.get(err, 0) + 1
            continue
        segments.append(seg)

    if not segments:
        return None, fail_counts

    t_grid = first_pta_module.build_common_grid(
        segments,
        first_pta_module.PLOT_PRE_SEC,
        first_pta_module.PLOT_POST_SEC,
    )
    if t_grid is None:
        fail_counts["invalid_common_grid"] = fail_counts.get("invalid_common_grid", 0) + 1
        return None, fail_counts

    Y = first_pta_module.interpolate_segments(segments, t_grid)
    with np.errstate(invalid="ignore"):
        y_mean = np.nanmean(Y, axis=0)
    y_sem = first_pta_module.nansem(Y, axis=0)
    y_spread = first_pta_module.nansd(Y, axis=0) if first_pta_module.SPREAD_MODE == "sd" else y_sem

    out = {
        "mouse": mouse,
        "date": date,
        "block": block,
        "analysis": "first_pulse_pta_first_constant_train",
        "settings": {
            "signal_mode": first_pta_module.SIGNAL_MODE,
            "plot_pre_sec": float(first_pta_module.PLOT_PRE_SEC),
            "plot_post_sec": float(first_pta_module.PLOT_POST_SEC),
            "baseline_pre_sec": float(first_pta_module.BASELINE_PRE_SEC),
            "fixed_response_window_sec": float(first_pta_module.FIXED_RESPONSE_WINDOW_SEC),
            "post_window_mode": first_pta_module.POST_WINDOW_MODE,
            "post_window_scale": float(first_pta_module.POST_WINDOW_SCALE),
            "period_fraction": float(first_pta_module.PERIOD_FRACTION),
            "baseline_mode": first_pta_module.BASELINE_MODE,
            "interp_mode": first_pta_module.INTERP_MODE,
            "min_samples_per_segment": int(first_pta_module.MIN_SAMPLES_PER_SEGMENT),
            "spread_mode": first_pta_module.SPREAD_MODE,
            "only_trial": ONLY_TRIAL,
            "subset_mode": "first_constant_train_only",
        },
        "trial_names_used": [seg["trial"] for seg in segments],
        "t_rel_s": np.asarray(t_grid, dtype=np.float64),
        "pta_mean": np.asarray(y_mean, dtype=np.float64),
        "pta_sem": np.asarray(y_sem, dtype=np.float64),
        "pta_spread": np.asarray(y_spread, dtype=np.float64),
        "segments": segments,
    }
    return out, fail_counts


def build_train_pta_output(mouse: str, date: str, block: str, img_trials: dict, e_trials: dict) -> tuple[dict | None, dict]:
    results = {}
    fail_counts = {}
    names = sorted(img_trials.keys(), key=trial_sort_key)

    old_nfft = train_pta_module.NFFT_SPEC
    old_overlap = train_pta_module.SPECTROGRAM_OVERLAP_FRAC
    try:
        train_pta_module.NFFT_SPEC = int(PULSE_SPECTROGRAM_NPERSEG)
        train_pta_module.SPECTROGRAM_OVERLAP_FRAC = float(PULSE_SPECTROGRAM_OVERLAP_FRAC)
        for name in names:
            tr, err = train_pta_module.analyze_trial(name, img_trials[name], e_trials[name])
            if tr is None:
                fail_counts[err] = fail_counts.get(err, 0) + 1
                continue
            results[name] = tr
    finally:
        train_pta_module.NFFT_SPEC = old_nfft
        train_pta_module.SPECTROGRAM_OVERLAP_FRAC = old_overlap

    if not results:
        return None, fail_counts

    trial_metrics = {name: results[name]["metrics"] for name in sorted(results.keys(), key=trial_sort_key)}
    out = {
        "mouse": mouse,
        "date": date,
        "block": block,
        "analysis": "pulse_train_pta_first_constant_train",
        "settings": {
            "signal_mode": train_pta_module.SIGNAL_MODE,
            "plot_pre_sec": float(train_pta_module.PRE_SEC),
            "plot_post_sec": float(train_pta_module.POST_SEC),
            "extract_pre_sec": float(train_pta_module.EXTRACT_PRE_SEC),
            "extract_post_sec": float(train_pta_module.EXTRACT_POST_SEC),
            "pulse_window_mode": train_pta_module.PULSE_WINDOW_MODE,
            "pulse_window_scale": float(train_pta_module.PULSE_WINDOW_SCALE),
            "period_fraction": float(train_pta_module.PERIOD_FRACTION),
            "baseline_mode": train_pta_module.BASELINE_MODE,
            "global_baseline_stat": train_pta_module.GLOBAL_BASELINE_STAT,
            "global_baseline_pre_sec": float(train_pta_module.GLOBAL_BASELINE_PRE_SEC),
            "min_pulses": int(train_pta_module.MIN_PULSES),
            "show_pulse_windows": bool(train_pta_module.SHOW_PULSE_WINDOWS),
            "spread_mode": train_pta_module.SPREAD_MODE,
            "save_spectrogram": bool(train_pta_module.SAVE_SPECTROGRAM),
            "spectrogram_window": train_pta_module.SPECTROGRAM_WINDOW,
            "spectrogram_nperseg": int(PULSE_SPECTROGRAM_NPERSEG),
            "spectrogram_overlap_frac": float(PULSE_SPECTROGRAM_OVERLAP_FRAC),
            "spectrogram_relative_baseline_end_s": float(train_pta_module.SPECTROGRAM_BASELINE_END_S),
            "spectrogram_relative_baseline_stat": train_pta_module.SPECTROGRAM_BASELINE_STAT,
            "spectrogram_mode": train_pta_module.SPECTROGRAM_MODE,
            "spectrogram_scale": train_pta_module.SPECTROGRAM_SCALE,
            "spectrogram_baseline_pre_sec": float(train_pta_module.SPECTROGRAM_BASELINE_PRE_SEC),
            "spectrogram_display_percentiles": [float(v) for v in train_pta_module.SPECTROGRAM_DISPLAY_PERCENTILES],
            "spectrogram_interpolation": train_pta_module.SPECTROGRAM_INTERPOLATION,
            "only_trial": ONLY_TRIAL,
            "subset_mode": "first_constant_train_only",
        },
        "trial_results": results,
        "trial_metrics": trial_metrics,
    }
    return out, fail_counts


def build_pulsogram_output(mouse: str, date: str, block: str, img_trials: dict, e_trials: dict) -> tuple[dict | None, dict]:
    results = {}
    fail_counts = {}
    names = sorted(img_trials.keys(), key=trial_sort_key)

    for name in names:
        tr, err = pulsogram_module.analyze_trial(name, img_trials[name], e_trials[name])
        if tr is None:
            fail_counts[err] = fail_counts.get(err, 0) + 1
            continue
        results[name] = tr

    if not results:
        return None, fail_counts

    pooled = pulsogram_module.pool_pulse_metrics(list(results.values()))
    if not pooled:
        fail_counts["no_pooled_metrics"] = fail_counts.get("no_pooled_metrics", 0) + 1
        return None, fail_counts

    out = {
        "mouse": mouse,
        "date": date,
        "block": block,
        "analysis": "pulsogram_first_constant_train",
        "settings": {
            "signal_mode": pulsogram_module.SIGNAL_MODE,
            "pre_sec": float(pulsogram_module.PRE_SEC),
            "post_sec": float(pulsogram_module.POST_SEC),
            "post_window_mode": pulsogram_module.POST_WINDOW_MODE,
            "post_window_scale": float(pulsogram_module.POST_WINDOW_SCALE),
            "period_fraction": float(pulsogram_module.PERIOD_FRACTION),
            "baseline_mode": pulsogram_module.BASELINE_MODE,
            "min_pulses": int(pulsogram_module.MIN_PULSES),
            "spread_mode": pulsogram_module.SPREAD_MODE,
            "peak_window_s": list(pulsogram_module.PEAK_WINDOW_S),
            "only_trial": ONLY_TRIAL,
            "save_csv": False,
            "show_heatmap": False,
            "show_overplotted_lines": False,
            "plot_pulse_stride": int(pulsogram_module.PLOT_PULSE_STRIDE),
            "subset_mode": "first_constant_train_only",
        },
        "trial_results": results,
        "pooled_metrics": pooled,
    }
    return out, fail_counts


def build_truncation_summary(meta_by_trial: dict, skipped: dict, n_input_trials: int) -> dict:
    cut_times = []
    n_total = []
    n_kept = []
    reason_counts = {"amplitude": 0, "width": 0, "frequency": 0, "none": 0}

    for meta in meta_by_trial.values():
        cut_s = safe_float(meta.get("cut_time_s"))
        if np.isfinite(cut_s):
            cut_times.append(cut_s)
        n_total.append(safe_float(meta.get("n_pulses_total")))
        n_kept.append(safe_float(meta.get("n_pulses_kept")))
        reasons = list(meta.get("change_reasons", []))
        if not reasons:
            reason_counts["none"] += 1
        for reason in reasons:
            if reason in reason_counts:
                reason_counts[reason] += 1

    keep_frac = []
    for a, b in zip(n_kept, n_total):
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            keep_frac.append(float(a / b))

    return {
        "n_input_trials": int(n_input_trials),
        "n_kept_trials": int(len(meta_by_trial)),
        "n_skipped_trials": int(len(skipped)),
        "cut_time_s_mean": safe_float(np.nanmean(cut_times)) if cut_times else np.nan,
        "cut_time_s_sd": safe_float(np.nanstd(cut_times, ddof=1)) if len(cut_times) >= 2 else np.nan,
        "n_pulses_total_mean": safe_float(np.nanmean(n_total)) if n_total else np.nan,
        "n_pulses_kept_mean": safe_float(np.nanmean(n_kept)) if n_kept else np.nan,
        "kept_fraction_mean": safe_float(np.nanmean(keep_frac)) if keep_frac else np.nan,
        "change_reason_counts": reason_counts,
        "skipped_trials": skipped,
    }


def build_union_axis_1d(arrays: list[np.ndarray]) -> np.ndarray | None:
    valid = [np.asarray(a, dtype=float) for a in arrays if a is not None and len(a) >= 2]
    if not valid:
        return None
    lo = min(float(np.nanmin(a)) for a in valid)
    hi = max(float(np.nanmax(a)) for a in valid)
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


def collect_union_grid_trials(trials: dict, field_names: list[str], t_key: str = "t") -> dict:
    out = {
        "trial_names": [],
        "t_common": np.array([], dtype=float),
    }
    for field in field_names:
        out[f"{field}_mean"] = np.array([], dtype=float)
        out[f"{field}_sd"] = np.array([], dtype=float)
        out[f"{field}_sem"] = np.array([], dtype=float)
        out[f"{field}_trial_names"] = []
        out[f"{field}_n"] = np.array([], dtype=float)

    time_axes = []
    for name in sorted(trials.keys()):
        t = np.asarray(trials[name].get(t_key, []), dtype=float)
        if len(t) >= 2:
            time_axes.append(t)
    t_ref = build_union_axis_1d(time_axes)
    if t_ref is None:
        return out

    out["t_common"] = np.asarray(t_ref, dtype=np.float64)
    used_names = {field: [] for field in field_names}
    stacks = {field: [] for field in field_names}

    for name in sorted(trials.keys()):
        td = trials[name]
        t = np.asarray(td.get(t_key, []), dtype=float)
        if len(t) < 2:
            continue
        for field in field_names:
            arr = td.get(field)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.shape != t.shape:
                continue
            yi = summarize_module.interpolate_to_ref_grid(t_ref, t, arr)
            if not np.any(np.isfinite(yi)):
                continue
            stacks[field].append(yi)
            used_names[field].append(name)

    out["trial_names"] = sorted({name for names in used_names.values() for name in names})
    for field in field_names:
        out[f"{field}_trial_names"] = used_names[field]
        if not stacks[field]:
            continue
        stack = np.vstack(stacks[field])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[f"{field}_mean"] = np.asarray(np.nanmean(stack, axis=0), dtype=np.float64)
            out[f"{field}_sd"] = (
                np.asarray(np.nanstd(stack, axis=0, ddof=1), dtype=np.float64)
                if stack.shape[0] >= 2
                else np.full(stack.shape[1], np.nan, dtype=np.float64)
            )
            out[f"{field}_sem"] = (
                np.asarray(np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0]), dtype=np.float64)
                if stack.shape[0] >= 2
                else np.full(stack.shape[1], np.nan, dtype=np.float64)
            )
        out[f"{field}_n"] = np.sum(np.isfinite(stack), axis=0).astype(np.float64)

    return out


def summarize_processed_notched_union(processed_notched: dict, ephys: dict) -> dict:
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
    all_summary = collect_union_grid_trials(all_trials, fields, t_key="t")
    baseline_summary = collect_union_grid_trials(baseline_trials, fields, t_key="t")
    stim_summary = collect_union_grid_trials(stim_trials, fields, t_key="t")

    baseline_spec = summarize_module.compute_spectrogram_data(
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


def build_union_spectrogram_summary(
    trial_results: dict,
    baseline_end_s: float = summarize_module.DEFAULT_SPEC_REL_BASELINE_END_S,
    baseline_stat: str = summarize_module.DEFAULT_SPEC_REL_BASELINE_STAT,
) -> dict:
    specs = []
    for name in sorted(trial_results.keys()):
        sec = trial_results[name].get("spectrogram", {})
        time_s = np.asarray(sec.get("time_s", []), dtype=float)
        freq_hz = np.asarray(sec.get("freq_hz", []), dtype=float)
        power_linear = summarize_module.spectrogram_linear_power(sec)
        if len(time_s) == 0 or len(freq_hz) == 0 or power_linear.shape != (len(freq_hz), len(time_s)):
            continue
        specs.append({"name": name, "time_s": time_s, "freq_hz": freq_hz, "power_linear": power_linear})

    if not specs:
        return summarize_module.empty_train_spectrogram_summary()

    time_ref = build_union_axis_1d([s["time_s"] for s in specs])
    freq_ref = build_union_axis_1d([s["freq_hz"] for s in specs])
    if time_ref is None or freq_ref is None:
        return summarize_module.empty_train_spectrogram_summary()

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
        return summarize_module.empty_train_spectrogram_summary()

    stack_linear = np.stack(stack_linear, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_linear = np.nanmean(stack_linear, axis=0)
        sd_linear = (
            np.nanstd(stack_linear, axis=0, ddof=1)
            if stack_linear.shape[0] >= 2
            else np.full((len(freq_ref), len(time_ref)), np.nan, dtype=float)
        )
        stack_db = summarize_module.db_from_linear(stack_linear)
        sd_db = (
            np.nanstd(stack_db, axis=0, ddof=1)
            if stack_linear.shape[0] >= 2
            else np.full((len(freq_ref), len(time_ref)), np.nan, dtype=float)
        )

    n_trials_per_bin = np.sum(np.isfinite(stack_linear), axis=0).astype(np.float64)
    rel = summarize_module.compute_relative_spectrogram(
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
        "power_db_mean": summarize_module.db_from_linear(mean_linear),
        "power_db_sd": np.asarray(sd_db, dtype=np.float64),
        "relative_linear_mean": np.asarray(rel.get("power_linear", []), dtype=np.float64),
        "relative_db_mean": np.asarray(rel.get("power_db", []), dtype=np.float64),
    }


def build_final_summary_dict(
    mouse: str,
    date: str,
    block: str,
    source_paths: dict[str, Path],
    processed_notched: dict,
    ephys: dict,
    truncation_meta: dict,
    skipped_trials: dict,
    first_pta: dict | None,
    train_pta: dict | None,
    pulsogram: dict | None,
) -> dict:
    processed_trials = processed_notched.get("trials", {})
    ephys_trials = ephys.get("trials", {})
    stim_trial_names = sorted(processed_trials.keys(), key=trial_sort_key)

    settings = {} if train_pta is None else dict(train_pta.get("settings", {}))
    spec_baseline_end_s = safe_float(settings.get("spectrogram_relative_baseline_end_s"))
    if not np.isfinite(spec_baseline_end_s):
        spec_baseline_end_s = float(summarize_module.DEFAULT_SPEC_REL_BASELINE_END_S)
    spec_baseline_stat = str(settings.get("spectrogram_relative_baseline_stat", summarize_module.DEFAULT_SPEC_REL_BASELINE_STAT)).lower()
    if spec_baseline_stat not in {"mean", "median"}:
        spec_baseline_stat = summarize_module.DEFAULT_SPEC_REL_BASELINE_STAT

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ephys_summary = summarize_module.summarize_ephys(ephys)
        processed_summary = summarize_module.summarize_processed_notched(processed_notched, ephys)
        single_pta_summary = summarize_module.summarize_first_pta(first_pta)
        train_pta_summary = summarize_module.summarize_train_pta(train_pta)
        pulsogram_summary = summarize_module.summarize_pulsogram(pulsogram)

    # For this special pulse script, keep the full-trace panel simple:
    # use the strict common-support mean and suppress the SD shading.
    for section_name in ["all", "baseline", "stim"]:
        sec = processed_summary.get(section_name, {})
        if not isinstance(sec, dict):
            continue
        for field in ["F_notched_sd", "F_raw_sd", "F_bleach_corr_sd", "dff_sd"]:
            if field in sec:
                sec[field] = np.array([], dtype=float)
    if train_pta is not None:
        train_pta_summary["spectrogram"] = build_union_spectrogram_summary(
            trial_results=train_pta.get("trial_results", {}),
            baseline_end_s=float(spec_baseline_end_s),
            baseline_stat=spec_baseline_stat,
        )

    return {
        "mouse": mouse,
        "date": date,
        "block": block,
        "analysis": "first_constant_train_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_paths": {k: str(v) for k, v in source_paths.items() if k != "output"},
        "source_settings": {
            "processed_notched": processed_notched.get("notch_processing", {}),
            "pulse_first_constant_train": {
                "pre_stim_keep_sec": float(PRE_STIM_KEEP_SEC),
                "reference_pulses": int(REFERENCE_PULSES),
                "min_total_pulses": int(MIN_TOTAL_PULSES),
                "min_kept_pulses": int(MIN_KEPT_PULSES),
                "confirm_window_pulses": int(CONFIRM_WINDOW_PULSES),
                "confirm_min_changed": int(CONFIRM_MIN_CHANGED),
                "max_pulse_analysis_sec": float(MAX_PULSE_ANALYSIS_SEC),
                "pulse_pre_baseline_sec": float(PULSE_PRE_BASELINE_SEC),
                "pulse_width_rel_height": float(PULSE_WIDTH_REL_HEIGHT),
                "amplitude_change_rel_tol": float(AMPLITUDE_CHANGE_REL_TOL),
                "width_change_rel_tol": float(WIDTH_CHANGE_REL_TOL),
                "ipi_change_rel_tol": float(IPI_CHANGE_REL_TOL),
                "spectrogram_nperseg": int(PULSE_SPECTROGRAM_NPERSEG),
                "spectrogram_overlap_frac": float(PULSE_SPECTROGRAM_OVERLAP_FRAC),
                "only_trial": ONLY_TRIAL,
            },
            "first_pta": {} if first_pta is None else first_pta.get("settings", {}),
            "train_pta": {} if train_pta is None else train_pta.get("settings", {}),
            "pulsogram": {} if pulsogram is None else pulsogram.get("settings", {}),
        },
        "summary": {
            "trial_counts": {
                "total": int(len(processed_trials)),
                "baseline": 0,
                "stim": int(len(stim_trial_names)),
                "single_pta_valid": 0 if first_pta is None else int(len(first_pta.get("trial_names_used", []))),
                "train_pta_valid": 0 if train_pta is None else int(len(train_pta.get("trial_results", {}))),
                "pulsogram_valid": 0 if pulsogram is None else int(len(pulsogram.get("trial_results", {}))),
            },
            "truncation": build_truncation_summary(truncation_meta, skipped_trials, len(processed_trials) + len(skipped_trials)),
            "ephys": ephys_summary,
            "processed_notched": processed_summary,
            "single_pta": single_pta_summary,
            "train_pta": train_pta_summary,
            "pulsogram": pulsogram_summary,
        },
        "trials": {
            "baseline_trial_names": [],
            "stim_trial_names": stim_trial_names,
            "processed_notched": processed_trials,
            "ephys": ephys_trials,
            "truncation": truncation_meta,
            "skipped_trials": skipped_trials,
            "first_pta_segments": [] if first_pta is None else first_pta.get("segments", []),
            "train_pta": {} if train_pta is None else train_pta.get("trial_results", {}),
            "pulsogram": {} if pulsogram is None else pulsogram.get("trial_results", {}),
        },
    }


def process_block(
    mouse: str,
    date: str,
    block: str,
    show_plots: bool | None = None,
    fixed_cut_time_s: float | None = None,
) -> dict:
    paths = get_block_paths(mouse, date, block)
    label = f"{mouse} | {date} | {block}"
    do_show_plots = SHOW_PLOTS if show_plots is None else bool(show_plots)

    missing = [name for name in ["processed_notched", "epoched_ephys"] if not paths[name].exists()]
    if missing:
        print(f"[SKIP] {label} | missing: {', '.join(missing)}")
        return {"status": "missing_inputs", "label": label, "missing": missing}

    if SAVE_OUTPUT and paths["output"].exists() and not OVERWRITE:
        print(f"[SKIP] {label} | output exists")
        return {"status": "skipped_existing", "label": label, "path": str(paths['output']), "output_path": paths["output"]}

    with open(paths["processed_notched"], "rb") as f:
        processed_raw = pickle.load(f)
    with open(paths["epoched_ephys"], "rb") as f:
        ephys_raw = pickle.load(f)

    img_trials, ephys_trials, truncation_meta, skipped = build_truncated_trials(
        processed_raw,
        ephys_raw,
        fixed_cut_time_s=fixed_cut_time_s,
    )
    if not img_trials or not ephys_trials:
        print(f"[SKIP] {label} | no valid first-train trials")
        if skipped:
            print(f"[INFO] skipped: {skipped}")
        return {"status": "no_valid_trials", "label": label, "skipped": skipped, "output_path": paths["output"]}

    img_trials, spectral_summary = populate_truncated_post_notch_spectral(img_trials)
    processed_trunc = {
        "mouse": processed_raw.get("mouse"),
        "date": processed_raw.get("date"),
        "block": processed_raw.get("block"),
        "analysis": "processed_notched_first_constant_train",
        "trials": img_trials,
        "notch_processing": processed_raw.get("notch_processing", {}),
        "spectral_summary_post_notch": spectral_summary,
    }
    ephys_trunc = {
        "mouse": ephys_raw.get("mouse"),
        "date": ephys_raw.get("date"),
        "block": ephys_raw.get("block"),
        "sample_rate": ephys_raw.get("sample_rate"),
        "analysis": "epoched_ephys_first_constant_train",
        "trials": ephys_trials,
    }

    first_pta, first_fail = build_first_pta_output(mouse, date, block, img_trials, ephys_trials)
    train_pta, train_fail = build_train_pta_output(mouse, date, block, img_trials, ephys_trials)
    pulsogram, pulse_fail = build_pulsogram_output(mouse, date, block, img_trials, ephys_trials)

    final_dict = build_final_summary_dict(
        mouse=mouse,
        date=date,
        block=block,
        source_paths=paths,
        processed_notched=processed_trunc,
        ephys=ephys_trunc,
        truncation_meta=truncation_meta,
        skipped_trials=skipped,
        first_pta=first_pta,
        train_pta=train_pta,
        pulsogram=pulsogram,
    )

    print(f"[RUN] {label}")
    print(f"[INFO] kept trials: {len(img_trials)} | skipped: {len(skipped)}")
    trunc_summary = final_dict["summary"]["truncation"]
    print(
        f"[INFO] kept pulses mean: {trunc_summary.get('n_pulses_kept_mean', np.nan):.2f} / "
        f"{trunc_summary.get('n_pulses_total_mean', np.nan):.2f} | "
        f"cut mean: {trunc_summary.get('cut_time_s_mean', np.nan):.3f} s"
    )
    print_detected_parameters(truncation_meta)
    if first_fail:
        print(f"[INFO] first PTA skipped: {first_fail}")
    if train_fail:
        print(f"[INFO] train PTA skipped: {train_fail}")
    if pulse_fail:
        print(f"[INFO] pulsogram skipped: {pulse_fail}")

    if SAVE_OUTPUT:
        with open(paths["output"], "wb") as f:
            pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] {paths['output']}")

    if do_show_plots:
        old_show = plot_summary_module.SHOW_FIGURE
        old_save = plot_summary_module.SAVE_FIGURE
        old_plot_stim = plot_summary_module.PLOT_STIM_TRACE
        try:
            plot_summary_module.SHOW_FIGURE = True
            plot_summary_module.SAVE_FIGURE = False
            plot_summary_module.PLOT_STIM_TRACE = PLOT_STIM_TRACE
            plot_summary_module.plot_summary(final_dict, save_path=None)
        finally:
            plot_summary_module.SHOW_FIGURE = old_show
            plot_summary_module.SAVE_FIGURE = old_save
            plot_summary_module.PLOT_STIM_TRACE = old_plot_stim

    return {
        "status": "saved" if SAVE_OUTPUT else "done",
        "label": label,
        "path": str(paths["output"]),
        "output_path": paths["output"],
        "summary_dict": final_dict,
    }


def run_batch(mouse: str) -> list[dict]:
    imaging_root = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data"
    if not imaging_root.exists():
        raise FileNotFoundError(f"Imaging root not found: {imaging_root}")

    results = []
    for date_dir in sorted(imaging_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
                continue
            results.append(process_block(mouse, date_dir.name, block_dir.name))
    return results


def run_single_date(mouse: str, date: str) -> list[dict]:
    imaging_root = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date
    if not imaging_root.exists():
        raise FileNotFoundError(f"Imaging date root not found: {imaging_root}")

    results = []
    for block_dir in sorted(imaging_root.iterdir()):
        if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
            continue
        results.append(process_block(mouse, date, block_dir.name))
    return results


def selection_output_stem(protocol_target: str | None) -> str:
    protocol_label = protocol_target if protocol_target is not None else "pulsewidth analysis"
    parts = ["pulsefirsttrain"]
    mouse_names = parse_mouse_names(MOUSE_NAME)
    if mouse_names:
        parts.append("-".join(mouse_names))
    else:
        parts.append("allmice")
    if FREQUENCY_HZ is not None and np.isfinite(safe_float(FREQUENCY_HZ)):
        parts.append(f"{float(FREQUENCY_HZ):g}Hz")
    if protocol_label:
        parts.append(protocol_label.replace(" ", "_"))
    if SELECTION_DATE:
        parts.append(SELECTION_DATE)
    if SELECTION_BLOCK:
        parts.append(SELECTION_BLOCK)
    return table_module.slugify("_".join(parts))


def run_logbook_selection(mouse_names: list[str]) -> list[dict]:
    protocol_target = pulsewidth_protocol_label(PULSEWIDTH_DIRECTION)
    rows = load_selected_logbook_rows(mouse_names)
    if not rows:
        print("No matching pulsewidth-analysis blocks found in stim_table_all_jamie.csv.")
        return []

    configure_table_plot_module(protocol_target)

    processed_specs: list[table_module.BlockSpec] = []
    results: list[dict] = []
    for row in rows:
        result = process_block(
            row["mouse"],
            row["date"],
            row["block"],
            show_plots=False,
            fixed_cut_time_s=FIXED_SELECTION_WINDOW_SEC,
        )
        results.append(result)
        summary_dict = result.get("summary_dict")
        if summary_dict is None:
            continue
        processed_specs.append(make_memory_blockspec(row, summary_dict, Path(result.get("path", ""))))

    if not processed_specs:
        print("No valid first-train summaries were produced for the selected blocks.")
        return results

    table_module.print_included_blocks(processed_specs, [])
    stem = selection_output_stem(protocol_target)
    out_png = DATA_ANALYSIS_ROOT / "tables" / f"{stem}.png"
    table_module.plot_condition_average(processed_specs, out_png)

    if SHOW_INCLUDED_BLOCKS:
        for block in processed_specs:
            print(f"\n[BLOCK VIEW] {block.label}")
            block_png = DATA_ANALYSIS_ROOT / "tables" / f"{table_module.block_output_stem(stem, block)}.png"
            table_module.plot_condition_average([block], block_png)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze only the first constant stimulation train within each trial.")
    parser.add_argument("--mouse", default=MOUSE_NAME)
    parser.add_argument("--date", default=SINGLE_DATE)
    parser.add_argument("--block", default=SINGLE_BLOCK)
    parser.add_argument("--trial", default=ONLY_TRIAL)
    parser.add_argument("--selection", action=argparse.BooleanOptionalAction, default=USE_LOGBOOK_SELECTION)
    parser.add_argument("--batch", action="store_true", default=RUN_BATCH)
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=SHOW_PLOTS)
    parser.add_argument("--save-output", action=argparse.BooleanOptionalAction, default=SAVE_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser.parse_args()


def main() -> None:
    global ONLY_TRIAL, SHOW_PLOTS, SAVE_OUTPUT, OVERWRITE

    args = parse_args()
    ONLY_TRIAL = args.trial
    SHOW_PLOTS = bool(args.show_plots)
    SAVE_OUTPUT = bool(args.save_output)
    OVERWRITE = bool(args.overwrite)

    mouse_names = resolve_mouse_names(args.mouse)
    if not mouse_names:
        print("No mice found to process.")
        return

    if args.selection:
        run_logbook_selection(mouse_names)
        return

    if args.batch:
        for mouse in mouse_names:
            run_batch(mouse)
        return

    if args.date is not None and args.block is None:
        for mouse in mouse_names:
            run_single_date(mouse, args.date)
        return

    if args.date is None or args.block is None:
        print("Set --date and --block to run one block, or set --batch / --date only for larger runs.")
        return

    for mouse in mouse_names:
        process_block(mouse, args.date, args.block)


if __name__ == "__main__":
    main()
