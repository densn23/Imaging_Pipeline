from __future__ import annotations

import csv
import math
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import mannwhitneyu


DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent
TABLES_DIR = DATA_ANALYSIS_ROOT / "tables"
FIGURES_DIR = DATA_ANALYSIS_ROOT / "figures"
STIM_TABLE_CSV = TABLES_DIR / "stim_table_all_jamie.csv"


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------
MOUSE_NAME = None
FREQUENCY_HZ = 10
AMPLITUDE_UA = None  # None = any non-class amplitude; use 30, 32.5, theta, ramp, up, down, cortex, thr, or baseline
THRESHOLD_AMPLITUDE_UA_BY_MOUSE = {
    "jamie6": 20.0,
    "jamie8": 25.0,
    "jamie10": 25.0,
    "jamie11": 25.0,
    "jamie12": 30.0,
    "vinnie1": 25.0,
}
THRESHOLD_AMPLITUDE_LABELS = {"thr", "threshold", "at_thr", "at threshold"}
PULSE_WIDTH_US = 100
EXPOSURE_MS = "1, 2"
STIMULATION_TIME_S = 10
PHASE = None # "Pre", "Post", or None
IMAGING_SIDE = "Left"  # "Left", "Right", or None
IMAGING_SIDE_FLIP_MICE = {"vinnie1"}  # requested Left selects Right for these mice, and vice versa
FIBER_PLACEMENT_CONTAINS = None  # e.g. "STN" or "ZI"
DATE = None
BLOCK = None
PROTOCOL_CONTAINS = None
EXCLUDE_BLOCKS = "Jamie12/06-04-26/R4, Jamie8/09-12-25/R5, Jamie8/19-12-25/R4, Jamie8/31-12-25/R1, Jamie12/01-04-26/R3, Jamie11/29-04-26/R5, Jamie11/29-04-26/R6, Jamie11/01-05-26/R5, Jamie11/01-05-26/R7, Jamie11/01-05-26/R6"      # e.g. "19-12-25 R4" or "Jamie8/19-12-25/R4, Jamie11/03-04-26/R18"
ALLOWED_N_TRIALS = (5, 10)  # None = allow any block trial count

# Optional stats printout. If True, print Group A vs B stats before plotting.
COMPARE_GROUPS = True
PLOT_GROUP_COMPARISON = True
# True = plot GROUP_A and GROUP_B as two rows instead of Selection
GROUP_COMPARISON_METRIC = "plv_pooled, hilbert_amp_ratio, vm_early, vm_late"  # options: hilbert_amp_ratio, plv, plv_pooled, vm_early, vm_late, spta_amp, mpta_lat, mpta_jit, mpta_amp, peak_1_latency_ms_median, peak_1_jitter_ms, r_vm_velocity, r_hilbert_velocity, r_hilbert_vm
GROUP_A = {
    "label": "Group A",
    "mouse_name": "Jamie10, Jamie11, Vinnie1",
    "frequency_hz": 135,
    "amplitude_uA": 25,
    "pulse_width_us": 100,
    "exposure_ms": "1, 2",
    "stimulation_time_s": 10,
    "phase": "Pre",
    "imaging_side": "Left",
    "fiber_placement_contains": None,
    "date": None,
    "block": None,
    "protocol_contains": None,
    "exclude_blocks": "Jamie12/06-04-26/R4, Jamie8/09-12-25/R5, Jamie8/19-12-25/R4, Jamie8/31-12-25/R1, Jamie12/01-04-26/R3",
    "allowed_n_trials": (5, 10),
}
GROUP_B = {
    "label": "Group B",
    "mouse_name": "Jamie10, Jamie11, Vinnie1",
    "frequency_hz": 135,
    "amplitude_uA": 25,
    "pulse_width_us": 100,
    "exposure_ms": "1, 2",
    "stimulation_time_s": 10,
    "phase": "Post",
    "imaging_side": "Left",
    "fiber_placement_contains": None,
    "date": None,
    "block": None,
    "protocol_contains": None,
    "exclude_blocks": "Jamie12/06-04-26/R4, Jamie8/09-12-25/R5, Jamie8/19-12-25/R4, Jamie8/31-12-25/R1, Jamie12/01-04-26/R3",
    "allowed_n_trials": (5, 10),
}


# -----------------------------------------------------------------------------
# Averaging / output
# -----------------------------------------------------------------------------
AVERAGING_MODE = "equal_blocks"  # "equal_blocks" or "equal_trials"
SAVE_FIGURE = False
FIG_DPI = 300
SAVE_INCLUDED_BLOCKS = False
SHOW_FIGURE = True
SHOW_INCLUDED_BLOCKS = False
SHOW_EPHYS_CHECK_SUMMARY = False
# True = print slower ephys/data-derived metadata check
STIM_UA_PER_V = 10.0  # rough scale: 2.5 V command = 25 uA


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------
PLOT_FULL_TRACE = True
PLOT_STIM_TRACE = False
PLOT_SINGLE_PTA = False  # False/off, True/"normal", "derivative", or "both"
PLOT_MEAN_PTA = True
PLOT_LFP = False
PLOT_VELOCITY = False
PLOT_SPECTROGRAM = True
PLOT_LFP_SPECTROGRAM = False
PLOT_PULSOGRAM = False
PLOT_PLV_HISTOGRAMS = False  # 1=stim frequency, 2=2x stim, 3=3x stim; e.g. "1+2+3"
PLOT_SIGNAL_HILBERT = True
PLOT_SIGNAL_HILBERT_HARMONICS = "1"  # e.g. "1", "2", "1+2+3", or "5+6"
PLOT_LFP_HILBERT = False
PLOT_VM_SUMMARY = False


# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------
SPECTROGRAM_MODE = "relative"  # "absolute" or "relative"
SPECTROGRAM_SCALE = "db"  # "db" or "linear"
SPECTROGRAM_ABS_CMAP = "magma"
SPECTROGRAM_REL_CMAP = "RdBu_r"
SPECTROGRAM_FMAX_HZ = 250.0
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
VM_EARLY_WINDOW_S = (0.0, 0.2)
VM_LATE_WINDOW_S = (5.0, 10.0)
MPTA_PEAK_SEARCH_WINDOW_S = (0.0, 0.005)

PULSOGRAM_CMAP = "RdBu_r"
PULSOGRAM_TIME_RANGE_MS = (-10.0, 10.0)
STIM_TRACE_DISPLAY_MAX_POINTS = 12000
USE_DECIMATED_LFP = True  # False = plot full 30 kHz LFP traces


NUMERIC_TOLERANCE = {
    "frequency_hz": 0.6,
    "amplitude_uA": 1.0,
    "pulse_width_us": 1.0,
    "exposure_ms": 0.2,
    "stimulation_time_s": 0.25,
}

SPECIAL_AMPLITUDE_CLASSES = {"theta", "ramp", "up", "down", "cortex", "assymetric", "baseline"}
CORTEX_PROTOCOLS = {"40hz m1", "20hz (5s) m1", "m1"}


@dataclass
class BlockSpec:
    row: dict[str, str]
    summary_path: Path
    _summary_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def mouse(self) -> str:
        return self.row.get("mouse", "")

    @property
    def date(self) -> str:
        return self.row.get("date", "")

    @property
    def block(self) -> str:
        return self.row.get("block", "")

    @property
    def protocol(self) -> str:
        return self.row.get("protocol_raw", "") or self.row.get("protocol", "")

    @property
    def frequency_hz(self) -> float | None:
        return safe_float(self.row.get("frequency_hz"))

    @property
    def amplitude_uA(self) -> float | None:
        return safe_float(self.row.get("amplitude_uA"))

    @property
    def amplitude_label(self) -> str:
        raw = str(self.row.get("amplitude_uA", "")).strip()
        return raw if raw else "?"

    @property
    def pulse_width_us(self) -> float | None:
        return safe_float(self.row.get("pulse_width_us"))

    @property
    def exposure_ms(self) -> float | None:
        return safe_float(self.row.get("exposure_ms"))

    @property
    def stimulation_time_s(self) -> float | None:
        return safe_float(self.row.get("stimulation_time_s"))

    @property
    def phase(self) -> str:
        raw = str(self.row.get("phase", "")).strip()
        return raw if raw else "?"

    @property
    def imaging_side(self) -> str:
        raw = str(self.row.get("imaging_side", "")).strip()
        return raw if raw else "?"

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.mouse, self.date, self.block)

    @property
    def label(self) -> str:
        return f"{self.mouse} | {self.date} | {self.block}"

    def load_summary(self) -> dict[str, Any]:
        if self._summary_cache is None:
            with self.summary_path.open("rb") as f:
                self._summary_cache = pickle.load(f)
        return self._summary_cache


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text == "?":
        return None
    text = text.replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def gevi_display(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * float(GEVI_DISPLAY_SCALE)


def finite_values(values) -> np.ndarray:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


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


def normalize_text_label(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_mouse_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def flip_left_right_label(value: Any) -> str:
    label = normalize_text_label(value)
    if label == "left":
        return "right"
    if label == "right":
        return "left"
    return label


def effective_imaging_side_for_mouse(requested_side: Any, mouse_name: Any) -> str:
    requested = normalize_text_label(requested_side)
    if normalize_mouse_label(mouse_name) in IMAGING_SIDE_FLIP_MICE:
        return flip_left_right_label(requested)
    return requested


def imaging_side_matches(row: dict[str, str], requested_side: Any) -> bool:
    actual = normalize_text_label(row.get("imaging_side"))
    mouse = row.get("mouse")
    for requested in selection_values(requested_side):
        if actual == effective_imaging_side_for_mouse(requested, mouse):
            return True
    return False


def is_numeric_like(value: Any) -> bool:
    return safe_float(value) is not None


def normalize_class_label(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text_label(value))


def single_pta_mode() -> str:
    if isinstance(PLOT_SINGLE_PTA, bool):
        return "normal" if PLOT_SINGLE_PTA else "off"
    mode = normalize_class_label(PLOT_SINGLE_PTA)
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


def selection_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [item for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text and not re.fullmatch(r"[-+]?\d+,\d+", text):
            return [part.strip() for part in text.split(",") if part.strip()]
    return [value]


def format_selection_value(value: Any, unit: str = "") -> str:
    if is_numeric_like(value):
        text = trim_float(safe_float(value))
        return f"{text} {unit}".strip()
    return str(value).strip()


def format_selection(value: Any, unit: str = "") -> str:
    values = selection_values(value)
    if not values:
        return ""
    return ", ".join(format_selection_value(item, unit) for item in values)


def row_is_pure_baseline(row: dict[str, str]) -> bool:
    return (
        normalize_class_label(row.get("protocol")) == "baseline"
        and safe_float(row.get("frequency_hz")) is None
        and safe_float(row.get("amplitude_uA")) is None
        and safe_float(row.get("pulse_width_us")) is None
    )


def selected_amplitude_classes(value: Any) -> set[str]:
    return {
        label
        for label in (normalize_class_label(item) for item in selection_values(value))
        if label in SPECIAL_AMPLITUDE_CLASSES
    }


def row_amplitude_classes(row: dict[str, str]) -> set[str]:
    protocol = normalize_class_label(row.get("protocol"))
    protocol_raw = normalize_class_label(row.get("protocol_raw"))
    amplitude = normalize_class_label(row.get("amplitude_uA"))

    labels: set[str] = set()
    if row_is_pure_baseline(row):
        labels.add("baseline")
    if protocol == "thetaburst" or amplitude == "theta":
        labels.add("theta")
    if amplitude == "assymetric" or protocol == "assymetric":
        labels.add("assymetric")
    if protocol in {"amplitude ramping", "frequency ramping"}:
        labels.add("ramp")
    elif amplitude == "ramp" and not protocol.startswith("pulsewidth analysis"):
        labels.add("ramp")
    if protocol == "pulsewidth analysis ramp up":
        labels.add("up")
    if protocol == "pulsewidth analysis ramp down":
        labels.add("down")
    if protocol in CORTEX_PROTOCOLS or "m1-dbs" in protocol_raw or "cortex" in protocol_raw:
        labels.add("cortex")
    return labels


def natural_block_key(block_name: str) -> tuple[int, str]:
    match = re.search(r"R(\d+)", str(block_name))
    return (int(match.group(1)) if match else 10**9, str(block_name))


def date_key(date_text: str) -> tuple[int, int, int, str]:
    try:
        dt = datetime.strptime(date_text, "%d-%m-%y")
        return (dt.year, dt.month, dt.day, date_text)
    except ValueError:
        return (9999, 12, 31, date_text)


def mouse_key(mouse_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", mouse_name or "")
    return (int(match.group(1)) if match else 10**9, mouse_name)


def parse_name_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_exclude_blocks(value: str | None) -> list[tuple[str | None, str | None, str | None]]:
    if not value:
        return []
    specs: list[tuple[str | None, str | None, str | None]] = []
    for raw_item in str(value).split(","):
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


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "condition"


def summary_path_from_row(row: dict[str, str]) -> Path:
    mouse = row["mouse"]
    date = row["date"]
    block = row["block"]
    return DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block / f"{block}_summary.pkl"


def numeric_match(actual: float | None, target: float | None, key: str) -> bool:
    if target is None:
        return True
    if actual is None:
        return False
    tol = NUMERIC_TOLERANCE.get(key, 0.0)
    return abs(actual - target) <= tol


def categorical_or_numeric_match(actual_raw: Any, target: Any, key: str) -> bool:
    if target is None:
        return True
    for item in selection_values(target):
        if isinstance(item, str) and not is_numeric_like(item):
            if normalize_text_label(actual_raw) == normalize_text_label(item):
                return True
        elif numeric_match(safe_float(actual_raw), safe_float(item), key):
            return True
    return False


def text_contains_selection(actual_raw: Any, target: Any) -> bool:
    if target is None:
        return True
    actual = normalize_text_label(actual_raw)
    for item in selection_values(target):
        needle = normalize_text_label(item)
        if needle and needle in actual:
            return True
    return False


def threshold_amplitude_for_mouse(mouse_name: Any) -> float | None:
    return THRESHOLD_AMPLITUDE_UA_BY_MOUSE.get(normalize_mouse_label(mouse_name))


def is_threshold_amplitude_selection(value: Any) -> bool:
    return normalize_class_label(value) in THRESHOLD_AMPLITUDE_LABELS


def amplitude_matches(row: dict[str, str], target: Any) -> bool:
    if target is None:
        return True
    for item in selection_values(target):
        if is_threshold_amplitude_selection(item):
            threshold = threshold_amplitude_for_mouse(row.get("mouse"))
            if numeric_match(safe_float(row.get("amplitude_uA")), threshold, "amplitude_uA"):
                return True
        elif categorical_or_numeric_match(row.get("amplitude_uA"), item, "amplitude_uA"):
            return True
    return False


def row_matches_config(row: dict[str, str], mice: list[str], cfg: dict[str, Any]) -> bool:
    if mice and row.get("mouse", "") not in mice:
        return False
    date = cfg.get("date")
    block = cfg.get("block")
    protocol_contains = cfg.get("protocol_contains")
    phase = cfg.get("phase")
    imaging_side = cfg.get("imaging_side")
    fiber_placement_contains = cfg.get("fiber_placement_contains")
    amplitude_uA = cfg.get("amplitude_uA")
    if date and row.get("date") != date:
        return False
    if block and row.get("block") != block:
        return False
    if protocol_contains:
        hay = f"{row.get('protocol', '')} {row.get('protocol_raw', '')}".lower()
        if str(protocol_contains).lower() not in hay:
            return False
    if phase is not None:
        if normalize_text_label(row.get("phase")) != normalize_text_label(phase):
            return False
    if imaging_side is not None:
        if not imaging_side_matches(row, imaging_side):
            return False
    if fiber_placement_contains is not None:
        if not text_contains_selection(row.get("fiber_placement"), fiber_placement_contains):
            return False
    if not categorical_or_numeric_match(row.get("exposure_ms"), cfg.get("exposure_ms"), "exposure_ms"):
        return False
    row_classes = row_amplitude_classes(row)
    requested_classes = selected_amplitude_classes(amplitude_uA)
    if row_classes:
        return bool(row_classes & requested_classes)
    if not categorical_or_numeric_match(row.get("frequency_hz"), cfg.get("frequency_hz"), "frequency_hz"):
        return False
    if not amplitude_matches(row, amplitude_uA):
        return False
    if not categorical_or_numeric_match(row.get("pulse_width_us"), cfg.get("pulse_width_us"), "pulse_width_us"):
        return False
    if not categorical_or_numeric_match(row.get("stimulation_time_s"), cfg.get("stimulation_time_s"), "stimulation_time_s"):
        return False
    return True


def current_selection_config() -> dict[str, Any]:
    return {
        "mouse_name": MOUSE_NAME,
        "frequency_hz": FREQUENCY_HZ,
        "amplitude_uA": AMPLITUDE_UA,
        "pulse_width_us": PULSE_WIDTH_US,
        "exposure_ms": EXPOSURE_MS,
        "stimulation_time_s": STIMULATION_TIME_S,
        "phase": PHASE,
        "imaging_side": IMAGING_SIDE,
        "fiber_placement_contains": FIBER_PLACEMENT_CONTAINS,
        "date": DATE,
        "block": BLOCK,
        "protocol_contains": PROTOCOL_CONTAINS,
        "exclude_blocks": EXCLUDE_BLOCKS,
        "allowed_n_trials": ALLOWED_N_TRIALS,
    }


def row_matches(row: dict[str, str], mice: list[str]) -> bool:
    return row_matches_config(row, mice, current_selection_config())


def load_selected_blocks() -> tuple[list[BlockSpec], list[str]]:
    return load_selected_blocks_for_config(current_selection_config())


def load_selected_blocks_for_config(cfg: dict[str, Any]) -> tuple[list[BlockSpec], list[str]]:
    mice = parse_name_list(cfg.get("mouse_name"))
    exclude_specs = parse_exclude_blocks(cfg.get("exclude_blocks"))
    allowed_n_trials = cfg.get("allowed_n_trials")
    blocks: list[BlockSpec] = []
    skipped: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    with STIM_TABLE_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row_matches_config(row, mice, cfg):
                continue
            if row_is_excluded(row, exclude_specs):
                continue
            summary_path = summary_path_from_row(row)
            if not summary_path.exists():
                skipped.append(f"{row['mouse']} | {row['date']} | {row['block']} | missing summary")
                continue
            spec = BlockSpec(
                row=row,
                summary_path=summary_path,
            )
            if allowed_n_trials is not None:
                summary = spec.load_summary().get("summary", {})
                n_trials = int(
                    summary.get("single_pta", {}).get(
                        "n_trials",
                        len(spec.load_summary().get("trials", {}).get("stim_trial_names", [])),
                    )
                    or 0
                )
                allowed = {int(v) for v in allowed_n_trials}
                if n_trials not in allowed:
                    skipped.append(
                        f"{row['mouse']} | {row['date']} | {row['block']} | n_trials={n_trials} not in {sorted(allowed)}"
                    )
                    continue
            if spec.key in seen:
                continue
            seen.add(spec.key)
            blocks.append(spec)

    blocks.sort(key=lambda b: (mouse_key(b.mouse), date_key(b.date), natural_block_key(b.block)))
    return blocks, skipped


def condition_label(blocks: list[BlockSpec]) -> str:
    parts: list[str] = []
    mice = sorted({b.mouse for b in blocks}, key=mouse_key)
    if mice:
        parts.append(", ".join(mice))
    if FREQUENCY_HZ is not None:
        parts.append(format_selection(FREQUENCY_HZ, "Hz"))
    if PULSE_WIDTH_US is not None:
        parts.append(format_selection(PULSE_WIDTH_US, "us"))
    if AMPLITUDE_UA is not None:
        parts.append(format_selection(AMPLITUDE_UA, "uA"))
    if EXPOSURE_MS is not None:
        parts.append(f"exp {format_selection(EXPOSURE_MS, 'ms')}")
    if STIMULATION_TIME_S is not None:
        parts.append(f"stim {format_selection(STIMULATION_TIME_S, 's')}")
    if PHASE is not None:
        parts.append(PHASE)
    if IMAGING_SIDE is not None:
        parts.append(f"imaging {IMAGING_SIDE}")
    if FIBER_PLACEMENT_CONTAINS is not None:
        parts.append(f"fiber {FIBER_PLACEMENT_CONTAINS}")
    if DATE:
        parts.append(DATE)
    if BLOCK:
        parts.append(BLOCK)
    if PROTOCOL_CONTAINS:
        parts.append(PROTOCOL_CONTAINS)
    return " | ".join(parts) if parts else "Condition average"


def figure_title(blocks: list[BlockSpec]) -> str:
    if len(blocks) == 1:
        block = blocks[0]
        title = f"{block.mouse} | {block.date} {block.block} | block mean"
        f_stim = block_stim_frequency(block)
        if f_stim is not None and np.isfinite(f_stim):
            title += f" | {f_stim:.1f} Hz DBS"
        return title

    title = f"{condition_label(blocks)} | condition mean | n_blocks={len(blocks)}"
    if AVERAGING_MODE != "equal_blocks":
        title += f" | averaging={AVERAGING_MODE}"
    return title


def output_stem(blocks: list[BlockSpec]) -> str:
    mice = "-".join(sorted({b.mouse for b in blocks}, key=mouse_key)) or "allmice"
    parts = ["tableavg", mice]
    if FREQUENCY_HZ is not None:
        parts.append(format_selection(FREQUENCY_HZ, "Hz"))
    if PULSE_WIDTH_US is not None:
        parts.append(format_selection(PULSE_WIDTH_US, "us"))
    if AMPLITUDE_UA is not None:
        parts.append(format_selection(AMPLITUDE_UA, "uA"))
    if EXPOSURE_MS is not None:
        parts.append(f"exp{format_selection(EXPOSURE_MS, 'ms')}")
    if STIMULATION_TIME_S is not None:
        parts.append(f"stim{format_selection(STIMULATION_TIME_S, 's')}")
    if PHASE is not None:
        parts.append(str(PHASE).strip())
    if IMAGING_SIDE is not None:
        parts.append(f"img{str(IMAGING_SIDE).strip()}")
    if FIBER_PLACEMENT_CONTAINS is not None:
        parts.append(f"fiber{str(FIBER_PLACEMENT_CONTAINS).strip()}")
    if DATE:
        parts.append(DATE)
    if BLOCK:
        parts.append(BLOCK)
    if PROTOCOL_CONTAINS:
        parts.append(PROTOCOL_CONTAINS)
    return slugify("_".join(parts))


def block_output_stem(base_stem: str, block: BlockSpec) -> str:
    return slugify(f"{base_stem}_{block.mouse}_{block.date}_{block.block}")


def trim_float(value: float | None) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def median_float(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.nanmedian(clean)) if clean else None


def block_panel_weight(block: BlockSpec, panel: str) -> float:
    if AVERAGING_MODE == "equal_blocks":
        return 1.0
    summary = block.load_summary().get("summary", {})
    if panel in {"full_trace", "velocity"}:
        stim_names = block.load_summary().get("trials", {}).get("stim_trial_names", [])
        return float(len(stim_names) or 1)
    if panel == "single_pta":
        return float(summary.get("single_pta", {}).get("n_trials", 1) or 1)
    if panel == "lfp":
        return float(summary.get("lfp", {}).get("n_trials", 1) or 1)
    if panel in {"mean_pta", "spectrogram", "lfp_spectrogram", "lfp_hilbert"} or str(panel).startswith("signal_hilbert"):
        train = summary.get("train_pta", {})
        spec = train.get("spectrogram", {})
        return float(spec.get("n_trials_used", train.get("n_trials", 1)) or 1)
    if panel == "pulsogram":
        return float(summary.get("pulsogram", {}).get("n_trials", 1) or 1)
    return 1.0


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


def weighted_nanmean(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight_shape = (len(weights),) + (1,) * (stack.ndim - 1)
    w = weights.reshape(weight_shape)
    valid = np.isfinite(stack)
    num = np.nansum(np.where(valid, stack, 0.0) * w, axis=0)
    den = np.nansum(np.where(valid, 1.0, 0.0) * w, axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def weighted_nanstd(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight_shape = (len(weights),) + (1,) * (stack.ndim - 1)
    w = weights.reshape(weight_shape)
    valid = np.isfinite(stack)
    mean = weighted_nanmean(stack, weights)
    mean_shape = (1,) + mean.shape
    diff2 = np.where(valid, (stack - mean.reshape(mean_shape)) ** 2, 0.0)
    num = np.nansum(diff2 * w, axis=0)
    den = np.nansum(np.where(valid, 1.0, 0.0) * w, axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return np.sqrt(out)


def average_curve_panel(
    blocks: list[BlockSpec],
    panel_name: str,
    getter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[BlockSpec]] | None:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    weights: list[float] = []
    used: list[BlockSpec] = []

    for block in blocks:
        result = getter(block)
        if result is None:
            continue
        x, y = result
        if len(x) < 2 or len(y) != len(x):
            continue
        xs.append(np.asarray(x, float))
        ys.append(np.asarray(y, float))
        weights.append(block_panel_weight(block, panel_name))
        used.append(block)

    if not used:
        return None

    x_ref = build_common_axis_1d(xs)
    if x_ref is None:
        return None

    stack = np.vstack([interpolate_curve(x_ref, x, y) for x, y in zip(xs, ys)])
    weight_arr = np.asarray(weights, float)
    mean = weighted_nanmean(stack, weight_arr)
    sd = weighted_nanstd(stack, weight_arr)
    return x_ref, mean, sd, used


def get_single_pta_curve(block: BlockSpec):
    sec = block.load_summary().get("summary", {}).get("single_pta", {})
    if not sec.get("available", False):
        return None
    display = sec.get("display", {})
    x = np.asarray(display.get("t_rel_s", sec.get("t_rel_s", [])), float)
    y = np.asarray(display.get("mean", sec.get("mean", [])), float)
    if len(x) < 2 or y.shape != x.shape:
        return None
    return x, y


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


def get_single_pta_derivative_curve(block: BlockSpec):
    result = get_single_pta_curve(block)
    if result is None:
        return None
    return first_derivative_curve(*result)


def get_full_trace_curve(block: BlockSpec):
    proc = block.load_summary().get("summary", {}).get("processed_notched", {})
    sec = proc.get("stim", {})
    t = np.asarray(sec.get("t_common", []), float)
    y = np.asarray(sec.get("F_notched_mean", []), float)
    if len(t) < 2 or y.shape != t.shape:
        sec = proc.get("baseline", {})
        t = np.asarray(sec.get("t_common", []), float)
        y = np.asarray(sec.get("F_notched_mean", []), float)
    if len(t) < 2 or y.shape != t.shape:
        return None
    return t, y


def block_vm_window_value(block: BlockSpec, start_s: float, end_s: float) -> float:
    result = get_full_trace_curve(block)
    if result is None:
        return np.nan
    t, y = result
    n = min(len(t), len(y))
    t = np.asarray(t[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    if int(np.sum(good)) < 2:
        return np.nan
    t = t[good]
    y = y[good]
    baseline_vals = y[t < 0]
    baseline = float(np.nanmedian(baseline_vals)) if len(baseline_vals) else float(np.nanmedian(y))
    y_rel = y - baseline
    keep = (t >= float(start_s)) & (t <= float(end_s)) & np.isfinite(y_rel)
    return float(np.nanmedian(y_rel[keep])) if np.any(keep) else np.nan


def block_vm_summary_values(blocks: list[BlockSpec]) -> tuple[np.ndarray, np.ndarray]:
    early = np.asarray([block_vm_window_value(block, *VM_EARLY_WINDOW_S) for block in blocks], dtype=float)
    late = np.asarray([block_vm_window_value(block, *VM_LATE_WINDOW_S) for block in blocks], dtype=float)
    return finite_values(early), finite_values(late)


def get_stim_trace_curve(block: BlockSpec):
    summary = block.load_summary()
    stim_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    ephys_trials = summary.get("trials", {}).get("ephys", {})
    for name in stim_names:
        td_e = ephys_trials.get(name, {})
        t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
        y = np.asarray(td_e.get("channels", {}).get("stim", []), dtype=float)
        if len(t) >= 2 and y.shape == t.shape:
            return t, y
    return None


def get_mean_pta_curve(block: BlockSpec):
    sec = block.load_summary().get("summary", {}).get("train_pta", {})
    if not sec.get("available", False):
        return None
    display = sec.get("display", {})
    x = np.asarray(display.get("t_rel_s", sec.get("t_rel_s", [])), float)
    y = np.asarray(display.get("mean", sec.get("mean_across_trials", [])), float)
    if len(x) < 2 or y.shape != x.shape:
        return None
    return x, y


def get_lfp_curve(block: BlockSpec):
    sec = block.load_summary().get("summary", {}).get("lfp", {})
    if not sec.get("available", False):
        return None
    t_key = "t_stim_s_display" if USE_DECIMATED_LFP else "t_stim_s_full"
    mean_key = "mean_display" if USE_DECIMATED_LFP else "mean_full"
    x = np.asarray(sec.get(t_key, sec.get("t_stim_s_full", [])), float)
    y = np.asarray(sec.get(mean_key, sec.get("mean_full", [])), float)
    if len(x) < 2 or y.shape != x.shape:
        return None
    return x, y


def get_velocity_curve(block: BlockSpec):
    summary = block.load_summary()
    stim_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    ephys_trials = summary.get("trials", {}).get("ephys", {})
    if not stim_names:
        stim_names = sorted(str(x) for x in ephys_trials.keys())

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for name in stim_names:
        td_e = ephys_trials.get(name, {})
        t = np.asarray(td_e.get("vel_bin_t_s", []), dtype=float)
        y = np.asarray(td_e.get("vel_bin_cmps", []), dtype=float)
        good = np.isfinite(t) & np.isfinite(y)
        if np.sum(good) >= 2:
            xs.append(t[good])
            ys.append(y[good])

    x_ref = build_common_axis_1d(xs)
    if x_ref is None or not ys:
        return None

    stack = np.vstack([interpolate_curve(x_ref, x, y) for x, y in zip(xs, ys)])
    mean = weighted_nanmean(stack, np.ones(stack.shape[0], dtype=float))
    if len(x_ref) < 2 or mean.shape != x_ref.shape:
        return None
    return x_ref, mean


def get_hilbert_curve(block: BlockSpec, source: str):
    sec = block.load_summary().get("summary", {}).get("train_pta", {}).get(source, {})
    x = np.asarray(sec.get("time_s_display", sec.get("time_s_full", [])), float)
    if HILBERT_VIEW == "relative":
        y = np.asarray(sec.get("relative_mean_display", sec.get("relative_mean_full", [])), float)
    else:
        y = np.asarray(sec.get("amplitude_mean_display", sec.get("amplitude_mean_full", [])), float)
    if len(x) < 2 or y.shape != x.shape:
        return None
    return x, y


def get_block_plv_phases(block: BlockSpec, section_key: str = "plv") -> np.ndarray:
    summary = block.load_summary()
    phases = np.asarray(summary.get("summary", {}).get("train_pta", {}).get(section_key, {}).get("phase_pulses_rad", []), float)
    phases = phases[np.isfinite(phases)]
    if len(phases):
        return phases

    out = []
    for tr in summary.get("trials", {}).get("train_pta", {}).values():
        ph = np.asarray(tr.get(section_key, {}).get("phase_pulses_rad", []), float)
        ph = ph[np.isfinite(ph)]
        if len(ph):
            out.append(ph)
    return np.concatenate(out) if out else np.array([], dtype=float)


def phases_to_plv(phases: np.ndarray) -> tuple[float, float]:
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if len(phases) == 0:
        return np.nan, np.nan
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z))


def block_stim_frequency(block: BlockSpec) -> float | None:
    metrics = block.load_summary().get("summary", {}).get("train_pta", {}).get("metrics", {})
    return safe_float(metrics.get("f_stim_hz_mean")) or block.frequency_hz


def block_single_pta_second_pulse(block: BlockSpec) -> float | None:
    sec = block.load_summary().get("summary", {}).get("single_pta", {}).get("display", {})
    return safe_float(sec.get("second_pulse_rel_s_mean"))


def block_train_pta_second_pulse(block: BlockSpec) -> float | None:
    sec = block.load_summary().get("summary", {}).get("train_pta", {}).get("display", {})
    return safe_float(sec.get("second_pulse_rel_s_mean"))


def block_last_pulse_time(block: BlockSpec) -> float | None:
    ephys_trials = block.load_summary().get("trials", {}).get("ephys", {})
    stim_names = block.load_summary().get("trials", {}).get("stim_trial_names", [])
    last_pulses = []
    for name in stim_names:
        td_e = ephys_trials.get(name, {})
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        pulse_times = pulse_times[np.isfinite(pulse_times)]
        if len(pulse_times):
            last_pulses.append(float(pulse_times[-1]))
    return float(np.nanmedian(last_pulses)) if last_pulses else None


def pulse_train_metrics_from_times(pulse_times: np.ndarray) -> dict[str, float | None]:
    pulse_times = np.asarray(pulse_times, dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < 2:
        return {"frequency_hz": None, "stim_time_s": None, "n_pulses": float(len(pulse_times))}

    ipi = np.diff(pulse_times)
    ipi = ipi[np.isfinite(ipi) & (ipi > 0)]
    if len(ipi) == 0:
        return {"frequency_hz": None, "stim_time_s": None, "n_pulses": float(len(pulse_times))}

    median_ipi = float(np.nanmedian(ipi))
    return {
        "frequency_hz": 1.0 / median_ipi if median_ipi > 0 else None,
        "stim_time_s": float(pulse_times[-1] - pulse_times[0] + median_ipi),
        "n_pulses": float(len(pulse_times)),
    }


def stim_waveform_metrics_from_trace(t: np.ndarray, y: np.ndarray, pulse_times: np.ndarray) -> dict[str, float | None]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    pulse_times = np.asarray(pulse_times, dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(t) < 3 or y.shape != t.shape or len(pulse_times) == 0:
        return {"amp_v": None, "pulse_width_us": None}

    dt = median_float(list(np.diff(t)))
    if dt is None or dt <= 0:
        return {"amp_v": None, "pulse_width_us": None}

    pre = y[t < pulse_times[0] - 0.002]
    baseline = float(np.nanmedian(pre)) if len(pre) else float(np.nanmedian(y))
    ipi = np.diff(pulse_times)
    ipi = ipi[np.isfinite(ipi) & (ipi > 0)]
    half_window = 0.002
    if len(ipi):
        half_window = min(half_window, 0.4 * float(np.nanmedian(ipi)))

    amps_v = []
    widths_us = []
    step = max(1, len(pulse_times) // 80)
    for pulse_time in pulse_times[::step]:
        keep = (t >= pulse_time - half_window) & (t <= pulse_time + half_window)
        if np.sum(keep) < 3:
            continue
        yw_abs = np.abs(y[keep] - baseline)
        if not np.any(np.isfinite(yw_abs)):
            continue
        peak_i = int(np.nanargmax(yw_abs))
        peak = float(yw_abs[peak_i])
        if not np.isfinite(peak) or peak <= 0:
            continue

        above = yw_abs >= 0.5 * peak
        left = peak_i
        right = peak_i
        while left > 0 and above[left - 1]:
            left -= 1
        while right + 1 < len(above) and above[right + 1]:
            right += 1

        amps_v.append(peak)
        widths_us.append(float((right - left + 1) * dt * 1e6))

    return {
        "amp_v": median_float(amps_v),
        "pulse_width_us": median_float(widths_us),
    }


def imaging_frame_interval_ms(block: BlockSpec) -> float | None:
    trials = block.load_summary().get("trials", {}).get("processed_notched", {})
    frame_ms = []
    for td in trials.values():
        fps = safe_float(td.get("fps_hz"))
        if fps is not None and fps > 0:
            frame_ms.append(1000.0 / fps)
            continue
        t = np.asarray(td.get("t", []), dtype=float)
        t = t[np.isfinite(t)]
        if len(t) >= 2:
            frame_ms.append(float(np.nanmedian(np.diff(t)) * 1000.0))
    return median_float(frame_ms)


def block_ephys_check_summary(block: BlockSpec) -> dict[str, float | None]:
    summary = block.load_summary()
    stim_names = [str(x) for x in summary.get("trials", {}).get("stim_trial_names", [])]
    ephys_trials = summary.get("trials", {}).get("ephys", {})

    freqs = []
    stim_times = []
    pulse_counts = []
    amps_v = []
    widths_us = []

    for name in stim_names:
        td_e = ephys_trials.get(name, {})
        pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
        train = pulse_train_metrics_from_times(pulse_times)
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
        "n_trials": float(len(stim_names)),
        "frequency_hz": median_float(freqs),
        "stim_time_s": median_float(stim_times),
        "amp_v": amp_v,
        "amp_uA_est": None if amp_v is None else amp_v * STIM_UA_PER_V,
        "pulse_width_us": median_float(widths_us),
        "exposure_ms": imaging_frame_interval_ms(block),
        "pulses_per_trial": median_float(pulse_counts),
    }


def common_grid(arrays: list[np.ndarray]) -> np.ndarray | None:
    return build_common_axis_1d(arrays)


def block_spectrogram_linear(block: BlockSpec, source: str = "spectrogram") -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sec = block.load_summary().get("summary", {}).get("train_pta", {}).get(source, {})
    t = np.asarray(sec.get("time_s", []), float)
    f = np.asarray(sec.get("freq_hz", []), float)
    if len(t) < 2 or len(f) < 2:
        return None

    if SPECTROGRAM_MODE == "relative":
        z = sec.get("relative_linear_mean")
        if z is None and sec.get("relative_db_mean") is not None:
            z = 10.0 ** (np.asarray(sec["relative_db_mean"], float) / 10.0)
    else:
        z = sec.get("power_linear_mean")
        if z is None and sec.get("power_db_mean") is not None:
            z = 10.0 ** (np.asarray(sec["power_db_mean"], float) / 10.0)

    if z is None:
        return None

    z = np.asarray(z, float)
    if z.shape != (len(f), len(t)):
        return None
    return t, f, z


def average_spectrogram_blocks(
    blocks: list[BlockSpec],
    source: str = "spectrogram",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[BlockSpec], float | None, float | None] | None:
    items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    weights: list[float] = []
    used: list[BlockSpec] = []

    for block in blocks:
        result = block_spectrogram_linear(block, source=source)
        if result is None:
            continue
        t, f, z = result
        items.append((t, f, z))
        weights.append(block_panel_weight(block, source))
        used.append(block)

    if not used:
        return None

    t_ref = common_grid([item[0] for item in items])
    f_ref = common_grid([item[1] for item in items])
    if t_ref is None or f_ref is None:
        return None
    if SPECTROGRAM_FMAX_HZ is not None:
        f_ref = f_ref[f_ref <= SPECTROGRAM_FMAX_HZ]
    if len(t_ref) < 2 or len(f_ref) < 2:
        return None

    ff, tt = np.meshgrid(f_ref, t_ref, indexing="ij")
    pts = np.column_stack([ff.ravel(), tt.ravel()])
    stack = []
    for t, f, z in items:
        interpolator = RegularGridInterpolator((f, t), z, bounds_error=False, fill_value=np.nan)
        stack.append(interpolator(pts).reshape(len(f_ref), len(t_ref)))
    stack_arr = np.stack(stack, axis=0)
    mean_linear = weighted_nanmean(stack_arr, np.asarray(weights, float))

    if SPECTROGRAM_SCALE == "db":
        display = 10.0 * np.log10(np.maximum(mean_linear, 1e-12))
    else:
        display = mean_linear

    f_values = [block_stim_frequency(b) for b in used if block_stim_frequency(b) is not None]
    f_mean = float(np.nanmean(f_values)) if f_values else None
    return t_ref, f_ref, display, used, f_mean


def block_pulsogram_matrix(block: BlockSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sec = block.load_summary().get("summary", {}).get("pulsogram", {}).get("heatmap", {})
    pulse_numbers = np.asarray(sec.get("pulse_numbers", []), int)
    t_rel = np.asarray(sec.get("t_rel_s", []), float)
    matrix = np.asarray(sec.get("mean", []), float)
    if len(pulse_numbers) == 0 or len(t_rel) == 0 or matrix.shape != (len(pulse_numbers), len(t_rel)):
        return None

    lo_ms, hi_ms = PULSOGRAM_TIME_RANGE_MS
    mask = (t_rel * 1000.0 >= lo_ms) & (t_rel * 1000.0 <= hi_ms)
    if np.sum(mask) < 2:
        return None
    return pulse_numbers, np.asarray(t_rel[mask], dtype=float), np.asarray(matrix[:, mask], dtype=float)


def average_pulsogram_blocks(
    blocks: list[BlockSpec],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[BlockSpec]] | None:
    items = []
    used = []
    weights = []
    for block in blocks:
        result = block_pulsogram_matrix(block)
        if result is None:
            continue
        items.append(result)
        used.append(block)
        weights.append(block_panel_weight(block, "pulsogram"))

    if not used:
        return None

    pulse_union = np.asarray(sorted({int(p) for item in items for p in item[0]}), int)
    t_ref = common_grid([item[1] for item in items])
    if t_ref is None or len(pulse_union) == 0:
        return None

    stack = []
    for pulse_numbers, t, matrix in items:
        aligned = np.full((len(pulse_union), len(t_ref)), np.nan, dtype=float)
        row_lookup = {int(p): i for i, p in enumerate(pulse_numbers)}
        for out_row, pulse_num in enumerate(pulse_union):
            if int(pulse_num) not in row_lookup:
                continue
            aligned[out_row] = interpolate_curve(t_ref, t, matrix[row_lookup[int(pulse_num)]])
        stack.append(aligned)

    stack_arr = np.stack(stack, axis=0)
    mean = weighted_nanmean(stack_arr, np.asarray(weights, float))
    return pulse_union, t_ref, mean, used


def summarise_unique(values: list[float | None]) -> str:
    cleaned = sorted({trim_float(v) for v in values if v is not None}, key=lambda x: (len(x), x))
    return ", ".join(cleaned) if cleaned else "?"


def summarise_unique_labels(values: list[str]) -> str:
    cleaned = sorted({str(v).strip() for v in values if str(v).strip() and str(v).strip() != "?"})
    return ", ".join(cleaned) if cleaned else "?"


def print_included_blocks(blocks: list[BlockSpec], skipped: list[str]) -> None:
    print("\nIncluded blocks:")
    for block in blocks:
        summary = block.load_summary().get("summary", {})
        single_n = summary.get("single_pta", {}).get("n_trials", "?")
        print(
            f"  {block.mouse} | {block.date} | {block.block} | "
            f"freq={trim_float(block.frequency_hz)} | amp={block.amplitude_label} | "
            f"PW={trim_float(block.pulse_width_us)} | exp={trim_float(block.exposure_ms)} | "
            f"stim={trim_float(block.stimulation_time_s)} | phase={block.phase} | "
            f"side={block.imaging_side} | n_trials={single_n}"
        )

    if SHOW_EPHYS_CHECK_SUMMARY:
        print("\nEphys/data-derived check:")
        print("  ampV comes from stim channel; amp~uA uses STIM_UA_PER_V.")
        print("  exp is derived from imaging frame interval/fps.")
        for block in blocks:
            check = block_ephys_check_summary(block)
            print(
                f"  {block.mouse} | {block.date} | {block.block} | "
                f"freq={trim_float(check['frequency_hz'])} | "
                f"ampV={trim_float(check['amp_v'])} | "
                f"amp~uA={trim_float(check['amp_uA_est'])} | "
                f"PW={trim_float(check['pulse_width_us'])} | "
                f"exp={trim_float(check['exposure_ms'])} | "
                f"stim={trim_float(check['stim_time_s'])} | "
                f"n_trials={trim_float(check['n_trials'])} | "
                f"pulses/trial={trim_float(check['pulses_per_trial'])}"
            )

    if skipped:
        print("\nSkipped rows:")
        for item in skipped[:20]:
            print(f"  {item}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    print("\nIncluded parameter spread:")
    print(f"  mice: {', '.join(sorted({b.mouse for b in blocks}, key=mouse_key))}")
    print(f"  frequencies: {summarise_unique([b.frequency_hz for b in blocks])}")
    print(f"  amplitudes: {summarise_unique_labels([b.amplitude_label for b in blocks])}")
    print(f"  pulse widths: {summarise_unique([b.pulse_width_us for b in blocks])}")
    print(f"  exposures: {summarise_unique([b.exposure_ms for b in blocks])}")
    print(f"  stim times: {summarise_unique([b.stimulation_time_s for b in blocks])}")
    print(f"  phases: {summarise_unique_labels([b.phase for b in blocks])}")
    print(f"  imaging sides: {summarise_unique_labels([b.imaging_side for b in blocks])}")


def save_included_blocks_csv(blocks: list[BlockSpec], out_csv: Path) -> None:
    fields = [
        "mouse",
        "date",
        "block",
        "protocol_raw",
        "frequency_hz",
        "amplitude_uA",
        "pulse_width_us",
        "exposure_ms",
        "stimulation_time_s",
        "phase",
        "imaging_side",
        "summary_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for block in blocks:
            writer.writerow(
                {
                    "mouse": block.mouse,
                    "date": block.date,
                    "block": block.block,
                    "protocol_raw": block.protocol,
                    "frequency_hz": trim_float(block.frequency_hz),
                    "amplitude_uA": block.amplitude_label,
                    "pulse_width_us": trim_float(block.pulse_width_us),
                    "exposure_ms": trim_float(block.exposure_ms),
                    "stimulation_time_s": trim_float(block.stimulation_time_s),
                    "phase": block.phase,
                    "imaging_side": block.imaging_side,
                    "summary_path": str(block.summary_path),
                }
            )


def spectrogram_display_settings(z: np.ndarray, source: str = "gevi") -> tuple[str, dict[str, Any]]:
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        cmap = SPECTROGRAM_REL_CMAP if SPECTROGRAM_MODE == "relative" else SPECTROGRAM_ABS_CMAP
        return cmap, {}
    if SPECTROGRAM_MODE == "relative":
        if SPECTROGRAM_SCALE_MODE == "manual":
            if source == "lfp":
                vmin, vmax = [float(v) for v in LFP_SPECTROGRAM_REL_DB_RANGE]
            else:
                vmin, vmax = [float(v) for v in GEVI_SPECTROGRAM_REL_DB_RANGE]
            center = 0.0 if SPECTROGRAM_SCALE == "db" else 1.0
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                return SPECTROGRAM_REL_CMAP, {"norm": TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)}
        lo, hi = np.nanpercentile(finite, SPECTROGRAM_REL_PERCENTILES)
        center = 0.0 if SPECTROGRAM_SCALE == "db" else 1.0
        span = max(abs(lo - center), abs(hi - center))
        if span > 0:
            return SPECTROGRAM_REL_CMAP, {"norm": TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)}
        return SPECTROGRAM_REL_CMAP, {}
    lo, hi = np.nanpercentile(finite, SPECTROGRAM_ABS_PERCENTILES)
    return SPECTROGRAM_ABS_CMAP, {"vmin": float(lo), "vmax": float(hi)}


def set_tight_xlim(ax: plt.Axes, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))


def set_tight_curve_xlim(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.any(keep):
        ax.set_xlim(float(np.nanmin(x[keep])), float(np.nanmax(x[keep])))


def enabled_panel_defs() -> list[str]:
    panel_defs = []
    if PLOT_FULL_TRACE:
        panel_defs.append("full_trace")
    if PLOT_VM_SUMMARY:
        panel_defs.append("vm_summary")
    if single_pta_mode() != "off":
        panel_defs.append("single_pta")
    if PLOT_MEAN_PTA:
        panel_defs.append("mean_pta")
    if PLOT_SPECTROGRAM:
        panel_defs.append("spectrogram")
    for harmonic in parse_harmonic_selection(PLOT_PLV_HISTOGRAMS):
        panel_defs.append(f"plv_h{harmonic}_histogram")
    if PLOT_PULSOGRAM:
        panel_defs.append("pulsogram")
    if PLOT_STIM_TRACE:
        panel_defs.append("stim_trace")
    if PLOT_LFP:
        panel_defs.append("lfp")
    if PLOT_VELOCITY:
        panel_defs.append("velocity")
    if PLOT_LFP_SPECTROGRAM:
        panel_defs.append("lfp_spectrogram")
    if PLOT_SIGNAL_HILBERT:
        for harmonic in parse_harmonic_selection(PLOT_SIGNAL_HILBERT_HARMONICS):
            panel_defs.append(signal_hilbert_section_key(harmonic))
    if PLOT_LFP_HILBERT:
        panel_defs.append("lfp_hilbert")
    if not panel_defs:
        raise RuntimeError("No panels are enabled.")
    return panel_defs


def add_panel_letters(
    fig: plt.Figure,
    axes: np.ndarray | list[plt.Axes],
    ncols: int,
    *,
    letter_offset: int = 0,
) -> None:
    axes = list(np.atleast_1d(axes).ravel())
    fig.canvas.draw()
    x_offset = 0.08
    y_offset = 0.045
    y_max = 0.985

    col_x: dict[int, float] = {}
    row_y: dict[int, float] = {}
    for i, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        row, col = divmod(i, ncols)
        bbox = ax.get_position()
        x = bbox.x0 - x_offset * bbox.width
        y = min(y_max, bbox.y1 + y_offset * bbox.height)
        if ax.name != "polar":
            col_x.setdefault(col, x)
            row_y.setdefault(row, y)

    for i, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        row, col = divmod(i, ncols)
        bbox = ax.get_position()
        x = col_x.get(col, bbox.x0 - x_offset * bbox.width)
        y = row_y.get(row, min(y_max, bbox.y1 + y_offset * bbox.height))
        fig.text(
            x,
            y,
            chr(97 + letter_offset + i),
            transform=fig.transFigure,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def plot_condition_average(
    blocks: list[BlockSpec],
    out_png: Path,
    *,
    fig: plt.Figure | None = None,
    axes: np.ndarray | list[plt.Axes] | None = None,
    panel_defs: list[str] | None = None,
    letter_offset: int = 0,
    save_and_show: bool = True,
) -> None:
    panel_defs = enabled_panel_defs() if panel_defs is None else panel_defs

    if fig is None or axes is None:
        ncols = min(3, len(panel_defs))
        nrows = int(math.ceil(len(panel_defs) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 2.6 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(panel_defs):]:
            ax.remove()
    else:
        axes = np.atleast_1d(axes).ravel()
        ncols = len(panel_defs)
        if len(axes) < len(panel_defs):
            raise RuntimeError("Not enough axes for enabled panels.")

    for i, (ax, panel_name) in enumerate(zip(axes, panel_defs)):
        if panel_name.startswith("plv_h") and panel_name.endswith("_histogram"):
            spec = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(spec, projection="polar")
            axes[i] = ax

        if panel_name == "full_trace":
            result = average_curve_panel(blocks, "full_trace", get_full_trace_curve)
            if result is None:
                ax.set_visible(False)
                continue
            x, mean, sd, used = result
            mean = gevi_display(mean)
            sd = gevi_display(sd)
            stim_off_vals = [block_last_pulse_time(b) for b in used if block_last_pulse_time(b) is not None]
            ax.plot(x, mean, color="tab:blue", lw=1.6)
            ax.fill_between(x, mean - sd, mean + sd, color="tab:blue", alpha=0.18)
            ax.axvline(0, color="red", ls="--", lw=1)
            if stim_off_vals:
                ax.axvline(float(np.nanmedian(stim_off_vals)), color="tab:orange", ls="--", lw=1)
            ax.set_title("GEVI signal")
            ax.set_ylabel(GEVI_YLABEL)
            ax.set_xlabel("time from stim onset (s)")
            set_tight_curve_xlim(ax, x, mean)

        elif panel_name == "stim_trace":
            block_ref = blocks[0] if blocks else None
            result = None if block_ref is None else get_stim_trace_curve(block_ref)
            if result is None:
                ax.set_visible(False)
                continue
            x, y = result
            stim_off = None if block_ref is None else block_last_pulse_time(block_ref)
            ax.plot(x, y, color="tab:purple", lw=1.0)
            ax.axvline(0, color="red", ls="--", lw=1)
            if stim_off is not None and np.isfinite(stim_off):
                ax.axvline(float(stim_off), color="tab:orange", ls="--", lw=1)
            ax.set_title("DBS pulse train")
            ax.set_ylabel("stim (V)")
            ax.set_xlabel("time from stim onset (s)")
            set_tight_curve_xlim(ax, x, mean)

        elif panel_name == "single_pta":
            mode = single_pta_mode()
            result = average_curve_panel(blocks, "single_pta", get_single_pta_curve)
            deriv_result = average_curve_panel(blocks, "single_pta", get_single_pta_derivative_curve) if mode in {"derivative", "both"} else None
            if (mode in {"normal", "both"} and result is None) or (mode == "derivative" and deriv_result is None):
                ax.set_visible(False)
                continue
            used = result[3] if result is not None else deriv_result[3]
            second_vals = [block_single_pta_second_pulse(b) for b in used if block_single_pta_second_pulse(b) is not None]
            second_pulse_s = float(np.nanmedian(second_vals)) if second_vals else None
            if mode in {"normal", "both"}:
                x, mean, sd, _used = result
                mean = gevi_display(mean)
                sd = gevi_display(sd)
                ax.plot(x, mean, color="black", lw=1.8)
                ax.fill_between(x, mean - sd, mean + sd, color="0.7", alpha=0.35)
                set_tight_curve_xlim(ax, x, mean)
            if mode == "derivative":
                x, mean, sd, _used = deriv_result
                mean = gevi_display(mean)
                sd = gevi_display(sd)
                ax.plot(x, mean, color="tab:green", lw=1.8)
                ax.fill_between(x, mean - sd, mean + sd, color="tab:green", alpha=0.18)
                ax.axhline(0, color="0.6", lw=0.8)
                ax.set_ylabel(GEVI_DERIV_YLABEL)
                set_tight_curve_xlim(ax, x, mean)
            elif mode == "both" and deriv_result is not None:
                xd, dmean, _dsd, _used = deriv_result
                dmean = gevi_display(dmean)
                ax2 = ax.twinx()
                ax2.plot(xd, dmean, color="tab:green", lw=1.4)
                ax2.axhline(0, color="tab:green", lw=0.7, alpha=0.5)
                ax2.set_ylabel(GEVI_DERIV_YLABEL, color="tab:green")
                ax2.tick_params(axis="y", labelcolor="tab:green")
            ax.axvline(0, color="red", ls="--", lw=1)
            if second_pulse_s is not None:
                ax.axvline(second_pulse_s, color="tab:orange", ls="--", lw=1)
            ax.set_title("First-pulse response" if mode != "derivative" else "First-pulse derivative")
            ax.set_xlabel("time from first pulse (s)")
            ax.set_ylabel(GEVI_DERIV_YLABEL if mode == "derivative" else GEVI_YLABEL)
            set_tight_curve_xlim(ax, x, mean)

        elif panel_name == "mean_pta":
            result = average_curve_panel(blocks, "mean_pta", get_mean_pta_curve)
            if result is None:
                ax.set_visible(False)
                continue
            x, mean, sd, used = result
            mean = gevi_display(mean)
            sd = gevi_display(sd)
            second_vals = [block_train_pta_second_pulse(b) for b in used if block_train_pta_second_pulse(b) is not None]
            second_pulse_s = float(np.nanmedian(second_vals)) if second_vals else None
            ax.plot(x, mean, color="black", lw=1.8)
            ax.fill_between(x, mean - sd, mean + sd, color="0.7", alpha=0.35)
            ax.axvline(0, color="red", ls="--", lw=1)
            if second_pulse_s is not None:
                ax.axvline(second_pulse_s, color="tab:orange", ls="--", lw=1)
            ax.set_title("Pulse-triggered response")
            ax.set_ylabel(GEVI_YLABEL)
            ax.set_xlabel("time from pulse (s)")
            set_tight_curve_xlim(ax, x, mean)

        elif panel_name == "lfp":
            result = average_curve_panel(blocks, "lfp", get_lfp_curve)
            if result is None:
                ax.set_visible(False)
                continue
            x, mean, sd, used = result
            stim_off_vals = [block_last_pulse_time(b) for b in used if block_last_pulse_time(b) is not None]
            ax.plot(x, mean, color="tab:green", lw=1.8)
            ax.fill_between(x, mean - sd, mean + sd, color="tab:green", alpha=0.18)
            ax.axvline(0, color="red", ls="--", lw=1)
            if stim_off_vals:
                ax.axvline(float(np.nanmedian(stim_off_vals)), color="tab:orange", ls="--", lw=1)
            ax.set_title("LFP signal")
            ax.set_ylabel("LFP (a.u.)")
            ax.set_xlabel("time from stim onset (s)")
            set_tight_xlim(ax, x)

        elif panel_name == "velocity":
            result = average_curve_panel(blocks, "velocity", get_velocity_curve)
            if result is None:
                ax.set_visible(False)
                continue
            x, mean, sd, used = result
            stim_off_vals = [block_last_pulse_time(b) for b in used if block_last_pulse_time(b) is not None]
            ax.plot(x, mean, color="tab:brown", lw=1.8)
            ax.fill_between(x, mean - sd, mean + sd, color="tab:brown", alpha=0.18)
            ax.axvline(0, color="red", ls="--", lw=1)
            if stim_off_vals:
                ax.axvline(float(np.nanmedian(stim_off_vals)), color="tab:orange", ls="--", lw=1)
            ax.set_title("Wheel velocity")
            ax.set_ylabel("velocity (cm/s)")
            ax.set_xlabel("time from stim onset (s)")
            set_tight_xlim(ax, x)

        elif panel_name == "spectrogram":
            result = average_spectrogram_blocks(blocks, source="spectrogram")
            if result is None:
                ax.set_visible(False)
                continue
            t, f, z, used, f_mean = result
            cmap, color_kwargs = spectrogram_display_settings(z, source="gevi")
            image = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[t[0], t[-1], f[0], f[-1]],
                cmap=cmap,
                interpolation=SPECTROGRAM_INTERPOLATION,
                **color_kwargs,
            )
            ax.set_title("GEVI spectrogram")
            ax.set_ylabel("frequency (Hz)")
            ax.set_xlabel("time from stim onset (s)")
            ax.set_xlim(float(t[0]), float(t[-1]))
            cbar = fig.colorbar(image, ax=ax, pad=0.01)
            if SPECTROGRAM_MODE == "relative" and SPECTROGRAM_SCALE == "db":
                cbar.set_label("relative power (dB)")
            elif SPECTROGRAM_MODE == "relative":
                cbar.set_label("power / baseline")
            elif SPECTROGRAM_SCALE == "db":
                cbar.set_label("power (dB)")
            else:
                cbar.set_label("power (linear)")

        elif panel_name == "lfp_spectrogram":
            result = average_spectrogram_blocks(blocks, source="lfp_spectrogram")
            if result is None:
                ax.set_visible(False)
                continue
            t, f, z, used, _ = result
            cmap, color_kwargs = spectrogram_display_settings(z, source="lfp")
            image = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[t[0], t[-1], f[0], f[-1]],
                cmap=cmap,
                interpolation=SPECTROGRAM_INTERPOLATION,
                **color_kwargs,
            )
            ax.set_title("LFP spectrogram")
            ax.set_ylabel("frequency (Hz)")
            ax.set_xlabel("time from stim onset (s)")
            ax.set_xlim(float(t[0]), float(t[-1]))
            cbar = fig.colorbar(image, ax=ax, pad=0.01)
            if SPECTROGRAM_MODE == "relative" and SPECTROGRAM_SCALE == "db":
                cbar.set_label("relative power (dB)")
            elif SPECTROGRAM_MODE == "relative":
                cbar.set_label("power / baseline")
            elif SPECTROGRAM_SCALE == "db":
                cbar.set_label("power (dB)")
            else:
                cbar.set_label("power (linear)")

        elif panel_name == "pulsogram":
            result = average_pulsogram_blocks(blocks)
            if result is None:
                ax.set_visible(False)
                continue
            pulse_numbers, t, matrix, used = result
            matrix = gevi_display(matrix)
            finite = matrix[np.isfinite(matrix)]
            if finite.size:
                vmax = float(np.nanpercentile(np.abs(finite), 99))
            else:
                vmax = 1.0
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=[t[0] * 1000.0, t[-1] * 1000.0, pulse_numbers[0], pulse_numbers[-1]],
                cmap=PULSOGRAM_CMAP,
                vmin=-vmax,
                vmax=vmax,
            )
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.set_title("Pulse-by-pulse response")
            ax.set_ylabel("pulse #")
            ax.set_xlabel("time (ms)")
            ax.set_xlim(float(t[0] * 1000.0), float(t[-1] * 1000.0))
            fig.colorbar(image, ax=ax, pad=0.01, label=GEVI_YLABEL)

        elif panel_name.startswith("plv_h") and panel_name.endswith("_histogram"):
            m = re.match(r"plv_h(\d+)_histogram", panel_name)
            harmonic = int(m.group(1)) if m else 1
            section_key = plv_section_key(harmonic)
            label = plv_label(harmonic)
            phase_sets = [get_block_plv_phases(block, section_key=section_key) for block in blocks]
            phase_sets = [ph for ph in phase_sets if len(ph)]
            if not phase_sets:
                ax.set_visible(False)
                continue
            centers = [
                safe_float(block.load_summary().get("summary", {}).get("train_pta", {}).get(section_key, {}).get("f_center_hz_mean"))
                for block in blocks
            ]
            centers = [v for v in centers if v is not None and np.isfinite(v)]
            phases = np.concatenate(phase_sets)
            plv, pref = phases_to_plv(phases)
            counts, edges = np.histogram(phases, bins=np.linspace(-np.pi, np.pi, 37))
            rmax = max(1.0, float(np.max(counts)))
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="tab:blue", alpha=0.45, edgecolor="white", linewidth=0.4)
            ax.annotate("", xy=(pref, plv * rmax), xytext=(0, 0), arrowprops=dict(color="crimson", lw=2.5, arrowstyle="->"))
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_ylim(0, rmax)
            radial_ticks = [tick for tick in ax.get_yticks() if np.isfinite(tick) and tick > 0 and tick <= rmax]
            if len(radial_ticks) >= 2:
                radial_ticks = [radial_ticks[0], radial_ticks[-1]]
            ax.set_yticks(radial_ticks)
            ax.set_yticklabels([])
            ax.set_title(label, fontsize=11, pad=2)

        elif panel_name.startswith("signal_hilbert"):
            source = panel_name
            m = re.match(r"signal_hilbert_h(\d+)$", panel_name)
            harmonic = int(m.group(1)) if m else 1
            result = average_curve_panel(blocks, source, lambda b, source=source: get_hilbert_curve(b, source))
            if result is None:
                ax.set_visible(False)
                continue
            x, mean, sd, used = result
            stim_off_vals = [block_last_pulse_time(b) for b in used if block_last_pulse_time(b) is not None]
            centers = [
                safe_float(b.load_summary().get("summary", {}).get("train_pta", {}).get(source, {}).get("f_center_hz_mean"))
                for b in used
            ]
            centers = [v for v in centers if v is not None and np.isfinite(v)]
            ax.plot(x, mean, color="tab:blue", lw=1.8)
            ax.fill_between(x, mean - sd, mean + sd, color="tab:blue", alpha=0.18)
            ax.axvline(0, color="red", ls="--", lw=1)
            if stim_off_vals:
                ax.axvline(float(np.nanmedian(stim_off_vals)), color="tab:orange", ls="--", lw=1)
            title = hilbert_panel_title(harmonic)
            ax.set_title(title)
            ax.set_ylabel("amp / baseline" if HILBERT_VIEW == "relative" else "amplitude")
            ax.set_xlabel("time from stim onset (s)")
            set_tight_xlim(ax, x)

        elif panel_name == "lfp_hilbert":
            result = average_curve_panel(blocks, "lfp_hilbert", lambda b: get_hilbert_curve(b, "lfp_hilbert"))
            if result is None:
                ax.set_visible(False)
                continue
            x, mean, sd, used = result
            stim_off_vals = [block_last_pulse_time(b) for b in used if block_last_pulse_time(b) is not None]
            centers = [
                safe_float(b.load_summary().get("summary", {}).get("train_pta", {}).get("lfp_hilbert", {}).get("f_center_hz_mean"))
                for b in used
            ]
            centers = [v for v in centers if v is not None and np.isfinite(v)]
            ax.plot(x, mean, color="tab:green", lw=1.8)
            ax.fill_between(x, mean - sd, mean + sd, color="tab:green", alpha=0.18)
            ax.axvline(0, color="red", ls="--", lw=1)
            if stim_off_vals:
                ax.axvline(float(np.nanmedian(stim_off_vals)), color="tab:orange", ls="--", lw=1)
            title = "LFP DBS-frequency amplitude"
            ax.set_title(title)
            ax.set_ylabel("amp / baseline" if HILBERT_VIEW == "relative" else "amplitude")
            ax.set_xlabel("time from stim onset (s)")
            set_tight_xlim(ax, x)

        elif panel_name == "vm_summary":
            early, late = block_vm_summary_values(blocks)
            groups = [gevi_display(early), gevi_display(late)]
            if not len(groups[0]) and not len(groups[1]):
                ax.set_visible(False)
                continue
            positions = np.array([1.0, 2.0])
            ax.boxplot(groups, positions=positions, widths=0.45, labels=["trans", "sust"], showfliers=False)
            colors = ["tab:blue", "tab:red"]
            for pos, vals, color in zip(positions, groups, colors):
                if len(vals):
                    jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=22, color=color, alpha=0.75, zorder=3)
            ax.axhline(0, color="0.25", lw=0.8)
            ax.set_xlim(0.45, 2.55)
            ax.set_title("Vm response")
            ax.set_ylabel(GEVI_YLABEL)

    add_panel_letters(fig, axes[:len(panel_defs)], ncols, letter_offset=letter_offset)

    if not save_and_show:
        return

    if SAVE_FIGURE:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(out_png, dpi=FIG_DPI)
        print(f"\nSaved figure: {out_png}")
    if SHOW_FIGURE:
        try:
            plt.show()
        finally:
            plt.close(fig)
    else:
        plt.close(fig)


def group_comparison_output_stem() -> str:
    label_a = str(GROUP_A.get("label", "Group A"))
    label_b = str(GROUP_B.get("label", "Group B"))
    return slugify(f"group_comparison_{label_a}_vs_{label_b}")


def sync_group_comparison_panel_scales(axes: np.ndarray) -> None:
    axes = np.asarray(axes, dtype=object)
    if axes.ndim != 2:
        return
    for col_idx in range(axes.shape[1]):
        col_axes = [
            ax for ax in axes[:, col_idx]
            if ax is not None and ax.get_visible() and (ax.has_data() or len(ax.images))
        ]
        if len(col_axes) < 2:
            continue

        if all(getattr(ax, "name", "") == "polar" for ax in col_axes):
            continue

        xlims = []
        for ax in col_axes:
            if getattr(ax, "name", "") == "polar":
                continue
            lo, hi = ax.get_xlim()
            if np.isfinite(lo) and np.isfinite(hi):
                xlims.append((float(lo), float(hi)))
        if xlims:
            lo = max(v[0] for v in xlims)
            hi = min(v[1] for v in xlims)
            if hi > lo:
                for ax in col_axes:
                    if getattr(ax, "name", "") != "polar":
                        ax.set_xlim(lo, hi)

        ylims = []
        for ax in col_axes:
            if getattr(ax, "name", "") == "polar":
                continue
            lo, hi = ax.get_ylim()
            if np.isfinite(lo) and np.isfinite(hi):
                ylims.append((float(lo), float(hi)))
        if ylims:
            lo = min(v[0] for v in ylims)
            hi = max(v[1] for v in ylims)
            if hi > lo:
                for ax in col_axes:
                    if getattr(ax, "name", "") != "polar":
                        ax.set_ylim(lo, hi)

        images = [image for ax in col_axes for image in ax.images]
        clims = []
        for image in images:
            lo, hi = image.get_clim()
            if np.isfinite(lo) and np.isfinite(hi):
                clims.append((float(lo), float(hi)))
        if clims:
            lo = min(v[0] for v in clims)
            hi = max(v[1] for v in clims)
            if lo < 0 < hi:
                span = max(abs(lo), abs(hi))
                lo, hi = -span, span
            if hi > lo:
                for image in images:
                    image.set_clim(lo, hi)


def plot_group_comparison_average(out_png: Path) -> None:
    group_cfgs = [
        (str(GROUP_A.get("label", "Group A")), GROUP_A),
        (str(GROUP_B.get("label", "Group B")), GROUP_B),
    ]
    group_blocks: list[tuple[str, list[BlockSpec], list[str]]] = []
    for label, cfg in group_cfgs:
        blocks, skipped = load_selected_blocks_for_config(cfg)
        if not blocks:
            raise SystemExit(f"No matching blocks with summary files were found for {label}.")
        group_blocks.append((label, blocks, skipped))

    print("\nComparison plot groups:")
    for label, blocks, skipped in group_blocks:
        print(f"  {label}: {len(blocks)} block(s)")
        for block in blocks:
            print(
                f"    {block.mouse} | {block.date} | {block.block} | "
                f"amp={block.amplitude_label} | side={block.imaging_side} | n_trials={block.load_summary().get('summary', {}).get('single_pta', {}).get('n_trials', '?')}"
            )
        if skipped:
            print(f"    skipped={len(skipped)}")

    panel_defs = enabled_panel_defs()
    nrows = len(group_blocks)
    ncols = len(panel_defs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 2.6 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)

    for row_idx, (label, blocks, _skipped) in enumerate(group_blocks):
        plot_condition_average(
            blocks,
            out_png,
            fig=fig,
            axes=axes[row_idx, :],
            panel_defs=panel_defs,
            letter_offset=row_idx * ncols,
            save_and_show=False,
        )

    sync_group_comparison_panel_scales(axes)

    if SAVE_FIGURE:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(out_png, dpi=FIG_DPI)
        print(f"\nSaved figure: {out_png}")
    if SHOW_FIGURE:
        try:
            plt.show()
        finally:
            plt.close(fig)
    else:
        plt.close(fig)


def metric_name_aliases(metric_name: str) -> list[str]:
    raw = str(metric_name or "").strip()
    key = re.sub(r"[\s-]+", "_", raw.lower())
    aliases = [raw, key]
    if key in {"plv", "plv_h1"}:
        aliases.insert(0, "plv_mean")
    else:
        match = re.fullmatch(r"plv_h(\d+)", key)
        if match:
            aliases.insert(0, f"plv_h{match.group(1)}_mean")
    out = []
    for item in aliases:
        if item and item not in out:
            out.append(item)
    return out


def pooled_plv_metric_value(block: BlockSpec, metric_name: str) -> float:
    key = re.sub(r"[\s-]+", "_", str(metric_name or "").strip().lower())
    section_key = None
    if key in {"plv_pooled", "pooled_plv", "plv_h1_pooled", "pooled_plv_h1"}:
        section_key = "plv"
    else:
        match = re.fullmatch(r"(?:plv_h(\d+)_pooled|pooled_plv_h(\d+))", key)
        if match:
            harmonic = match.group(1) or match.group(2)
            section_key = plv_section_key(int(harmonic))
    if section_key is None:
        return np.nan
    phases = get_block_plv_phases(block, section_key=section_key)
    plv, _ = phases_to_plv(phases)
    return float(plv) if np.isfinite(plv) else np.nan


def block_vm_metric_value(block: BlockSpec, metric_name: str) -> float:
    key = re.sub(r"[\s-]+", "_", str(metric_name or "").strip().lower())
    if key in {"vm_early", "early_vm", "vmearly", "earlyvm", "vm_early_median"}:
        return block_vm_window_value(block, *VM_EARLY_WINDOW_S)
    if key in {"vm_late", "late_vm", "vmlate", "latevm", "vm_late_median"}:
        return block_vm_window_value(block, *VM_LATE_WINDOW_S)
    return np.nan


def mpta_peak_metric_kind(metric_name: str) -> str | None:
    key = re.sub(r"[\s-]+", "_", str(metric_name or "").strip().lower())
    if key in {
        "mpta_lat",
        "mpta_latency",
        "mpta_latency_ms",
        "mpta_peak_latency",
        "mpta_peak_latency_ms",
        "peak_1_latency_ms_median",
    }:
        return "latency"
    if key in {
        "mpta_jit",
        "mpta_jitter",
        "mpta_jitter_ms",
        "mpta_peak_jitter",
        "mpta_peak_jitter_ms",
        "peak_1_jitter_ms",
    }:
        return "jitter"
    if key in {
        "mpta_amp",
        "mpta_amplitude",
        "mpta_peak_amp",
        "mpta_peak_amplitude",
        "mean_pta_amp",
        "mean_pta_amplitude",
        "train_pta_amp",
        "peak_1_amplitude_median",
    }:
        return "amplitude"
    return None


def mpta_highest_peak_from_trace(t_rel_s, y) -> tuple[float, float]:
    t = np.asarray(t_rel_s, dtype=float).ravel()
    signal = np.asarray(y, dtype=float).ravel()
    n = min(len(t), len(signal))
    if n == 0:
        return np.nan, np.nan
    t = t[:n]
    signal = signal[:n]
    start_s, stop_s = MPTA_PEAK_SEARCH_WINDOW_S
    keep = np.flatnonzero((t >= start_s) & (t <= stop_s) & np.isfinite(t) & np.isfinite(signal))
    if len(keep) == 0:
        return np.nan, np.nan
    peak_idx = keep[int(np.nanargmax(signal[keep]))]
    return float(t[peak_idx] * 1000.0), float(signal[peak_idx])


def block_mpta_peak_arrays(block: BlockSpec) -> tuple[np.ndarray, np.ndarray]:
    data = block.load_summary()
    latencies: list[float] = []
    amplitudes: list[float] = []

    trials = data.get("trials", {}).get("train_pta", {})
    if isinstance(trials, dict):
        for trial in trials.values():
            if not isinstance(trial, dict):
                continue
            latency_ms, amplitude = mpta_highest_peak_from_trace(trial.get("t_rel_s"), trial.get("pta_mean"))
            if np.isfinite(latency_ms) and np.isfinite(amplitude):
                latencies.append(latency_ms)
                amplitudes.append(amplitude)

    if latencies:
        return np.asarray(latencies, dtype=float), np.asarray(amplitudes, dtype=float)

    train = data.get("summary", {}).get("train_pta", {})
    latency_ms, amplitude = mpta_highest_peak_from_trace(train.get("t_rel_s"), train.get("mean_across_trials"))
    if np.isfinite(latency_ms) and np.isfinite(amplitude):
        return np.asarray([latency_ms], dtype=float), np.asarray([amplitude], dtype=float)
    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def block_mpta_average_peak(block: BlockSpec) -> tuple[float, float]:
    train = block.load_summary().get("summary", {}).get("train_pta", {})
    t_rel = train.get("t_rel_s")
    y = train.get("mean_across_trials")
    if len(np.asarray(y if y is not None else []).ravel()) == 0:
        display = train.get("display", {})
        t_rel = display.get("t_rel_s", t_rel)
        y = display.get("mean", y)
    return mpta_highest_peak_from_trace(t_rel, y)


def block_mpta_peak_metric_value(block: BlockSpec, metric_name: str) -> float:
    kind = mpta_peak_metric_kind(metric_name)
    if kind is None:
        return np.nan
    avg_latency, avg_amplitude = block_mpta_average_peak(block)
    if kind == "latency" and np.isfinite(avg_latency):
        return avg_latency
    if kind == "amplitude" and np.isfinite(avg_amplitude):
        return avg_amplitude
    latencies, amplitudes = block_mpta_peak_arrays(block)
    if kind == "latency":
        vals = latencies[np.isfinite(latencies)]
        return float(np.nanmedian(vals)) if len(vals) else np.nan
    if kind == "jitter":
        vals = latencies[np.isfinite(latencies)]
        return float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else np.nan
    if kind == "amplitude":
        vals = amplitudes[np.isfinite(amplitudes)]
        return float(np.nanmedian(vals)) if len(vals) else np.nan
    return np.nan


def block_pta_amp_metric_value(block: BlockSpec, metric_name: str) -> float:
    key = re.sub(r"[\s-]+", "_", str(metric_name or "").strip().lower())
    summary = block.load_summary().get("summary", {})
    if key in {"spta_amp", "spta_mp", "spta_amplitude", "single_pta_amp", "single_pta_amplitude"}:
        value = safe_float(summary.get("single_pta", {}).get("latency", {}).get("peak_1_amplitude"))
        return value if value is not None else np.nan
    if key in {"mpta_amp", "mpta_amplitude", "mean_pta_amp", "mean_pta_amplitude", "train_pta_amp"}:
        value = safe_float(summary.get("train_pta", {}).get("latency", {}).get("peak_1_amplitude_median"))
        if value is None:
            value = safe_float(summary.get("train_pta", {}).get("metrics", {}).get("peak_1_amplitude_median"))
        return value if value is not None else np.nan
    return np.nan


def block_metric_value(block: BlockSpec, metric_name: str) -> float:
    summary = block.load_summary().get("summary", {})
    vm_value = block_vm_metric_value(block, metric_name)
    if np.isfinite(vm_value):
        return vm_value
    mpta_peak_value = block_mpta_peak_metric_value(block, metric_name)
    if np.isfinite(mpta_peak_value):
        return mpta_peak_value
    pta_amp_value = block_pta_amp_metric_value(block, metric_name)
    if np.isfinite(pta_amp_value):
        return pta_amp_value
    pooled_plv = pooled_plv_metric_value(block, metric_name)
    if np.isfinite(pooled_plv):
        return pooled_plv
    metric_aliases = metric_name_aliases(metric_name)
    sections = [
        summary.get("train_pta", {}).get("metrics", {}),
        summary.get("train_pta", {}).get("latency", {}),
        summary.get("train_pta", {}).get("hilbert_entrainment", {}),
        summary.get("first_pta", {}).get("latency", {}),
        summary.get("first_pta", {}).get("latency_jitter", {}),
    ]
    for metrics in sections:
        if not isinstance(metrics, dict):
            continue
        for key in metric_aliases:
            if key not in metrics:
                continue
            value = safe_float(metrics.get(key))
            return value if value is not None else np.nan

    entrainment_path = block.summary_path.with_name(f"{block.block}_entrainment_analysis.pkl")
    if entrainment_path.exists():
        with entrainment_path.open("rb") as f:
            entrainment = pickle.load(f)
        metrics = entrainment.get("metrics", {})
        if isinstance(metrics, dict):
            for key in metric_aliases:
                if key not in metrics:
                    continue
                value = safe_float(metrics.get(key))
                return value if value is not None else np.nan
    return np.nan


def format_median_iqr(vals: np.ndarray) -> str:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return "n=0"
    q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
    return f"median={med:.3g}, IQR={q1:.3g}-{q3:.3g}, n={len(vals)}"


def run_single_group_comparison(metric_name: str) -> None:
    blocks_a, skipped_a = load_selected_blocks_for_config(GROUP_A)
    blocks_b, skipped_b = load_selected_blocks_for_config(GROUP_B)
    vals_a = np.asarray([block_metric_value(b, metric_name) for b in blocks_a], dtype=float)
    vals_b = np.asarray([block_metric_value(b, metric_name) for b in blocks_b], dtype=float)
    keep_a = np.isfinite(vals_a)
    keep_b = np.isfinite(vals_b)
    blocks_a = [b for b, keep in zip(blocks_a, keep_a) if keep]
    blocks_b = [b for b, keep in zip(blocks_b, keep_b) if keep]
    vals_a = vals_a[keep_a]
    vals_b = vals_b[keep_b]

    label_a = str(GROUP_A.get("label", "Group A"))
    label_b = str(GROUP_B.get("label", "Group B"))
    print(f"\nMetric: {metric_name}")
    print(f"{label_a}: {format_median_iqr(vals_a)}")
    print(f"{label_b}: {format_median_iqr(vals_b)}")

    if len(vals_a) and len(vals_b):
        res = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        print(f"Mann-Whitney U: U={float(res.statistic):.3g}, p={float(res.pvalue):.4g}")
    else:
        print("Mann-Whitney U: not enough finite values")

    print("\nIncluded A:")
    for block, value in zip(blocks_a, vals_a):
        print(f"  {block.label} | {metric_name}={value:.4g}")
    print("\nIncluded B:")
    for block, value in zip(blocks_b, vals_b):
        print(f"  {block.label} | {metric_name}={value:.4g}")
    if skipped_a or skipped_b:
        print("\nSkipped:")
        for item in skipped_a:
            print(f"  A | {item}")
        for item in skipped_b:
            print(f"  B | {item}")


def run_group_comparison() -> None:
    metrics = selection_values(GROUP_COMPARISON_METRIC)
    if not metrics:
        metrics = [GROUP_COMPARISON_METRIC]
    for metric in metrics:
        run_single_group_comparison(str(metric).strip())


def main() -> None:
    if COMPARE_GROUPS:
        run_group_comparison()

    if PLOT_GROUP_COMPARISON:
        out_png = FIGURES_DIR / f"{group_comparison_output_stem()}.png"
        plot_group_comparison_average(out_png)
        return

    blocks, skipped = load_selected_blocks()
    if not blocks:
        raise SystemExit("No matching blocks with summary files were found.")

    print_included_blocks(blocks, skipped)

    stem = output_stem(blocks)
    out_png = FIGURES_DIR / f"{stem}.png"
    out_csv = TABLES_DIR / f"{stem}_included_blocks.csv"

    if SAVE_INCLUDED_BLOCKS:
        save_included_blocks_csv(blocks, out_csv)
        print(f"\nSaved included-block table: {out_csv}")

    plot_condition_average(blocks, out_png)

    if SHOW_INCLUDED_BLOCKS:
        for block in blocks:
            print(f"\n[BLOCK VIEW] {block.label}")
            block_png = FIGURES_DIR / f"{block_output_stem(stem, block)}.png"
            plot_condition_average([block], block_png)


if __name__ == "__main__":
    main()
