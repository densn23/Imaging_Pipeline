from pathlib import Path
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, peak_widths, iirnotch, filtfilt
from config import DATA_ANALYSIS_ROOT

MOUSE_NAME = "Jamie10"  # change to "Jamie5", etc.
SINGLE_DATE = "15-12-25"
SINGLE_BLOCK = "R1"


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
        DATA_ANALYSIS_ROOT / _SINGLE_MOUSE_NAME / "Imaging_Data" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_traces_processed.pkl"
    )
)

# -------------------------
# ROOTS (batch mode)
# -------------------------
IMAGING_ROOT = Path(
    DATA_ANALYSIS_ROOT / (_SINGLE_MOUSE_NAME or str(MOUSE_NAME)) / "Imaging_Data"
)
BLOCK_RE = re.compile(r"^R\d+$")

# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = False
SAVE_OUTPUT = False
SHOW_PLOTS = True
SHOW_POST_NOTCH_PSD = False
SHOW_POST_NOTCH_BAND_POWER = False

# -------------------------
# SETTINGS
# -------------------------
SIGNAL_MODE = "dff"  # "dff", "raw", "bleach", "bleach_or_raw"
NEGATIVE_POLARITY_MICE = {"vinnie1", "vinnie2"}
TRIAL_LIMIT = None             # None = all trials

# Fixed notch settings
NOTCH_Q = 50.0
FIXED_NOTCH_FREQS = [114.0, 123.0, 124.0, 131.0]

# PSD settings
N_PERSEG = 1024
FREQ_MIN = 0.0
FREQ_MAX = None  # None => Nyquist
SEARCH_BAND_HZ = (100.0, 150.0)
MIN_PEAK_PROMINENCE_DB = 3.0
PEAK_WIDTH_REL_HEIGHT = 0.5
PREVIEW_TRIALS = 3
REL_POWER_REF_BAND_HZ = (1.0, 100.0)
POST_NOTCH_PLOT_BAND_HZ = (1.0, 100.0)
POST_NOTCH_NORMALIZE_PERCENT = True
BAND_DEFS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 55.0),
    "high_gamma": (65.0, 100.0),
}


def same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(a, b, equal_nan=True)


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def get_fs_hz(trial_dict: dict) -> float:
    fs = trial_dict.get("fps_hz", np.nan)
    if np.isfinite(fs) and fs > 0:
        return float(fs)

    t = np.asarray(trial_dict.get("t", []), dtype=float)
    if len(t) < 2:
        return np.nan
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return np.nan
    return 1.0 / dt


def choose_signal(trial_dict: dict, mode: str) -> np.ndarray | None:
    raw = trial_dict.get("F_raw", None)
    corr = trial_dict.get("F_bleach_corr", None)
    dff = trial_dict.get("dff", None)

    if mode == "dff":
        return None if dff is None else np.asarray(dff, dtype=float)
    if mode == "raw":
        return None if raw is None else np.asarray(raw, dtype=float)
    if mode == "bleach":
        return None if corr is None else np.asarray(corr, dtype=float)
    if mode == "bleach_or_raw":
        if corr is not None:
            return np.asarray(corr, dtype=float)
        return None if raw is None else np.asarray(raw, dtype=float)
    return None


def signal_polarity(mouse_name: str | None) -> float:
    key = re.sub(r"[^a-z0-9]+", "", str(mouse_name or "").lower())
    return -1.0 if key in NEGATIVE_POLARITY_MICE else 1.0


def notch_filter(x: np.ndarray, fs: float, freqs: list[float], q: float) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    for f0 in freqs:
        if f0 <= 0 or f0 >= fs / 2.0:
            continue
        b, a = iirnotch(f0, q, fs)
        y = filtfilt(b, a, y)
    return y



def build_harmonic_notch_freqs(base_freqs: list[float], nyq: float) -> list[float]:
    out = []
    for f0 in base_freqs:
        if not np.isfinite(f0) or f0 <= 0:
            continue
        k = 1
        while True:
            fk = k * float(f0)
            if fk >= nyq:
                break
            out.append(float(fk))
            k += 1
    return sorted(out)

def compute_psd_summary(trials: dict):
    trial_names = sorted(trials.keys())
    if TRIAL_LIMIT is not None:
        trial_names = trial_names[: int(TRIAL_LIMIT)]

    psd_list = []
    used_names = []
    f_ref = None
    fs_ref = None

    for name in trial_names:
        td = trials[name]
        x = choose_signal(td, SIGNAL_MODE)
        if x is None or len(x) < 8:
            continue
        fs = get_fs_hz(td)
        if not np.isfinite(fs) or fs <= 0:
            continue

        nperseg = min(N_PERSEG, len(x))
        f, p = welch(x, fs=fs, nperseg=nperseg, detrend="constant")

        if f_ref is None:
            f_ref = f
            fs_ref = fs
        else:
            if len(f) != len(f_ref) or not np.allclose(f, f_ref, rtol=1e-6, atol=1e-8):
                continue

        psd_list.append(p)
        used_names.append(name)

    if not psd_list:
        return None

    psd_arr = np.vstack(psd_list)
    psd_med = np.median(psd_arr, axis=0)
    psd_db = 10.0 * np.log10(np.maximum(psd_med, 1e-30))

    nyq = fs_ref / 2.0
    fmax = nyq if FREQ_MAX is None else min(float(FREQ_MAX), nyq)
    band_lo, band_hi = SEARCH_BAND_HZ
    band_hi = min(band_hi, nyq)

    band_mask = (f_ref >= band_lo) & (f_ref <= band_hi)
    band_db = psd_db[band_mask]
    band_f = f_ref[band_mask]

    peaks, props = find_peaks(band_db, prominence=MIN_PEAK_PROMINENCE_DB)
    detected = []
    if len(peaks) > 0:
        widths_samp, _, left_ips, right_ips = peak_widths(
            band_db, peaks, rel_height=PEAK_WIDTH_REL_HEIGHT
        )
        idx_axis = np.arange(len(band_f), dtype=float)
        for i, pk in enumerate(peaks):
            left_f = float(np.interp(left_ips[i], idx_axis, band_f))
            right_f = float(np.interp(right_ips[i], idx_axis, band_f))
            width_hz = right_f - left_f
            detected.append(
                (
                    float(band_f[pk]),
                    float(band_db[pk]),
                    float(props["prominences"][i]),
                    float(width_hz),
                )
            )
    detected.sort(key=lambda z: z[1], reverse=True)
    notch_candidates = build_harmonic_notch_freqs(FIXED_NOTCH_FREQS, nyq)

    return {
        "trial_names": trial_names,
        "used_names": used_names,
        "f_ref": f_ref,
        "psd_arr": psd_arr,
        "psd_db": psd_db,
        "nyq": nyq,
        "fmax": fmax,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_mask": band_mask,
        "band_f": band_f,
        "band_db": band_db,
        "detected": detected,
        "notch_candidates": notch_candidates,
        "fs_ref": fs_ref,
    }



def compute_post_notch_psd_summary(trials: dict, notch_freqs: list[float], notch_q: float):
    trial_names = sorted(trials.keys())
    if TRIAL_LIMIT is not None:
        trial_names = trial_names[: int(TRIAL_LIMIT)]

    psd_list = []
    used_names = []
    f_ref = None
    fs_ref = None

    for name in trial_names:
        td = trials[name]
        x = choose_signal(td, SIGNAL_MODE)
        if x is None or len(x) < 8:
            continue
        fs = get_fs_hz(td)
        if not np.isfinite(fs) or fs <= 0:
            continue

        y = notch_filter(x, fs, notch_freqs, notch_q)
        nperseg = min(N_PERSEG, len(y))
        f, p = welch(y, fs=fs, nperseg=nperseg, detrend="constant")

        if f_ref is None:
            f_ref = f
            fs_ref = fs
        else:
            if len(f) != len(f_ref) or not np.allclose(f, f_ref, rtol=1e-6, atol=1e-8):
                continue

        psd_list.append(p)
        used_names.append(name)

    if not psd_list:
        return None

    psd_arr = np.vstack(psd_list)
    psd_med = np.median(psd_arr, axis=0)
    psd_db = 10.0 * np.log10(np.maximum(psd_med, 1e-30))
    return {
        "used_names": used_names,
        "f_ref": f_ref,
        "psd_arr": psd_arr,
        "psd_db": psd_db,
        "fs_ref": fs_ref,
    }


def integrate_band_power(freq_hz: np.ndarray, psd_linear: np.ndarray, band_hz: tuple[float, float]) -> float:
    lo, hi = band_hz
    m = (freq_hz >= float(lo)) & (freq_hz <= float(hi))
    if np.sum(m) < 2:
        return np.nan
    return float(np.trapezoid(psd_linear[m], freq_hz[m]))


def compute_post_notch_trial_spectral(y: np.ndarray, fs: float) -> dict:
    if len(y) < 8 or not np.isfinite(fs) or fs <= 0:
        return {}

    nperseg = min(N_PERSEG, len(y))
    freq_hz, psd_linear = welch(y, fs=fs, nperseg=nperseg, detrend="constant")
    psd_db = 10.0 * np.log10(np.maximum(psd_linear, 1e-30))

    ref_power = integrate_band_power(freq_hz, psd_linear, REL_POWER_REF_BAND_HZ)
    band_powers = {}
    rel_band_powers = {}
    log_band_powers = {}
    for band_name, band_hz in BAND_DEFS_HZ.items():
        p_abs = integrate_band_power(freq_hz, psd_linear, band_hz)
        band_powers[band_name] = float(p_abs) if np.isfinite(p_abs) else np.nan
        rel_band_powers[band_name] = (
            float(p_abs / ref_power)
            if np.isfinite(p_abs) and np.isfinite(ref_power) and ref_power > 0
            else np.nan
        )
        log_band_powers[band_name] = (
            float(10.0 * np.log10(max(p_abs, 1e-30)))
            if np.isfinite(p_abs) and p_abs > 0
            else np.nan
        )

    return {
        "freq_hz": np.asarray(freq_hz, dtype=np.float64),
        "psd_linear": np.asarray(psd_linear, dtype=np.float64),
        "psd_db": np.asarray(psd_db, dtype=np.float64),
        "ref_band_hz": list(REL_POWER_REF_BAND_HZ),
        "ref_band_power": float(ref_power) if np.isfinite(ref_power) else np.nan,
        "band_defs_hz": {k: list(v) for k, v in BAND_DEFS_HZ.items()},
        "band_power_abs": band_powers,
        "band_power_rel": rel_band_powers,
        "band_power_db": log_band_powers,
    }


def summarize_post_notch_spectral(trials: dict) -> dict:
    spectra = []
    used_names = []
    freq_ref = None
    band_abs_by_name = {name: [] for name in BAND_DEFS_HZ}
    band_rel_by_name = {name: [] for name in BAND_DEFS_HZ}
    band_db_by_name = {name: [] for name in BAND_DEFS_HZ}

    for name in sorted(trials.keys()):
        spec = trials[name].get("spectral_post_notch", {})
        freq_hz = np.asarray(spec.get("freq_hz", []), dtype=float)
        psd_linear = np.asarray(spec.get("psd_linear", []), dtype=float)
        if freq_ref is None and len(freq_hz) and len(psd_linear):
            freq_ref = freq_hz.copy()
        if freq_ref is not None and len(freq_hz) and same_grid(freq_hz, freq_ref) and psd_linear.shape == freq_ref.shape:
            spectra.append(psd_linear)
            used_names.append(name)
        for band_name in BAND_DEFS_HZ:
            band_abs_by_name[band_name].append(safe_float(spec.get("band_power_abs", {}).get(band_name)))
            band_rel_by_name[band_name].append(safe_float(spec.get("band_power_rel", {}).get(band_name)))
            band_db_by_name[band_name].append(safe_float(spec.get("band_power_db", {}).get(band_name)))

    mean_abs = {k: safe_float(np.nanmean(v)) if v else np.nan for k, v in band_abs_by_name.items()}
    mean_rel = {k: safe_float(np.nanmean(v)) if v else np.nan for k, v in band_rel_by_name.items()}
    mean_db = {k: safe_float(np.nanmean(v)) if v else np.nan for k, v in band_db_by_name.items()}
    sd_abs = {k: safe_float(np.nanstd(v, ddof=1)) if len(v) >= 2 else np.nan for k, v in band_abs_by_name.items()}
    sd_rel = {k: safe_float(np.nanstd(v, ddof=1)) if len(v) >= 2 else np.nan for k, v in band_rel_by_name.items()}
    sd_db = {k: safe_float(np.nanstd(v, ddof=1)) if len(v) >= 2 else np.nan for k, v in band_db_by_name.items()}

    psd_mean = np.nanmean(np.vstack(spectra), axis=0) if spectra else np.array([], dtype=float)
    psd_sd = (
        np.nanstd(np.vstack(spectra), axis=0, ddof=1)
        if len(spectra) >= 2
        else (np.full(len(freq_ref), np.nan, dtype=float) if freq_ref is not None else np.array([], dtype=float))
    )

    return {
        "used_names": used_names,
        "freq_hz": freq_ref if freq_ref is not None else np.array([], dtype=float),
        "psd_linear_mean": np.asarray(psd_mean, dtype=np.float64),
        "psd_db_mean": 10.0 * np.log10(np.maximum(psd_mean, 1e-30)) if len(psd_mean) else np.array([], dtype=float),
        "psd_linear_sd": np.asarray(psd_sd, dtype=np.float64),
        "ref_band_hz": list(REL_POWER_REF_BAND_HZ),
        "band_defs_hz": {k: list(v) for k, v in BAND_DEFS_HZ.items()},
        "band_power_abs_mean": mean_abs,
        "band_power_abs_sd": sd_abs,
        "band_power_rel_mean": mean_rel,
        "band_power_rel_sd": sd_rel,
        "band_power_db_mean": mean_db,
        "band_power_db_sd": sd_db,
    }


def plot_post_notch_band_power(block_label: str, spectral_summary: dict):
    freq_hz = np.asarray(spectral_summary.get("freq_hz", []), dtype=float)
    psd_linear_mean = np.asarray(spectral_summary.get("psd_linear_mean", []), dtype=float)
    psd_linear_sd = np.asarray(spectral_summary.get("psd_linear_sd", []), dtype=float)
    if len(freq_hz) == 0 or len(psd_linear_mean) == 0:
        return

    lo, hi = POST_NOTCH_PLOT_BAND_HZ
    m = (freq_hz >= float(lo)) & (freq_hz <= float(hi))
    if np.sum(m) < 2:
        return

    x = freq_hz[m]
    y_mean = psd_linear_mean[m]
    y_sd = psd_linear_sd[m]

    ylabel = "power"
    title_suffix = "post-notch spectrum"
    if POST_NOTCH_NORMALIZE_PERCENT:
        ref_power = float(np.trapezoid(y_mean, x))
        if np.isfinite(ref_power) and ref_power > 0:
            y_mean = 100.0 * y_mean / ref_power
            y_sd = 100.0 * y_sd / ref_power
            ylabel = "normalized power (%)"
            title_suffix = f"post-notch normalized spectrum ({lo:.0f}-{hi:.0f} Hz)"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for band_name, (band_lo, band_hi) in BAND_DEFS_HZ.items():
        left = max(float(lo), float(band_lo))
        right = min(float(hi), float(band_hi))
        if right <= left:
            continue
        ax.axvspan(left, right, color="0.92", alpha=0.8, zorder=0)
        ax.text(
            0.5 * (left + right),
            0.98,
            band_name.replace("_", " "),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )

    ax.plot(x, y_mean, color="tab:blue", lw=2.0, label="mean")
    if np.any(np.isfinite(y_sd)):
        ax.fill_between(x, y_mean - y_sd, y_mean + y_sd, color="tab:blue", alpha=0.2, label="SD")
    ax.set_xlim(float(lo), float(hi))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{block_label} | {title_suffix}")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def apply_notch_to_trials(trials: dict, notch_freqs: list[float], notch_q: float, mouse_name: str | None = None):
    out_trials = {}
    skipped = []
    polarity = signal_polarity(mouse_name)
    for name, td in trials.items():
        td_out = dict(td)
        x = choose_signal(td, SIGNAL_MODE)
        fs = get_fs_hz(td)

        if x is None or not np.isfinite(fs) or fs <= 0:
            td_out["F_notched"] = None
            td_out["notch_applied"] = False
            td_out["spectral_post_notch"] = {}
            skipped.append(name)
        else:
            x = polarity * x
            td_out["F_notched"] = notch_filter(x, fs, notch_freqs, notch_q)
            td_out["notch_applied"] = True
            td_out["spectral_post_notch"] = compute_post_notch_trial_spectral(td_out["F_notched"], fs)

        td_out["notch_source_mode"] = SIGNAL_MODE
        td_out["signal_polarity"] = float(polarity)
        td_out["notch_q"] = float(notch_q)
        td_out["notch_freqs_hz"] = list(notch_freqs)
        out_trials[name] = td_out
    return out_trials, skipped


    f_ref = s["f_ref"]
    psd_arr = s["psd_arr"]
    psd_db = s["psd_db"]
    band_lo = s["band_lo"]
    band_hi = s["band_hi"]
    band_f = s["band_f"]
    band_db = s["band_db"]
    band_mask = s["band_mask"]
    notch_candidates = s["notch_candidates"]
    fmax = s["fmax"]

    n_rows = 3 if (SHOW_POST_NOTCH_PSD and s_post is not None) else 2
    fig, ax = plt.subplots(n_rows, 1, figsize=(11, 4 * n_rows))
    if n_rows == 1:
        ax = [ax]

    for p in psd_arr:
        ax[0].plot(f_ref, 10.0 * np.log10(np.maximum(p, 1e-30)), color="tab:blue", alpha=0.15, lw=0.8)
    ax[0].plot(f_ref, psd_db, color="black", lw=2.0, label="median PSD before notch")
    ax[0].set_xlim(FREQ_MIN, fmax)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("PSD (dB)")
    ax[0].set_title(f"Welch PSD across trials | {block_label}")
    ax[0].legend(loc="best")

    for p in psd_arr:
        p_db = 10.0 * np.log10(np.maximum(p, 1e-30))
        ax[1].plot(band_f, p_db[band_mask], color="tab:blue", alpha=0.15, lw=0.8)
    ax[1].plot(band_f, band_db, color="black", lw=2.0, label="median PSD (zoom)")
    for f0 in notch_candidates:
        ax[1].axvline(f0, color="tab:red", ls="--", alpha=0.8)
    ax[1].set_xlim(band_lo, band_hi)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("PSD (dB)")
    ax[1].set_title("Artifact band zoom")
    ax[1].legend(loc="best")

    if SHOW_POST_NOTCH_PSD and s_post is not None:
        f_post = s_post["f_ref"]
        psd_arr_post = s_post["psd_arr"]
        psd_db_post = s_post["psd_db"]
        for p in psd_arr_post:
            ax[2].plot(f_post, 10.0 * np.log10(np.maximum(p, 1e-30)), color="tab:green", alpha=0.15, lw=0.8)
        ax[2].plot(f_post, psd_db_post, color="darkgreen", lw=2.0, label="median PSD after notch")
        for f0 in notch_candidates:
            ax[2].axvline(f0, color="tab:red", ls="--", alpha=0.5)
        ax[2].set_xlim(FREQ_MIN, fmax)
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("PSD (dB)")
        ax[2].set_title("Full PSD after notch")
        ax[2].legend(loc="best")

    plt.tight_layout()
    plt.show()

    preview_names = s["used_names"][: min(PREVIEW_TRIALS, len(s["used_names"]))]
    if not preview_names or not notch_candidates:
        return

    fig2, ax2 = plt.subplots(len(preview_names), 1, figsize=(11, 3 * len(preview_names)))
    if len(preview_names) == 1:
        ax2 = [ax2]

    for i, name in enumerate(preview_names):
        td = trials[name]
        x = choose_signal(td, SIGNAL_MODE)
        fs = get_fs_hz(td)
        y = notch_filter(x, fs, notch_candidates, NOTCH_Q)

        nperseg = min(N_PERSEG, len(x))
        f0, p0 = welch(x, fs=fs, nperseg=nperseg, detrend="constant")
        f1, p1 = welch(y, fs=fs, nperseg=nperseg, detrend="constant")
        ax2[i].plot(f0, 10.0 * np.log10(np.maximum(p0, 1e-30)), lw=1.0, label="before")
        ax2[i].plot(f1, 10.0 * np.log10(np.maximum(p1, 1e-30)), lw=1.0, label="after notch")
        ax2[i].set_xlim(FREQ_MIN, fmax)
        for f_notch in notch_candidates:
            ax2[i].axvline(f_notch, color="tab:red", ls="--", alpha=0.6)
        ax2[i].set_title(f"{name} notch preview (Q={NOTCH_Q})")
        ax2[i].set_xlabel("Frequency (Hz)")
        ax2[i].set_ylabel("PSD (dB)")
        ax2[i].legend(loc="best")

    plt.tight_layout()
    plt.show()

def plot_psd_and_preview(block_label: str, trials: dict, s: dict, s_post: dict | None = None):
    f_ref = s["f_ref"]
    psd_arr = s["psd_arr"]
    psd_db = s["psd_db"]
    band_lo = s["band_lo"]
    band_hi = s["band_hi"]
    band_f = s["band_f"]
    band_db = s["band_db"]
    band_mask = s["band_mask"]
    notch_candidates = s["notch_candidates"]
    fmax = s["fmax"]

    n_rows = 3 if (SHOW_POST_NOTCH_PSD and s_post is not None) else 2
    fig, ax = plt.subplots(n_rows, 1, figsize=(11, 4 * n_rows))
    if n_rows == 1:
        ax = [ax]

    for p in psd_arr:
        ax[0].plot(f_ref, 10.0 * np.log10(np.maximum(p, 1e-30)), color="tab:blue", alpha=0.15, lw=0.8)
    ax[0].plot(f_ref, psd_db, color="black", lw=2.0, label="median PSD before notch")
    ax[0].set_xlim(FREQ_MIN, fmax)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("PSD (dB)")
    ax[0].set_title(f"Welch PSD across trials | {block_label}")
    ax[0].legend(loc="best")

    for p in psd_arr:
        p_db = 10.0 * np.log10(np.maximum(p, 1e-30))
        ax[1].plot(band_f, p_db[band_mask], color="tab:blue", alpha=0.15, lw=0.8)
    ax[1].plot(band_f, band_db, color="black", lw=2.0, label="median PSD (zoom)")
    for f0 in notch_candidates:
        ax[1].axvline(f0, color="tab:red", ls="--", alpha=0.8)
    ax[1].set_xlim(band_lo, band_hi)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("PSD (dB)")
    ax[1].set_title("Artifact band zoom")
    ax[1].legend(loc="best")

    if SHOW_POST_NOTCH_PSD and s_post is not None:
        f_post = s_post["f_ref"]
        psd_arr_post = s_post["psd_arr"]
        psd_db_post = s_post["psd_db"]
        for p in psd_arr_post:
            ax[2].plot(f_post, 10.0 * np.log10(np.maximum(p, 1e-30)), color="tab:green", alpha=0.15, lw=0.8)
        ax[2].plot(f_post, psd_db_post, color="darkgreen", lw=2.0, label="median PSD after notch")
        for f0 in notch_candidates:
            ax[2].axvline(f0, color="tab:red", ls="--", alpha=0.5)
        ax[2].set_xlim(FREQ_MIN, fmax)
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("PSD (dB)")
        ax[2].set_title("Full PSD after notch")
        ax[2].legend(loc="best")

    plt.tight_layout()
    plt.show()

    preview_names = s["used_names"][: min(PREVIEW_TRIALS, len(s["used_names"]))]
    if not preview_names or not notch_candidates:
        return

    fig2, ax2 = plt.subplots(len(preview_names), 1, figsize=(11, 3 * len(preview_names)))
    if len(preview_names) == 1:
        ax2 = [ax2]

    for i, name in enumerate(preview_names):
        td = trials[name]
        x = choose_signal(td, SIGNAL_MODE)
        fs = get_fs_hz(td)
        y = notch_filter(x, fs, notch_candidates, NOTCH_Q)

        nperseg = min(N_PERSEG, len(x))
        f0, p0 = welch(x, fs=fs, nperseg=nperseg, detrend="constant")
        f1, p1 = welch(y, fs=fs, nperseg=nperseg, detrend="constant")
        ax2[i].plot(f0, 10.0 * np.log10(np.maximum(p0, 1e-30)), lw=1.0, label="before")
        ax2[i].plot(f1, 10.0 * np.log10(np.maximum(p1, 1e-30)), lw=1.0, label="after notch")
        ax2[i].set_xlim(FREQ_MIN, fmax)
        for f_notch in notch_candidates:
            ax2[i].axvline(f_notch, color="tab:red", ls="--", alpha=0.6)
        ax2[i].set_title(f"{name} notch preview (Q={NOTCH_Q})")
        ax2[i].set_xlabel("Frequency (Hz)")
        ax2[i].set_ylabel("PSD (dB)")
        ax2[i].legend(loc="best")

    plt.tight_layout()
    plt.show()

def run_single(processed_path: Path):
    with open(processed_path, "rb") as f:
        d = pickle.load(f)

    trials = d.get("trials", {})
    if not trials:
        print(f"[SKIP] no trials in {processed_path}")
        return

    s = compute_psd_summary(trials)
    if s is None:
        print(f"[SKIP] no valid PSD trials in {processed_path}")
        return

    notch_candidates = s["notch_candidates"]
    print(f"[RUN] {d.get('date')} {d.get('block')} | {processed_path.name}")
    print(f"[INFO] Trials used for PSD: {len(s['used_names'])} / {len(s['trial_names'])}")
    print(f"[INFO] fs_hz={s['fs_ref']:.3f}, Nyquist={s['nyq']:.3f}")
    print(f"[INFO] Notch freqs used (Hz): {notch_candidates} | Q={NOTCH_Q}")
    if s["detected"]:
        print("[INFO] Detected peaks (freq_hz, power_db, prominence_db, approx_width_hz):")
        for f0, pw, prm, whz in s["detected"]:
            print(f"  {f0:8.3f} Hz | {pw:8.3f} dB | prom {prm:6.3f} dB | width {whz:6.3f} Hz")
        mean_width = float(np.mean([w for _, _, _, w in s["detected"]]))
        print(f"[INFO] Mean peak width in search band: {mean_width:.3f} Hz")
    else:
        print("[INFO] No peaks detected in search band.")

    s_post = compute_post_notch_psd_summary(trials, notch_candidates, NOTCH_Q) if SHOW_POST_NOTCH_PSD else None

    if SHOW_PLOTS:
        block_label = f"{d.get('date')} {d.get('block')}"
        plot_psd_and_preview(block_label, trials, s, s_post)


    mouse_name = d.get("mouse")
    out_trials, skipped = apply_notch_to_trials(trials, notch_candidates, NOTCH_Q, mouse_name=mouse_name)
    if skipped:
        print(f"[INFO] Notch skipped for {len(skipped)} trial(s): {', '.join(skipped[:8])}" +
              (" ..." if len(skipped) > 8 else ""))
    spectral_summary_post = summarize_post_notch_spectral(out_trials)

    if SHOW_PLOTS and SHOW_POST_NOTCH_BAND_POWER:
        block_label = f"{d.get('date')} {d.get('block')}"
        plot_post_notch_band_power(block_label, spectral_summary_post)

    if SAVE_OUTPUT:
        out_path = processed_path.parent / f"{processed_path.stem}_notched.pkl"
        out_dict = dict(d)
        out_dict["trials"] = out_trials
        out_dict["notch_processing"] = {
            "signal_mode": SIGNAL_MODE,
            "signal_polarity": float(signal_polarity(mouse_name)),
            "notch_q": float(NOTCH_Q),
            "notch_freqs_hz": notch_candidates,
            "spectral_ref_band_hz": list(REL_POWER_REF_BAND_HZ),
            "spectral_band_defs_hz": {k: list(v) for k, v in BAND_DEFS_HZ.items()},
            "source_file": str(processed_path),
        }
        out_dict["spectral_summary_post_notch"] = spectral_summary_post
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] {out_path}")


def run_batch(root: Path):
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
                continue
            p = block_dir / f"{block_dir.name}_traces_processed.pkl"
            if not p.exists():
                continue
            run_single(p)


def run_single_date(root: Path, date_name: str):
    date_dir = root / date_name
    if not date_dir.exists():
        print(f"[WARN] date folder not found: {date_dir}")
        return

    for block_dir in sorted(date_dir.iterdir()):
        if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
            continue
        p = block_dir / f"{block_dir.name}_traces_processed.pkl"
        if not p.exists():
            continue
        run_single(p)


def main() -> None:
    mouse_names = resolve_mouse_names(MOUSE_NAME)
    if not mouse_names:
        print("No mice found to process.")
        return

    if RUN_BATCH:
        for mouse_name in mouse_names:
            run_batch(DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data")
    elif SINGLE_DATE is not None and SINGLE_BLOCK is None:
        for mouse_name in mouse_names:
            run_single_date(DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data", SINGLE_DATE)
    else:
        if SINGLE_DATE is None or SINGLE_BLOCK is None:
            print("Set SINGLE_DATE to run one date, or set both SINGLE_DATE and SINGLE_BLOCK to run one block.")
            return
        for mouse_name in mouse_names:
            single_processed = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_traces_processed.pkl"
            if not single_processed.exists():
                print(f"[SKIP] missing single-block input for {mouse_name} | {SINGLE_DATE} | {SINGLE_BLOCK}")
                continue
            run_single(single_processed)


if __name__ == "__main__":
    main()








