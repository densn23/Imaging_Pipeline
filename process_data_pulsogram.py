from pathlib import Path
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import csv
from config import DATA_ANALYSIS_ROOT

MOUSE_NAME = "Jamie10"
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
IMAGING_ROOT = DATA_ANALYSIS_ROOT / (_SINGLE_MOUSE_NAME or str(MOUSE_NAME)) / "Imaging_Data"
EPHYS_ROOT = DATA_ANALYSIS_ROOT / (_SINGLE_MOUSE_NAME or str(MOUSE_NAME)) / "Open_Ephys"
BLOCK_RE = re.compile(r"^R\d+$")

# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = False
SAVE_OUTPUT = False
SHOW_PLOTS = True
SHOW_HEATMAP = True
SHOW_OVERPLOTTED_LINES = False
PLOT_PULSE_STRIDE = 1  # 1 = every pulse, 5 = every 5th pulse, etc.

# -------------------------
# SETTINGS
# -------------------------
SIGNAL_MODE = "notched"  # "notched_or_bleach_or_raw", "notched", "bleach", "raw"
PRE_SEC = 0.010
POST_SEC = 0.030
POST_WINDOW_MODE = "freq_aware"  # "fixed" or "freq_aware"
POST_WINDOW_SCALE = 1.0     # In freq_aware mode, use this many post-pulse periods on the right; left side is baseline only
PERIOD_FRACTION = 1.0        # Always use one full pulse period left/right in the pulsogram view
BASELINE_MODE = "median_pre"  # "none", "median_pre", "mean_pre"
MIN_PULSES = 5
SPREAD_MODE = "sd"  # "sd" or "sem"
ONLY_TRIAL = None  # e.g. "R1_1"
PRINT_FIRST_N_PULSES = 0
SAVE_CSV = False

# Peak window for pulsogram metrics
PEAK_WINDOW_S = (0.000, 0.007)


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


def estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return np.nan
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return 1.0 / dt


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
    if POST_WINDOW_MODE != "freq_aware":
        return float(PRE_SEC), float(POST_SEC)

    ipi = estimate_ipi_s(pulse_times)
    if not np.isfinite(ipi) or ipi <= 0:
        return float(PRE_SEC), float(POST_SEC)

    frac = min(1.0, max(0.0, float(PERIOD_FRACTION)))
    win_s = frac * ipi
    return float(win_s), float(win_s)


def baseline_correct_segment(seg: np.ndarray, rel_grid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return seg
    pre = seg[rel_grid < 0]
    if len(pre) == 0:
        return seg
    if mode == "mean_pre":
        b = float(np.mean(pre))
    else:
        b = float(np.median(pre))
    return seg - b


def build_rel_grid(t: np.ndarray, pre_s: float, post_s: float) -> np.ndarray | None:
    fs = estimate_fs(t)
    if not np.isfinite(fs):
        return None
    dt = 1.0 / fs
    n_half_pre = max(1, int(round(float(pre_s) / dt)))
    n_half_post = max(1, int(round(float(post_s) / dt)))
    n = int(n_half_pre + n_half_post + 1)
    if n < 4 or not np.isfinite(pre_s) or not np.isfinite(post_s):
        return None
    return np.linspace(-float(pre_s), float(post_s), n, dtype=float)


def extract_pulse_segment_dict(t: np.ndarray, x: np.ndarray, tp: float, pre_s: float, post_s: float):
    t_rel = t - float(tp)
    keep = (t_rel >= -pre_s) & (t_rel <= post_s)
    if np.sum(keep) < 3:
        return None
    t_seg = t_rel[keep]
    x_seg = x[keep]
    x_seg = baseline_correct_segment(x_seg, t_seg, BASELINE_MODE)
    return {
        "t_rel_s": np.asarray(t_seg, dtype=np.float64),
        "signal": np.asarray(x_seg, dtype=np.float64),
    }


def build_freq_aware_folded_segment_dict(
    t: np.ndarray,
    x: np.ndarray,
    pulse_times: np.ndarray,
    pulse_idx: int,
    pre_s: float,
    post_s: float,
):
    rel_grid = build_rel_grid(t, pre_s, post_s)
    if len(rel_grid) < 4:
        return None

    n_periods = max(1, int(round(float(POST_WINDOW_SCALE))))
    if pulse_idx < 1 or (pulse_idx + n_periods) >= len(pulse_times):
        return None

    pre_mask = rel_grid < 0
    post_mask = rel_grid >= 0
    if not np.any(pre_mask) or not np.any(post_mask):
        return None

    pre_grid = rel_grid[pre_mask]
    post_grid = rel_grid[post_mask]
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
    y = baseline_correct_segment(y, rel_grid, BASELINE_MODE)
    return {
        "t_rel_s": np.asarray(rel_grid, dtype=np.float64),
        "signal": np.asarray(y, dtype=np.float64),
    }


def build_common_grid_from_segments(segments: list[dict], pre_s: float, post_s: float) -> np.ndarray | None:
    valid = [np.asarray(s.get("t_rel_s", []), dtype=float) for s in segments if len(s.get("t_rel_s", [])) >= 2]
    if not valid:
        return None

    ref = valid[0]
    if all(tr.shape == ref.shape and np.allclose(tr, ref, equal_nan=True) for tr in valid[1:]):
        return ref.copy()

    dts = []
    for tr in valid:
        dt = np.diff(tr)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            dts.append(float(np.median(dt)))
    if not dts:
        return None

    dt_med = float(np.median(dts))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return None

    lo = max(float(np.nanmin(tr)) for tr in valid)
    hi = min(float(np.nanmax(tr)) for tr in valid)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    grid = np.arange(lo, hi + 0.5 * dt_med, dt_med, dtype=float)
    return grid if len(grid) >= 4 else None


def interpolate_segment_dicts(segments: list[dict], t_grid: np.ndarray) -> np.ndarray:
    Y = np.full((len(segments), len(t_grid)), np.nan, dtype=float)
    for i, s in enumerate(segments):
        tx = s["t_rel_s"]
        yx = s["signal"]
        if len(tx) < 2:
            continue
        m = (t_grid >= tx[0]) & (t_grid <= tx[-1])
        if np.any(m):
            Y[i, m] = np.interp(t_grid[m], tx, yx)
    return Y


def extract_all_pulse_segment_dicts(t: np.ndarray, x: np.ndarray, pulse_times: np.ndarray, pre_s: float, post_s: float):
    segments = []
    used_idx = []
    for k, tp in enumerate(pulse_times):
        if POST_WINDOW_MODE == "freq_aware":
            seg = build_freq_aware_folded_segment_dict(t, x, pulse_times, k, pre_s, post_s)
        else:
            seg = extract_pulse_segment_dict(t, x, tp, pre_s, post_s)
        if seg is None:
            continue
        segments.append(seg)
        used_idx.append(int(k))
    return segments, used_idx


def nansem(y: np.ndarray, axis=0):
    n = np.sum(np.isfinite(y), axis=axis)
    sd = np.nanstd(y, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sd / np.sqrt(n)


def spread(y: np.ndarray) -> np.ndarray:
    return nansem(y, axis=0) if SPREAD_MODE == "sem" else np.nanstd(y, axis=0, ddof=1)


def detect_two_strongest(seg: np.ndarray, rel_grid: np.ndarray, win_s: tuple[float, float]):
    a, b = win_s
    idx = np.where((rel_grid >= a) & (rel_grid <= b))[0]
    if len(idx) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    vals = np.asarray(seg[idx], dtype=float)
    ts = np.asarray(rel_grid[idx], dtype=float)
    order = [int(i) for i in np.argsort(vals)[::-1]]

    j1 = order[0]
    amp1 = float(vals[j1])
    lat1 = float(ts[j1])

    # If another top candidate is adjacent and earlier, use the earlier sample as peak 1.
    for cand in order[1:]:
        if abs(cand - j1) == 1 and cand < j1:
            j1 = int(cand)
            amp1 = float(vals[j1])
            lat1 = float(ts[j1])
            break

    s1 = int(j1 + 1)

    j2 = None
    for cand in order:
        cand = int(cand)
        if cand <= j1:
            continue
        if abs(cand - j1) >= 2:
            j2 = cand
            break

    if j2 is None:
        return amp1, lat1, np.nan, np.nan, s1, np.nan

    amp2 = float(vals[j2])
    lat2 = float(ts[j2])
    s2 = int(j2 + 1)
    return amp1, lat1, amp2, lat2, s1, s2


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
    seg_dicts, used_idx = extract_all_pulse_segment_dicts(t, x, pulse_times, pre_s, post_s)
    if len(seg_dicts) < MIN_PULSES:
        return None, "too_few_valid_pulses"

    return {
        "trial": trial_name,
        "segment_dicts": seg_dicts,
        "pulse_times_s": pulse_times.astype(np.float64),
        "used_pulse_indices": np.asarray(used_idx, dtype=int),
        "n_pulses_used": int(len(used_idx)),
        "window_pre_s": float(pre_s),
        "window_post_s": float(post_s),
    }, None


def pool_pulse_metrics(trial_results: list[dict]):
    by_pulse: dict[int, list[dict]] = {}
    for tr in trial_results:
        segs = tr.get("segment_dicts", [])
        pulse_idxs = tr.get("used_pulse_indices", [])
        for pulse_idx, seg in zip(pulse_idxs, segs):
            by_pulse.setdefault(int(pulse_idx), []).append(seg)

    pooled = []
    for pulse_idx in sorted(by_pulse.keys()):
        segs = by_pulse[pulse_idx]
        if not segs:
            continue

        pre_s = float(-segs[0].get("t_rel_s", [-PRE_SEC, 0.0])[0]) if segs else float(PRE_SEC)
        post_s = float(segs[0].get("t_rel_s", [0.0, POST_SEC])[-1]) if segs else float(POST_SEC)
        t_grid = build_common_grid_from_segments(segs, pre_s, post_s)
        if t_grid is None:
            continue

        Y = interpolate_segment_dicts(segs, t_grid)
        finite_cols = np.any(np.isfinite(Y), axis=0)
        if not np.any(finite_cols):
            continue
        mean_seg = np.full(len(t_grid), np.nan, dtype=float)
        mean_seg[finite_cols] = np.nanmean(Y[:, finite_cols], axis=0)

        keep = (t_grid >= PEAK_WINDOW_S[0]) & (t_grid <= PEAK_WINDOW_S[1])
        if not np.any(keep):
            continue

        win_t = t_grid[keep]
        win_y = mean_seg[keep]
        p1_amp, p1_lat_s, p2_amp, p2_lat_s, s1, s2 = detect_two_strongest(mean_seg, t_grid, PEAK_WINDOW_S)

        pooled.append(
            {
                "pulse_number": pulse_idx + 1,
                "n_trials": int(len(segs)),
                "full_rel_s": t_grid.astype(np.float64),
                "full_values_mean": mean_seg.astype(np.float64),
                "window_rel_s": win_t.astype(np.float64),
                "window_values_mean": win_y.astype(np.float64),
                "first_peak_amp": float(p1_amp),
                "first_peak_lat_s": float(p1_lat_s),
                "second_peak_amp": float(p2_amp),
                "second_peak_lat_s": float(p2_lat_s),
                "first_peak_sample": int(s1) if np.isfinite(s1) else np.nan,
                "second_peak_sample": int(s2) if np.isfinite(s2) else np.nan,
            }
        )

    return pooled


def save_pulsogram_csv(out_path: Path, pooled: list[dict]):
    if not pooled:
        return

    max_len = max(len(p["window_values_mean"]) for p in pooled)
    sample_cols = [f"sample_{i+1}" for i in range(max_len)]
    header = [
        "pulse_number",
        "first_peak_amp",
        "first_peak_lat_ms",
        "first_peak_sample",
        "second_peak_amp",
        "second_peak_lat_ms",
        "second_peak_sample",
        "n_trials",
        *sample_cols,
    ]

    lines = [",".join(header)]
    for p in pooled:
        vals = [f"{float(v):.6f}" for v in p["window_values_mean"]]
        vals += [""] * (max_len - len(vals))
        row = [
            str(int(p["pulse_number"])),
            f"{float(p['first_peak_amp']):.6f}",
            f"{1000.0 * float(p['first_peak_lat_s']):.6f}",
            str(int(p['first_peak_sample'])) if np.isfinite(p['first_peak_sample']) else "",
            f"{float(p['second_peak_amp']):.6f}",
            f"{1000.0 * float(p['second_peak_lat_s']):.6f}",
            str(int(p['second_peak_sample'])) if np.isfinite(p['second_peak_sample']) else "",
            str(int(p["n_trials"])),
            *vals,
        ]
        lines.append(",".join(row))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_pulsogram_table(pooled: list[dict], n_rows: int):
    print("\npulse | samples (0-7 ms mean waveform) | p1_amp | p1_lat_ms | p1_samp | p2_amp | p2_lat_ms | p2_samp | n_trials")
    print("-" * 110)
    for p in pooled[:n_rows]:
        vals = ", ".join(f"{float(v):.2f}" for v in p["window_values_mean"])
        print(
            f"{int(p['pulse_number']):>5} | {vals} | "
            f"{float(p['first_peak_amp']):>6.2f} | {1000.0 * float(p['first_peak_lat_s']):>9.2f} | "
            f"{int(p['first_peak_sample']):>7} | {float(p['second_peak_amp']):>6.2f} | "
            f"{1000.0 * float(p['second_peak_lat_s']):>9.2f} | {int(p['second_peak_sample']):>7} | "
            f"{int(p['n_trials']):>8}"
        )


def build_plot_matrix(pooled: list[dict]):
    if not pooled:
        return None, None, None

    dts = []
    for p in pooled:
        t = np.asarray(p.get("full_rel_s", []), dtype=float)
        if len(t) >= 2:
            dt = float(np.median(np.diff(t)))
            if np.isfinite(dt) and dt > 0:
                dts.append(dt)
    if not dts:
        return None, None, None

    dt = float(np.median(dts))
    max_pre = max(float(-np.nanmin(np.asarray(p.get("full_rel_s", [-PRE_SEC, 0.0]), dtype=float))) for p in pooled)
    max_post = max(float(np.nanmax(np.asarray(p.get("full_rel_s", [POST_SEC]), dtype=float))) for p in pooled)
    t_plot = -max_pre + np.arange(int(np.floor((max_pre + max_post) / dt)) + 1) * dt
    pulse_numbers = np.array([int(p["pulse_number"]) for p in pooled], dtype=int)
    M = np.full((len(pooled), len(t_plot)), np.nan, dtype=float)

    for i, p in enumerate(pooled):
        tx = np.asarray(p.get("full_rel_s", []), dtype=float)
        yx = np.asarray(p.get("full_values_mean", []), dtype=float)
        if len(tx) < 2 or len(yx) != len(tx):
            continue
        m = (t_plot >= tx[0]) & (t_plot <= tx[-1])
        if np.any(m):
            M[i, m] = np.interp(t_plot[m], tx, yx)

    return pulse_numbers, t_plot, M


def plot_pulse_heatmap(block_label: str, pooled: list[dict]):
    pulse_numbers, t_plot, M = build_plot_matrix(pooled)
    if pulse_numbers is None:
        return

    stride = max(1, int(PLOT_PULSE_STRIDE))
    keep = np.arange(len(pulse_numbers))[::stride]
    pulse_numbers = pulse_numbers[keep]
    M = M[keep]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    im = ax.imshow(
        M,
        aspect='auto',
        origin='lower',
        extent=[1000.0 * t_plot[0], 1000.0 * t_plot[-1], pulse_numbers[0], pulse_numbers[-1]],
        cmap='RdBu_r',
    )
    ax.axvline(0.0, color='k', ls='--', lw=1.0)
    ax.set_title(f"Pulsogram heatmap | {block_label} | every {stride} pulse")
    ax.set_xlabel('time from pulse (ms)')
    ax.set_ylabel('pulse number')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('signal (a.u.)')
    plt.tight_layout()
    plt.show()


def plot_pulse_lines(block_label: str, pooled: list[dict]):
    pulse_numbers, t_plot, M = build_plot_matrix(pooled)
    if pulse_numbers is None:
        return

    stride = max(1, int(PLOT_PULSE_STRIDE))
    keep = np.arange(len(pulse_numbers))[::stride]
    pulse_numbers = pulse_numbers[keep]
    M = M[keep]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for pulse_num, y in zip(pulse_numbers, M):
        ax.plot(1000.0 * t_plot, y, lw=0.8, alpha=0.35, label=str(pulse_num))
    ax.axvline(0.0, color='k', ls='--', lw=1.0)
    ax.set_title(f"Pulsogram lines | {block_label} | every {stride} pulse")
    ax.set_xlabel('time from pulse (ms)')
    ax.set_ylabel('signal (a.u.)')
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
        print(f"[SKIP] no valid pulsogram trials: {imaging_path}")
        if fail_counts:
            print(f"[INFO] fail reasons: {fail_counts}")
        return

    pooled = pool_pulse_metrics(list(results.values()))
    block_label = f"{img.get('date')} {img.get('block')}"
    print(f"[RUN] {block_label} | {imaging_path.name}")
    print(f"[INFO] valid trials: {len(results)} / {len(trial_names)}")
    if fail_counts:
        print(f"[INFO] skipped: {fail_counts}")
    if pooled:
        print(f"[INFO] pulse numbers analyzed: 1 .. {int(pooled[-1]['pulse_number'])}")
        print_pulsogram_table(pooled, PRINT_FIRST_N_PULSES)
    else:
        print("[SKIP] no pooled pulse metrics")
        return

    if SAVE_CSV:
        csv_path = imaging_path.parent / f"{imaging_path.stem}_pulsogram_table.csv"
        save_pulsogram_csv(csv_path, pooled)
        print(f"[SAVED] {csv_path}")

    if SHOW_PLOTS:
        if SHOW_HEATMAP:
            plot_pulse_heatmap(block_label, pooled)
        if SHOW_OVERPLOTTED_LINES:
            plot_pulse_lines(block_label, pooled)

    if SAVE_OUTPUT:
        out_path = imaging_path.parent / f"{imaging_path.stem}_pulsogram.pkl"
        out = {
            "mouse": img.get("mouse"),
            "date": img.get("date"),
            "block": img.get("block"),
            "analysis": "pulsogram",
            "settings": {
                "signal_mode": SIGNAL_MODE,
                "pre_sec": float(PRE_SEC),
                "post_sec": float(POST_SEC),
                "post_window_mode": POST_WINDOW_MODE,
                "post_window_scale": float(POST_WINDOW_SCALE),
                "period_fraction": float(PERIOD_FRACTION),
                "baseline_mode": BASELINE_MODE,
                "min_pulses": int(MIN_PULSES),
                "spread_mode": SPREAD_MODE,
                "peak_window_s": list(PEAK_WINDOW_S),
                "only_trial": ONLY_TRIAL,
                "save_csv": bool(SAVE_CSV),
                "show_heatmap": bool(SHOW_HEATMAP),
                "show_overplotted_lines": bool(SHOW_OVERPLOTTED_LINES),
                "plot_pulse_stride": int(PLOT_PULSE_STRIDE),
            },
            "trial_results": results,
            "pooled_metrics": pooled,
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
