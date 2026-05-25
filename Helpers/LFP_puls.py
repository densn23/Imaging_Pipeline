from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = "Jamie10"
SINGLE_DATE = "09-12-25"
SINGLE_BLOCK = "R2"

# -------------------------
# ROOTS / INPUT
# -------------------------
EPHYS_ROOT = DATA_ANALYSIS_ROOT / MOUSE_NAME / "Open_Ephys"
BLOCK_RE = re.compile(r"^R\d+$")
ephys_pkl = None if SINGLE_BLOCK is None else (
    EPHYS_ROOT / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_epoched_ephys.pkl"
)

# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = False
SAVE_OUTPUT = False
SHOW_PLOTS = True
SAVE_FIGURE = False

# -------------------------
# PLOT TOGGLES
# -------------------------
SHOW_RAW_PULSOGRAM = True
SHOW_ARTIFACT_REDUCED_PULSOGRAM = False
PLOT_PULSE_STRIDE = 1

# -------------------------
# SETTINGS
# -------------------------
SIGNAL_CHANNEL = "LFP"
PRE_SEC = 0.010
POST_SEC = 0.010
POST_WINDOW_MODE = "freq_aware"   # "fixed" or "freq_aware"
POST_WINDOW_SCALE = 1.0
PERIOD_FRACTION = 0.95
BASELINE_MODE = "median_pre"      # "none", "median_pre", "mean_pre"
ARTIFACT_REDUCTION_MODE = "template_subtract"   # "none" or "template_subtract"
ARTIFACT_TEMPLATE_STAT = "median"               # "median" or "mean"
MIN_PULSES = 5
ONLY_TRIAL = None


def trial_sort_key(name: str):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return np.nan
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return 1.0 / dt


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


def baseline_correct_segment(seg: np.ndarray, rel_grid: np.ndarray) -> np.ndarray:
    if BASELINE_MODE == "none":
        return np.asarray(seg, dtype=float)

    seg = np.asarray(seg, dtype=float)
    pre = seg[rel_grid < 0]
    if len(pre) == 0:
        return seg

    if BASELINE_MODE == "mean_pre":
        baseline = float(np.mean(pre))
    else:
        baseline = float(np.median(pre))
    return seg - baseline


def build_rel_grid(t: np.ndarray, pre_s: float, post_s: float) -> np.ndarray | None:
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0:
        return None
    dt = 1.0 / fs
    n = int(np.floor((pre_s + post_s) / dt)) + 1
    if n < 4:
        return None
    return -pre_s + np.arange(n) * dt


def extract_pulse_segment_dict(t: np.ndarray, x: np.ndarray, tp: float, pre_s: float, post_s: float):
    t_rel = t - float(tp)
    keep = (t_rel >= -pre_s) & (t_rel <= post_s)
    if np.sum(keep) < 4:
        return None
    t_seg = np.asarray(t_rel[keep], dtype=np.float64)
    x_seg = np.asarray(x[keep], dtype=np.float64)
    x_seg = baseline_correct_segment(x_seg, t_seg)
    return {"t_rel_s": t_seg, "signal": x_seg}


def build_freq_aware_folded_segment_dict(
    t: np.ndarray,
    x: np.ndarray,
    pulse_times: np.ndarray,
    pulse_idx: int,
    pre_s: float,
    post_s: float,
):
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0:
        return None

    rel_grid = build_rel_grid(t, pre_s, post_s)
    if rel_grid is None or len(rel_grid) < 4:
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
        rows = []
        for j in range(n_periods):
            post_start = float(pulse_times[pulse_idx + j])
            post_end = float(pulse_times[pulse_idx + j + 1])
            if not (post_end > post_start):
                return None
            if post_start < t[0] or post_end > t[-1]:
                return None
            post_dur = post_end - post_start
            post_sample_end = post_start + frac * post_dur
            rows.append(np.interp(post_start + post_alpha * (post_sample_end - post_start), t, x))
        y[post_mask] = np.mean(np.vstack(rows), axis=0)

    y = baseline_correct_segment(y, rel_grid)
    return {"t_rel_s": np.asarray(rel_grid, dtype=np.float64), "signal": np.asarray(y, dtype=np.float64)}


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


def reduce_artifact(segments: list[dict]) -> tuple[list[dict], dict | None]:
    if ARTIFACT_REDUCTION_MODE == "none" or not segments:
        return segments, None

    t_ref = np.asarray(segments[0].get("t_rel_s", []), dtype=float)
    if len(t_ref) < 4:
        return segments, None

    rows = []
    for seg in segments:
        tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
        yx = np.asarray(seg.get("signal", []), dtype=float)
        if len(tx) == len(t_ref) and np.allclose(tx, t_ref) and yx.shape == tx.shape:
            rows.append(yx)
    if not rows:
        return segments, None

    Y = np.vstack(rows)
    if ARTIFACT_TEMPLATE_STAT == "mean":
        template = np.nanmean(Y, axis=0)
    else:
        template = np.nanmedian(Y, axis=0)

    reduced = []
    for seg in segments:
        tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
        yx = np.asarray(seg.get("signal", []), dtype=float)
        if len(tx) == len(t_ref) and np.allclose(tx, t_ref) and yx.shape == tx.shape:
            yr = baseline_correct_segment(yx - template, tx)
            reduced.append({"t_rel_s": tx.astype(np.float64), "signal": yr.astype(np.float64)})
        else:
            reduced.append(seg)

    template_dict = {"t_rel_s": t_ref.astype(np.float64), "signal": np.asarray(template, dtype=np.float64)}
    return reduced, template_dict


def build_common_grid_from_segments(segments: list[dict]) -> np.ndarray | None:
    dts = []
    lo = []
    hi = []
    for s in segments:
        tr = np.asarray(s.get("t_rel_s", []), dtype=float)
        if len(tr) >= 2:
            dt = float(np.median(np.diff(tr)))
            if np.isfinite(dt) and dt > 0:
                dts.append(dt)
                lo.append(float(tr[0]))
                hi.append(float(tr[-1]))
    if not dts:
        return None
    dt_med = float(np.median(dts))
    t0 = float(max(lo))
    t1 = float(min(hi))
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        return None
    n = int(np.floor((t1 - t0) / dt_med)) + 1
    if n < 4:
        return None
    return t0 + np.arange(n) * dt_med


def interpolate_segment_dicts(segments: list[dict], t_grid: np.ndarray) -> np.ndarray:
    Y = np.full((len(segments), len(t_grid)), np.nan, dtype=float)
    for i, seg in enumerate(segments):
        tx = np.asarray(seg.get("t_rel_s", []), dtype=float)
        yx = np.asarray(seg.get("signal", []), dtype=float)
        if len(tx) < 2 or yx.shape != tx.shape:
            continue
        m = (t_grid >= tx[0]) & (t_grid <= tx[-1])
        if np.any(m):
            Y[i, m] = np.interp(t_grid[m], tx, yx)
    return Y


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
        if len(tx) < 2 or yx.shape != tx.shape:
            continue
        m = (t_plot >= tx[0]) & (t_plot <= tx[-1])
        if np.any(m):
            M[i, m] = np.interp(t_plot[m], tx, yx)

    return pulse_numbers, t_plot, M


def pool_pulse_metrics(trial_results: list[dict], field_name: str):
    by_pulse: dict[int, list[dict]] = {}
    for tr in trial_results:
        segs = tr.get(field_name, [])
        pulse_idxs = tr.get("used_pulse_indices", [])
        for pulse_idx, seg in zip(pulse_idxs, segs):
            by_pulse.setdefault(int(pulse_idx), []).append(seg)

    pooled = []
    for pulse_idx in sorted(by_pulse.keys()):
        segs = by_pulse[pulse_idx]
        if not segs:
            continue
        t_grid = build_common_grid_from_segments(segs)
        if t_grid is None:
            continue
        Y = interpolate_segment_dicts(segs, t_grid)
        finite_cols = np.any(np.isfinite(Y), axis=0)
        if not np.any(finite_cols):
            continue
        mean_seg = np.full(len(t_grid), np.nan, dtype=float)
        mean_seg[finite_cols] = np.nanmean(Y[:, finite_cols], axis=0)
        pooled.append(
            {
                "pulse_number": pulse_idx + 1,
                "n_trials": int(len(segs)),
                "full_rel_s": t_grid.astype(np.float64),
                "full_values_mean": mean_seg.astype(np.float64),
            }
        )
    return pooled


def plot_pulse_heatmap(ax, pooled: list[dict], title: str):
    pulse_numbers, t_plot, M = build_plot_matrix(pooled)
    if pulse_numbers is None:
        ax.axis("off")
        return

    stride = max(1, int(PLOT_PULSE_STRIDE))
    keep = np.arange(len(pulse_numbers))[::stride]
    pulse_numbers = pulse_numbers[keep]
    M = M[keep]

    finite = M[np.isfinite(M)]
    vmax = np.percentile(np.abs(finite), 99.0) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[1000.0 * t_plot[0], 1000.0 * t_plot[-1], pulse_numbers[0], pulse_numbers[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("time from pulse (ms)")
    ax.set_ylabel("pulse number")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("LFP (a.u.)")


def extract_lfp_trial(trial_name: str, td_e: dict):
    channels = td_e.get("channels", {})
    x = np.asarray(channels.get(SIGNAL_CHANNEL, []), dtype=float)
    t = np.asarray(td_e.get("t_stim_s", []), dtype=float)
    if len(x) == 0 or len(t) == 0:
        return None, "missing_lfp_or_time"

    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]
    if len(t) < 8:
        return None, "too_short_trace"

    pulse_times = np.asarray(td_e.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) < MIN_PULSES:
        return None, "too_few_pulses"

    pre_s, post_s = resolve_pulse_window_s(pulse_times)
    raw_segments, used_idx = extract_all_pulse_segment_dicts(t, x, pulse_times, pre_s, post_s)
    if len(raw_segments) < MIN_PULSES:
        return None, "too_few_valid_pulses"

    reduced_segments, template = reduce_artifact(raw_segments)
    return {
        "trial": trial_name,
        "pulse_times_s": pulse_times.astype(np.float64),
        "used_pulse_indices": np.asarray(used_idx, dtype=int),
        "n_pulses_used": int(len(used_idx)),
        "window_pre_s": float(pre_s),
        "window_post_s": float(post_s),
        "raw_segment_dicts": raw_segments,
        "artifact_reduced_segment_dicts": reduced_segments,
        "artifact_template": template,
    }, None


def run_single(ephys_path: Path):
    with open(ephys_path, "rb") as f:
        eph = pickle.load(f)

    e_trials = eph.get("trials", {})
    if not e_trials:
        print(f"[SKIP] missing trials in {ephys_path.name}")
        return

    trial_names = sorted(e_trials.keys(), key=trial_sort_key)
    results = {}
    fail_counts = {}
    for name in trial_names:
        if ONLY_TRIAL is not None and name != ONLY_TRIAL:
            continue
        tr, err = extract_lfp_trial(name, e_trials[name])
        if tr is None:
            fail_counts[err] = fail_counts.get(err, 0) + 1
            continue
        results[name] = tr

    if not results:
        print(f"[SKIP] no valid LFP pulsogram trials: {ephys_path}")
        if fail_counts:
            print(f"[INFO] fail reasons: {fail_counts}")
        return

    trial_list = list(results.values())
    raw_pooled = pool_pulse_metrics(trial_list, "raw_segment_dicts")
    reduced_pooled = pool_pulse_metrics(trial_list, "artifact_reduced_segment_dicts")

    block_label = f"{eph.get('block', ephys_path.parent.name)}"
    print(f"[RUN] {ephys_path.parent.parent.name} {block_label} | {ephys_path.name}")
    print(f"[INFO] valid trials: {len(results)} / {len(trial_names)}")
    if fail_counts:
        print(f"[INFO] skipped: {fail_counts}")
    if raw_pooled:
        print(f"[INFO] raw pulses analyzed: 1 .. {int(raw_pooled[-1]['pulse_number'])}")
    if reduced_pooled:
        print(f"[INFO] artifact-reduced pulses analyzed: 1 .. {int(reduced_pooled[-1]['pulse_number'])}")

    if SHOW_PLOTS and (SHOW_RAW_PULSOGRAM or SHOW_ARTIFACT_REDUCED_PULSOGRAM):
        n_cols = int(SHOW_RAW_PULSOGRAM) + int(SHOW_ARTIFACT_REDUCED_PULSOGRAM)
        fig, axes = plt.subplots(1, n_cols, figsize=(8.5 * n_cols, 6.0))
        axes = np.atleast_1d(axes)
        j = 0
        if SHOW_RAW_PULSOGRAM:
            plot_pulse_heatmap(axes[j], raw_pooled, f"{ephys_path.parent.parent.name} {block_label} | LFP pulsogram | raw")
            j += 1
        if SHOW_ARTIFACT_REDUCED_PULSOGRAM:
            plot_pulse_heatmap(axes[j], reduced_pooled, f"{ephys_path.parent.parent.name} {block_label} | LFP pulsogram | artifact reduced")
        plt.tight_layout()
        if SAVE_FIGURE:
            out_fig = ephys_path.parent / f"{block_label}_lfp_pulsogram.png"
            fig.savefig(out_fig, dpi=150, bbox_inches="tight")
            print(f"[SAVED] {out_fig}")
        plt.show()

    if SAVE_OUTPUT:
        out_path = ephys_path.parent / f"{ephys_path.stem.replace('_epoched_ephys', '')}_lfp_pulsogram.pkl"
        out = {
            "block": eph.get("block"),
            "sample_rate": eph.get("sample_rate"),
            "analysis": "lfp_pulsogram",
            "settings": {
                "signal_channel": SIGNAL_CHANNEL,
                "pre_sec": float(PRE_SEC),
                "post_sec": float(POST_SEC),
                "post_window_mode": POST_WINDOW_MODE,
                "post_window_scale": float(POST_WINDOW_SCALE),
                "period_fraction": float(PERIOD_FRACTION),
                "baseline_mode": BASELINE_MODE,
                "artifact_reduction_mode": ARTIFACT_REDUCTION_MODE,
                "artifact_template_stat": ARTIFACT_TEMPLATE_STAT,
                "min_pulses": int(MIN_PULSES),
                "only_trial": ONLY_TRIAL,
            },
            "trial_results": results,
            "raw_pooled_metrics": raw_pooled,
            "artifact_reduced_pooled_metrics": reduced_pooled,
        }
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] {out_path}")


def run_batch(ephys_root: Path):
    for date_dir in sorted(ephys_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
                continue
            eph_p = block_dir / f"{block_dir.name}_epoched_ephys.pkl"
            if not eph_p.exists():
                continue
            run_single(eph_p)


def run_single_date(ephys_root: Path, date_name: str):
    date_dir = ephys_root / date_name
    if not date_dir.exists():
        print(f"[WARN] date folder not found: {date_dir}")
        return

    for block_dir in sorted(date_dir.iterdir()):
        if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
            continue
        eph_p = block_dir / f"{block_dir.name}_epoched_ephys.pkl"
        if not eph_p.exists():
            continue
        run_single(eph_p)


def main() -> None:
    if RUN_BATCH:
        run_batch(EPHYS_ROOT)
    elif SINGLE_DATE is not None and SINGLE_BLOCK is None:
        run_single_date(EPHYS_ROOT, SINGLE_DATE)
    else:
        if ephys_pkl is None:
            print("Set SINGLE_DATE to run one date, or set both SINGLE_DATE and SINGLE_BLOCK to run one block.")
            return
        run_single(ephys_pkl)


if __name__ == "__main__":
    main()
