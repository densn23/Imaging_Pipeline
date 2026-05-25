from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
from config import DATA_ANALYSIS_ROOT

MOUSE_NAME = None
SINGLE_DATE = None
SINGLE_BLOCK = None


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
# PATHS (single-block mode)
# -------------------------
pkl_path = None if SINGLE_BLOCK is None else (
    None if _SINGLE_MOUSE_NAME is None else (
        DATA_ANALYSIS_ROOT / _SINGLE_MOUSE_NAME / "Imaging_Data" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_traces.pkl"
    )
)
ephys_pkl_path = None if SINGLE_BLOCK is None else (
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

# -------------------------
# SETTINGS
# -------------------------
K = 1
N_TO_PLOT = 2
PLOT_ALL_TRIALS = False  # True = plot all trials, False = plot N_TO_PLOT evenly spaced trials
ONLY_TRIAL = None        # e.g. "R5_1"; None = use normal plot selection
PLOT_MODE = "processed"  # "processed" or "trim_preview"

BLEACH_MODE = "double_exp"  # "none", "linear", "single_exp", "double_exp"
DFF_MODE = "post"           # "none", "pre", "post"
BLEACH_RUNAWAY_Z = 3.14     # reject positive corrected dF/F runaway after last pulse

# -------------------------
# EXECUTION TOGGLES
# -------------------------
RUN_BATCH = False
SAVE_OUTPUT = False
SHOW_PLOTS = True


def trial_key(name: str) -> int:
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def choose_trials_to_plot(names: list[str]) -> list[str]:
    if ONLY_TRIAL is not None:
        if ONLY_TRIAL in names:
            return [ONLY_TRIAL]
        print(f"[WARN] ONLY_TRIAL not found in this block: {ONLY_TRIAL}")
        return []

    if PLOT_ALL_TRIALS:
        return list(names)

    idxs = np.linspace(0, len(names) - 1, min(N_TO_PLOT, len(names))).astype(int)
    return [names[i] for i in idxs]


def trim_by_level(F, k=5):
    F = np.asarray(F, float)
    n = len(F)

    a = int(0.05 * n)
    b = int(0.95 * n)
    mid = F[a:b]

    mu = np.mean(mid)
    sigma = np.std(mid)
    lower = mu - 5 * sigma
    upper = mu + 5 * sigma
    valid = (F >= lower) & (F <= upper)

    start = 0
    run = 0
    for i in range(n):
        run = run + 1 if valid[i] else 0
        if run >= k:
            start = i - k + 1
            break

    end = n
    run = 0
    for i in range(n - 1, -1, -1):
        run = run + 1 if valid[i] else 0
        if run >= k:
            end = i + k
            break

    start = max(0, start)
    end = min(n, end)
    return F[start:end], start, end


def get_aligned_trace(trial_name, trials, ephys_trials):
    F = np.asarray(trials[trial_name]["trace_raw"], float)
    e_trial = ephys_trials[trial_name]
    t = np.asarray(e_trial["cam_frame_times_stim_s"], float)
    stim_on = int(e_trial.get("stim_on_frame_idx", -1))

    n = min(len(F), len(t))
    F = F[:n]
    t = t[:n]
    stim_on = max(0, min(stim_on, n))
    return F, t, stim_on


def compute_dff(F, stim_on_idx):
    if stim_on_idx <= 0:
        return None
    F0 = np.mean(F[:stim_on_idx])
    if F0 == 0:
        return None
    return (F - F0) / F0


def pre_stim_slope(t, F, stim_on_idx) -> float:
    if stim_on_idx < 5:
        return np.nan
    try:
        return float(np.polyfit(np.asarray(t[:stim_on_idx], float), np.asarray(F[:stim_on_idx], float), 1)[0])
    except Exception:
        return np.nan


def last_pulse_time_s(ephys_trial: dict) -> float:
    pulse_times = np.asarray(ephys_trial.get("stim_pulse_times_s", []), dtype=float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if len(pulse_times) == 0:
        return np.nan
    return float(pulse_times[-1])


def bleach_runaway_qc(
    t: np.ndarray,
    F_corr: np.ndarray,
    stim_on_idx: int,
    last_pulse_s: float,
) -> dict:
    qc = {
        "checked": False,
        "reject": False,
        "reject_reason": None,
        "runaway_z_threshold": float(BLEACH_RUNAWAY_Z),
        "runaway_z": np.nan,
        "pre_median": np.nan,
        "pre_sd": np.nan,
        "post_median": np.nan,
        "last_pulse_time_s": float(last_pulse_s) if np.isfinite(last_pulse_s) else np.nan,
    }

    dff_corr = compute_dff(F_corr, stim_on_idx)
    if dff_corr is None or not np.isfinite(last_pulse_s):
        return qc

    t = np.asarray(t, dtype=float)
    dff_corr = np.asarray(dff_corr, dtype=float)
    if len(t) == 0 or dff_corr.shape != t.shape:
        return qc

    pre = dff_corr[(t < 0.0) & np.isfinite(dff_corr)]
    post = dff_corr[(t >= float(last_pulse_s)) & np.isfinite(dff_corr)]
    if len(pre) < 3 or len(post) < 3:
        return qc

    pre_median = float(np.nanmedian(pre))
    pre_sd = float(np.nanstd(pre, ddof=1))
    post_median = float(np.nanmedian(post))
    qc.update(
        {
            "checked": True,
            "pre_median": pre_median,
            "pre_sd": pre_sd,
            "post_median": post_median,
        }
    )

    if not np.isfinite(pre_sd) or pre_sd <= 0:
        return qc

    runaway_z = float((post_median - pre_median) / pre_sd)
    qc["runaway_z"] = runaway_z
    if runaway_z > float(BLEACH_RUNAWAY_Z):
        qc["reject"] = True
        qc["reject_reason"] = "positive_runaway"
    return qc


def fit_bleach_model(t, F, stim_on_idx, mode="none"):
    if mode == "none":
        return None, None, None, False
    if stim_on_idx < 5:
        return None, None, None, False

    t = np.asarray(t, float)
    F = np.asarray(F, float)
    t_pre = t[:stim_on_idx]
    F_pre = F[:stim_on_idx]

    t0 = t_pre[0]
    x_pre = t_pre - t0
    x_full = t - t0

    def fit_linear():
        m, b = np.polyfit(t_pre, F_pre, 1)
        return m * t_pre + b, m * t + b

    def fit_single_exp():
        from scipy.optimize import curve_fit

        def sexp(x, a, tau, c):
            return a * np.exp(-x / tau) + c

        c0 = float(np.median(F_pre))
        a0 = max(0.0, float(F_pre[0] - c0))
        p0 = (a0, 0.5 * max(1e-3, float(x_pre[-1])), c0)

        popt, _ = curve_fit(
            sexp,
            x_pre,
            F_pre,
            p0=p0,
            bounds=((0.0, 1e-6, -np.inf), (np.inf, np.inf, np.inf)),
            maxfev=20000,
        )
        return sexp(x_pre, *popt), sexp(x_full, *popt)

    if mode == "linear":
        B_pre, B_full = fit_linear()

    elif mode == "single_exp":
        try:
            B_pre, B_full = fit_single_exp()
        except Exception:
            return None, None, None, False

    elif mode == "double_exp":
        from scipy.optimize import curve_fit

        def dexp(x, a, tau1, b, tau2, c):
            return a * np.exp(-x / tau1) + b * np.exp(-x / tau2) + c

        c0 = float(np.median(F_pre))
        a0 = max(0.0, float(F_pre[0] - c0))
        b0 = 0.5 * a0
        p0 = (
            a0,
            0.2 * max(1e-3, float(x_pre[-1])),
            b0,
            0.8 * max(1e-3, float(x_pre[-1])),
            c0,
        )

        try:
            popt, _ = curve_fit(
                dexp,
                x_pre,
                F_pre,
                p0=p0,
                bounds=((0.0, 1e-6, 0.0, 1e-6, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf)),
                maxfev=20000,
            )
            B_pre = dexp(x_pre, *popt)
            B_full = dexp(x_full, *popt)
        except Exception:
            try:
                B_pre, B_full = fit_single_exp()
            except Exception:
                try:
                    B_pre, B_full = fit_linear()
                except Exception:
                    return None, None, None, False

    else:
        return None, None, None, False

    # Enforce strict monotonic decay over full interval.
    if np.any(np.diff(B_full) > 0):
        return None, None, None, False

    drift = B_full - B_full[0]
    F_corr = F - drift
    return B_pre, B_full, F_corr, True


def plot_trim_preview(names_to_plot: list[str], trials: dict, ephys_trials: dict) -> None:
    if not names_to_plot:
        print("[WARN] no trials selected for trim preview")
        return

    fig, axes = plt.subplots(len(names_to_plot), 1, figsize=(12, 2.8 * len(names_to_plot)))
    if len(names_to_plot) == 1:
        axes = np.array([axes])

    for row, name in enumerate(names_to_plot):
        if name not in ephys_trials:
            continue

        ax = axes[row]
        F_full = np.asarray(trials[name]["trace_raw"], float)
        F_keep, s, e = trim_by_level(F_full, K)

        x_full = np.arange(len(F_full))
        x_keep = np.arange(s, e)
        n_left = int(s)
        n_right = int(len(F_full) - e)

        ax.plot(x_full, F_full, color="0.75", lw=1.0, label="raw full trace")
        ax.plot(x_keep, F_keep, color="tab:blue", lw=1.2, label="would keep")

        if s > 0:
            ax.axvspan(0, s - 1, color="tab:red", alpha=0.12, label="would trim start")
        if e < len(F_full):
            ax.axvspan(e, len(F_full) - 1, color="tab:orange", alpha=0.12, label="would trim end")
        ax.axvline(s, color="tab:red", ls="--", lw=1.0)
        ax.axvline(e, color="tab:orange", ls="--", lw=1.0)

        ax.set_title(
            f"{name} | raw trace with suggested trim | kept [{s}:{e}] | "
            f"trimmed_left={n_left} trimmed_right={n_right}"
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("F (a.u.)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    plt.show()


def run_single_block(imaging_pkl: Path, ephys_pkl: Path, save_output=False, show_plots=True):
    with open(imaging_pkl, "rb") as f:
        data = pickle.load(f)
    with open(ephys_pkl, "rb") as f:
        ephys = pickle.load(f)

    trials = data["trials"]
    ephys_trials = ephys["trials"]
    names = sorted(trials.keys(), key=trial_key)
    names_to_plot = choose_trials_to_plot(names)

    if PLOT_MODE == "trim_preview":
        if show_plots:
            plot_trim_preview(names_to_plot, trials, ephys_trials)
        else:
            print("[INFO] PLOT_MODE='trim_preview' but SHOW_PLOTS=False")
        return

    processed = {}
    fit_not_applied = []
    fit_not_applied_reasons = {}

    if show_plots:
        fig, axes = plt.subplots(len(names_to_plot), 2, figsize=(14, 3 * len(names_to_plot)))
        if len(names_to_plot) == 1:
            axes = np.array([axes])
    else:
        axes = None

    for name in names:
        if name not in ephys_trials:
            continue

        F, t, stim_on = get_aligned_trace(name, trials, ephys_trials)
        e_trial = ephys_trials[name]
        n_frames = int(len(F))
        if len(t) >= 2:
            frame_dt_s = float(np.median(np.diff(t)))
            fps_hz = float(1.0 / frame_dt_s) if frame_dt_s > 0 else np.nan
        else:
            frame_dt_s = np.nan
            fps_hz = np.nan
        x = np.arange(len(F))
        bleach_reject_reason = None
        bleach_qc = {
            "mode": BLEACH_MODE,
            "pre_slope": pre_stim_slope(t, F, stim_on),
            "last_pulse_time_s": last_pulse_time_s(e_trial),
            "checked": False,
            "reject": False,
            "reject_reason": None,
            "runaway_z_threshold": float(BLEACH_RUNAWAY_Z),
            "runaway_z": np.nan,
            "pre_median": np.nan,
            "pre_sd": np.nan,
            "post_median": np.nan,
        }
        try:
            B_pre, B_full, F_corr, ok = fit_bleach_model(t, F, stim_on, mode=BLEACH_MODE)
        except Exception:
            B_pre, B_full, F_corr, ok = None, None, None, False
            bleach_reject_reason = "fit_exception"

        if ok:
            runaway_qc = bleach_runaway_qc(t, F_corr, stim_on, bleach_qc["last_pulse_time_s"])
            bleach_qc.update(runaway_qc)
            if runaway_qc.get("reject", False):
                ok = False
                bleach_reject_reason = str(runaway_qc.get("reject_reason") or "positive_runaway")
                B_pre, B_full, F_corr = None, None, None
        elif BLEACH_MODE != "none" and bleach_reject_reason is None:
            bleach_reject_reason = "fit_failed_or_invalid"

        if BLEACH_MODE != "none" and not ok:
            fit_not_applied.append(name)
            fit_not_applied_reasons[name] = bleach_reject_reason or "not_applied"

        if DFF_MODE == "none":
            F_dff = None
        elif DFF_MODE == "pre":
            F_dff = compute_dff(F, stim_on)
        elif DFF_MODE == "post":
            F_dff = compute_dff(F_corr if ok else F, stim_on)
        else:
            F_dff = compute_dff(F, stim_on)

        processed[name] = {
            "t": t,
            "time_units": "s",
            "n_frames": n_frames,
            "frame_dt_s": frame_dt_s,
            "fps_hz": fps_hz,
            "stim_on_idx": stim_on,
            "trim_bounds": (0, int(len(F))),
            "F_raw": F,
            "F_bleach_corr": F_corr if ok else None,
            "B_pre": B_pre if ok else None,
            "B_full": B_full if ok else None,
            "dff": F_dff,
            "bleach_applied": bool(ok),
            "bleach_reject_reason": bleach_reject_reason,
            "bleach_qc": bleach_qc,
        }

        if not show_plots or name not in names_to_plot:
            continue

        row = names_to_plot.index(name)
        axF = axes[row, 0]
        axD = axes[row, 1]

        if stim_on > 1:
            axF.axvspan(x[0], x[stim_on - 1], alpha=0.15, color="tab:blue", label="pre-stim")
            axD.axvspan(x[0], x[stim_on - 1], alpha=0.15, color="tab:blue", label="pre-stim")

        axF.plot(x, F, color="tab:blue", lw=1.0, alpha=0.70, zorder=1, label="F raw")
        if ok:
            axF.plot(x[:stim_on], B_pre, color="tab:orange", lw=2.0, alpha=0.95, zorder=3, label=f"bleach fit ({BLEACH_MODE}, pre)")
            axF.plot(x, F_corr, color="tab:green", lw=1.2, alpha=0.75, zorder=4, label="F bleach-corrected")

        axF.set_title(f"{name} | F | stim_on={stim_on}")
        axF.set_xlabel("frame")
        axF.set_ylabel("F (a.u.)")
        axF.legend(loc="lower left", fontsize=8, framealpha=0.85)

        if DFF_MODE == "none":
            axD.set_title(f"{name} | dF/F (OFF)")
            axD.axis("off")
        elif F_dff is None:
            axD.set_title(f"{name} | dF/F unavailable")
            axD.axis("off")
        else:
            if DFF_MODE == "pre":
                dff_label = "dF/F (raw)"
            elif DFF_MODE == "post":
                dff_label = "dF/F (bleach-corr)" if ok else "dF/F (raw fallback)"
            else:
                dff_label = f"dF/F ({DFF_MODE})"
            axD.plot(x, F_dff, lw=1.2, color="tab:blue", label=dff_label)
            axD.set_title(f"{name} | dF/F")
            axD.set_xlabel("frame")
            axD.set_ylabel("dF/F")
            axD.legend(loc="lower left", fontsize=8, framealpha=0.85)

    if show_plots:
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    if BLEACH_MODE != "none" and fit_not_applied:
        print(f"[FIT_NOT_APPLIED] {imaging_pkl.parent.name} {imaging_pkl.stem}: {len(fit_not_applied)} trial(s)")
        print("  " + ", ".join(f"{name} ({fit_not_applied_reasons.get(name, 'unknown')})" for name in fit_not_applied))

    if save_output:
        out_path = imaging_pkl.parent / f"{imaging_pkl.stem}_processed.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "mouse": data.get("mouse"),
                    "date": data.get("date"),
                    "block": data.get("block"),
                    "time_units": "s",
                    "trials": processed,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[SAVED] {out_path}")


def run_batch(imaging_root: Path, ephys_root: Path):
    block_re = re.compile(r"^R\d+$")
    for date_dir in sorted(imaging_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not block_re.match(block_dir.name):
                continue
            block = block_dir.name
            imaging_pkl = block_dir / f"{block}_traces.pkl"
            ephys_pkl = ephys_root / date_dir.name / block / f"{block}_epoched_ephys.pkl"
            if not imaging_pkl.exists() or not ephys_pkl.exists():
                continue
            print(f"[RUN] {date_dir.name} {block}")
            run_single_block(imaging_pkl, ephys_pkl, save_output=SAVE_OUTPUT, show_plots=SHOW_PLOTS)


def run_single_date(imaging_root: Path, ephys_root: Path, date_name: str):
    date_dir = imaging_root / date_name
    if not date_dir.exists():
        print(f"[WARN] date folder not found: {date_dir}")
        return

    block_re = re.compile(r"^R\d+$")
    for block_dir in sorted(date_dir.iterdir()):
        if not block_dir.is_dir() or not block_re.match(block_dir.name):
            continue
        block = block_dir.name
        imaging_pkl = block_dir / f"{block}_traces.pkl"
        ephys_pkl = ephys_root / date_name / block / f"{block}_epoched_ephys.pkl"
        if not imaging_pkl.exists() or not ephys_pkl.exists():
            continue
        print(f"[RUN] {date_name} {block}")
        run_single_block(imaging_pkl, ephys_pkl, save_output=SAVE_OUTPUT, show_plots=SHOW_PLOTS)


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
            single_imaging = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_traces.pkl"
            single_ephys = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_epoched_ephys.pkl"
            if not single_imaging.exists() or not single_ephys.exists():
                print(f"[SKIP] missing single-block inputs for {mouse_name} | {SINGLE_DATE} | {SINGLE_BLOCK}")
                continue
            run_single_block(single_imaging, single_ephys, save_output=SAVE_OUTPUT, show_plots=SHOW_PLOTS)


if __name__ == "__main__":
    main()


