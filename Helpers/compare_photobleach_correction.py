from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_ANALYSIS_ROOT
from preprocess_data import compute_dff, fit_bleach_model


MICE = ["Jamie5", "Jamie6", "Jamie8", "Jamie10", "Jamie11", "Jamie12"]
OUTPUT_DIR = DATA_ANALYSIS_ROOT / "tables" / "photobleach_comparison"
MAX_EXAMPLES = 6

# Best diagnostic fit from Jamie11 | 01-05-26 | R13_1:
# double_skip10s_lower_envelope.
LONG_A_FAST = 16.45131175080517
LONG_TAU_FAST_S = 24.58504471770192
LONG_A_SLOW = 602.8506464600371
LONG_TAU_SLOW_S = 4699.474843817622


def trial_key(name: str) -> int:
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def block_info_from_trace_path(path: Path) -> tuple[str, str, str]:
    block = path.parent.name
    date = path.parent.parent.name
    mouse = path.parent.parent.parent.parent.name
    return mouse, date, block


def matching_ephys_path(mouse: str, date: str, block: str) -> Path:
    return DATA_ANALYSIS_ROOT / mouse / "Open_Ephys" / date / block / f"{block}_epoched_ephys.pkl"


def aligned_trace(trials: dict, ephys_trials: dict, trial_name: str):
    F = np.asarray(trials[trial_name]["trace_raw"], dtype=float)
    e_trial = ephys_trials[trial_name]
    t = np.asarray(e_trial["cam_frame_times_stim_s"], dtype=float)
    stim_on = int(e_trial.get("stim_on_frame_idx", -1))
    n = min(len(F), len(t))
    return F[:n], t[:n], max(0, min(stim_on, n))


def trace_is_usable(F: np.ndarray, stim_on_idx: int) -> bool:
    if len(F) < 20 or stim_on_idx < 5:
        return False
    pre_level = float(np.nanmedian(F[:stim_on_idx]))
    tail_level = float(np.nanmedian(F[int(0.9 * len(F)):]))
    if not np.isfinite(pre_level) or not np.isfinite(tail_level) or pre_level <= 0:
        return False
    # Reject traces where the recording clearly falls to an off/camera-baseline level.
    if tail_level < 0.5 * pre_level:
        return False
    return True


def fixed_long_decay_correction(t: np.ndarray, F: np.ndarray, stim_on_idx: int):
    if len(t) < 5 or len(F) != len(t) or stim_on_idx < 2:
        return None, None, False

    t = np.asarray(t, dtype=float)
    F = np.asarray(F, dtype=float)
    x = t - t[0]
    shape = (
        LONG_A_FAST * np.exp(-x / LONG_TAU_FAST_S)
        + LONG_A_SLOW * np.exp(-x / LONG_TAU_SLOW_S)
    )

    if not np.all(np.isfinite(shape)) or shape[0] == 0:
        return None, None, False

    pre = F[:stim_on_idx]
    F_ref = float(np.median(pre))
    B_full = F_ref * (shape / shape[0])
    drift = B_full - B_full[0]
    F_corr = F - drift
    return B_full, F_corr, True


def summarize_trace(F: np.ndarray, F_corr: np.ndarray | None, stim_on_idx: int):
    if F_corr is None:
        return np.nan, np.nan, np.nan
    dff = compute_dff(F_corr, stim_on_idx)
    lift_end = float(F_corr[-1] - F[-1])
    if dff is None:
        return lift_end, np.nan, np.nan
    return lift_end, float(np.nanmin(dff)), float(np.nanmax(dff))


def discover_examples() -> list[dict]:
    selected = []

    for mouse in MICE:
        imaging_root = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data"
        if not imaging_root.exists():
            continue

        fallback = None
        for trace_path in sorted(imaging_root.rglob("*_traces.pkl")):
            _, date, block = block_info_from_trace_path(trace_path)
            ephys_path = matching_ephys_path(mouse, date, block)
            if not ephys_path.exists():
                continue

            try:
                traces = load_pickle(trace_path)
                ephys = load_pickle(ephys_path)
            except Exception:
                continue

            trials = traces.get("trials", {})
            ephys_trials = ephys.get("trials", {})
            common = sorted(set(trials) & set(ephys_trials), key=trial_key)
            if not common:
                continue

            for trial_name in common:
                F, t, stim_on = aligned_trace(trials, ephys_trials, trial_name)
                if not trace_is_usable(F, stim_on):
                    continue

                candidate = {
                    "mouse": mouse,
                    "date": date,
                    "block": block,
                    "trial": trial_name,
                    "F": F,
                    "t": t,
                    "stim_on": stim_on,
                }

                if fallback is None:
                    fallback = candidate

                _, _, _, ok = fit_bleach_model(t, F, stim_on, mode="double_exp")
                if ok:
                    selected.append(candidate)
                    break

            if selected and selected[-1]["mouse"] == mouse:
                break

        if (not selected or selected[-1]["mouse"] != mouse) and fallback is not None:
            selected.append(fallback)

        if len(selected) >= MAX_EXAMPLES:
            break

    return selected[:MAX_EXAMPLES]


def compare_example(example: dict) -> dict:
    F = example["F"]
    t = example["t"]
    stim_on = example["stim_on"]

    B_pre, B_current, F_current, current_ok = fit_bleach_model(t, F, stim_on, mode="double_exp")
    if not current_ok:
        B_current = None
        F_current = F.copy()

    B_fixed, F_fixed, fixed_ok = fixed_long_decay_correction(t, F, stim_on)
    if not fixed_ok:
        B_fixed = None
        F_fixed = F.copy()

    current_lift, current_dff_min, current_dff_max = summarize_trace(F, F_current, stim_on)
    fixed_lift, fixed_dff_min, fixed_dff_max = summarize_trace(F, F_fixed, stim_on)
    current_dff = compute_dff(F_current, stim_on)
    fixed_dff = compute_dff(F_fixed, stim_on)
    if current_dff is None or fixed_dff is None:
        max_abs_dff_diff = np.nan
    else:
        max_abs_dff_diff = float(np.nanmax(np.abs(current_dff - fixed_dff)))

    return {
        **example,
        "B_current": B_current,
        "F_current": F_current,
        "current_ok": current_ok,
        "B_fixed": B_fixed,
        "F_fixed": F_fixed,
        "fixed_ok": fixed_ok,
        "current_lift_end": current_lift,
        "fixed_lift_end": fixed_lift,
        "current_dff_min": current_dff_min,
        "current_dff_max": current_dff_max,
        "fixed_dff_min": fixed_dff_min,
        "fixed_dff_max": fixed_dff_max,
        "max_abs_dff_diff": max_abs_dff_diff,
    }


def plot_comparison(results: list[dict]) -> Path:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(17, 3.0 * n), squeeze=False)

    for row, r in enumerate(results):
        F = r["F"]
        t = r["t"]
        stim_on = r["stim_on"]
        label = f"{r['mouse']} | {r['date']} | {r['block']} | {r['trial']}"

        ax = axes[row, 0]
        ax.plot(t, F, color="0.25", lw=1.0, label="raw F")
        if r["B_current"] is not None:
            ax.plot(t, r["B_current"], color="tab:orange", lw=1.6, label="current model")
        if r["B_fixed"] is not None:
            ax.plot(t, r["B_fixed"], color="tab:green", lw=1.6, label="fixed long-decay model")
        ax.axvline(0, color="0.55", lw=1.0, ls="--")
        ax.set_title(label)
        ax.set_ylabel("F")
        ax.legend(fontsize=7, loc="best")

        ax = axes[row, 1]
        ax.plot(t, r["F_current"] - np.nanmean(r["F_current"][:stim_on]), color="tab:orange", lw=1.0, label="current corrected")
        ax.plot(t, r["F_fixed"] - np.nanmean(r["F_fixed"][:stim_on]), color="tab:green", lw=1.0, label="fixed corrected")
        ax.axvline(0, color="0.55", lw=1.0, ls="--")
        ax.set_title("corrected F, pre-mean removed")
        ax.set_ylabel("F - pre mean")
        ax.legend(fontsize=7, loc="best")

        ax = axes[row, 2]
        dff_current = compute_dff(r["F_current"], stim_on)
        dff_fixed = compute_dff(r["F_fixed"], stim_on)
        if dff_current is not None:
            ax.plot(t, 100 * dff_current, color="tab:orange", lw=1.0, label="current dF/F")
        if dff_fixed is not None:
            ax.plot(t, 100 * dff_fixed, color="tab:green", lw=1.0, label="fixed dF/F")
        ax.axvline(0, color="0.55", lw=1.0, ls="--")
        ax.set_title("dF/F comparison")
        ax.set_ylabel("dF/F (%)")
        ax.legend(fontsize=7, loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("time from stim onset (s)")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "photobleach_current_vs_fixed_examples.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_summary(results: list[dict]) -> Path:
    out_path = OUTPUT_DIR / "photobleach_current_vs_fixed_summary.csv"
    fields = [
        "mouse",
        "date",
        "block",
        "trial",
        "n_frames",
        "duration_s",
        "stim_on_frame",
        "current_ok",
        "fixed_ok",
        "current_lift_end",
        "fixed_lift_end",
        "current_dff_min",
        "current_dff_max",
        "fixed_dff_min",
        "fixed_dff_max",
        "max_abs_dff_diff",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            t = r["t"]
            writer.writerow({
                "mouse": r["mouse"],
                "date": r["date"],
                "block": r["block"],
                "trial": r["trial"],
                "n_frames": int(len(r["F"])),
                "duration_s": float(t[-1] - t[0]) if len(t) else np.nan,
                "stim_on_frame": int(r["stim_on"]),
                "current_ok": bool(r["current_ok"]),
                "fixed_ok": bool(r["fixed_ok"]),
                "current_lift_end": r["current_lift_end"],
                "fixed_lift_end": r["fixed_lift_end"],
                "current_dff_min": r["current_dff_min"],
                "current_dff_max": r["current_dff_max"],
                "fixed_dff_min": r["fixed_dff_min"],
                "fixed_dff_max": r["fixed_dff_max"],
                "max_abs_dff_diff": r["max_abs_dff_diff"],
            })
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = discover_examples()
    if not examples:
        raise RuntimeError("No matching imaging/ephys examples found.")

    results = [compare_example(example) for example in examples]
    image_path = plot_comparison(results)
    csv_path = save_summary(results)

    print(f"Saved image: {image_path}")
    print(f"Saved table: {csv_path}")
    print("")
    print("Selected examples:")
    for r in results:
        print(
            f"{r['mouse']} {r['date']} {r['block']} {r['trial']} | "
            f"current_ok={r['current_ok']} | "
            f"current_lift_end={r['current_lift_end']:.3f} | "
            f"fixed_lift_end={r['fixed_lift_end']:.3f} | "
            f"max_abs_dff_diff={r['max_abs_dff_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
