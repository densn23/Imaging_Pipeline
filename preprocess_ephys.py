# epoch_trials_plot_first_with_prints_and_save.py
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_ANALYSIS_ROOT

MOUSE_NAME = None   #change to "Jamie5", etc.
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
        if not (mouse_dir / "Open_Ephys").exists():
            continue
        out.append(mouse_dir.name)
    return out


def resolve_mouse_names(raw: str | None) -> list[str]:
    names = parse_mouse_names(raw)
    return names if names else discover_mouse_names()


_SINGLE_MOUSE_NAME = single_mouse_name(MOUSE_NAME)

SAVE_PKLS = False  # True = save epoched pkls
RUN_BATCH = False # True = run on ALL folders, False = only run the single pkl_in below
OVERWRITE = False # True = replace existing epoched pkls

PLOT_FIRST = True # True = show plots (sanity check)
N_PLOT = 2          # how many trials to plot when PLOT_FIRST=True
MIN_STIM_GAP_MS = 1.0  # reject extra threshold crossings within this interval of a previous pulse
STIM_PULSE_THRESHOLD_V = 0.6  # fixed stim threshold; robust for ramp blocks where early pulses are weaker

# -------------------------
# SETTINGS FOR VELOCITY
# -------------------------
BIN_DURATION_S = 0.020      # 20 ms bins
TICKS_PER_REV = 2048        # transitions per revolution (A rising+falling)
WHEEL_DIAM_M = 0.17         # 17 cm = 0.17 m

# -------------------------
# SINGLE INPUT (when RUN_BATCH=False)
# -------------------------
pkl_in = None if SINGLE_BLOCK is None else (
    None if _SINGLE_MOUSE_NAME is None else (
        DATA_ANALYSIS_ROOT / _SINGLE_MOUSE_NAME / "Open_Ephys" / SINGLE_DATE / SINGLE_BLOCK / f"{SINGLE_BLOCK}_ephys.pkl"
    )
)


def stim_pulse_samples(stim_seg: np.ndarray, fs: float, v_min: float = STIM_PULSE_THRESHOLD_V) -> np.ndarray:
    """
    ALL stim pulse onset sample indices within stim_seg (0..len-1)
    """
    x = np.abs(stim_seg)
    mx = float(np.max(x))
    if mx < v_min:
        return np.array([], dtype=int)

    # Use a fixed absolute threshold so early weaker pulses in ramp blocks
    # are not missed just because later pulses are much larger.
    thr_s = float(v_min)
    b = x > thr_s
    e = np.flatnonzero((~b[:-1]) & (b[1:])) + 1
    if len(e) <= 1:
        return e.astype(int)

    # Collapse duplicate detections caused by fast ringing/re-crossings of the threshold.
    # Real DBS pulses are separated by many ms in these datasets; sub-ms repeats are artifacts.
    min_gap_samp = max(1, int(round((float(MIN_STIM_GAP_MS) / 1000.0) * fs)))
    keep = [int(e[0])]
    for idx in e[1:]:
        if int(idx) - keep[-1] >= min_gap_samp:
            keep.append(int(idx))
    return np.asarray(keep, dtype=int)


def velocity_from_phase_a(
    phase_a_seg: np.ndarray,
    fs: float,
    bin_duration_s: float,
    ticks_per_rev: int,
    wheel_diameter_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_bins_s: time (s) at bin centers, relative to segment start
      vel_mps : velocity (m/s) per bin

    Steps:
      1) threshold phase_a using midpoint(p5,p95)
      2) binarize
      3) count ALL transitions (0->1 and 1->0) as ticks
      4) ticks/bin -> revs/bin -> meters/bin -> m/s
    """
    thr = 0.5 * (np.percentile(phase_a_seg, 5) + np.percentile(phase_a_seg, 95))
    b = (phase_a_seg > thr).astype(np.int8)

    # transitions between samples = ticks
    ticks = (np.diff(b) != 0).astype(np.int8)

    bin_samp = int(round(bin_duration_s * fs))
    bin_samp = max(bin_samp, 1)

    n = len(ticks)
    n_bins = n // bin_samp
    if n_bins < 1:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ticks = ticks[: n_bins * bin_samp]
    ticks_per_bin = ticks.reshape(n_bins, bin_samp).sum(axis=1).astype(np.float64)

    revs_per_bin = ticks_per_bin / float(ticks_per_rev)
    wheel_circ_m = np.pi * float(wheel_diameter_m)
    dist_m_per_bin = revs_per_bin * wheel_circ_m
    vel_mps = dist_m_per_bin / float(bin_duration_s)

    t_bins_s = (np.arange(n_bins) + 0.5) * float(bin_duration_s)
    return t_bins_s.astype(np.float64), vel_mps.astype(np.float64)


def process_one_pkl(pkl_path: Path, date_name: str | None = None):
    block = pkl_path.stem.replace("_ephys", "")
    out_dir = pkl_path.parent
    out_pkl = out_dir / f"{block}_epoched_ephys.pkl"
    label = f"{date_name + ' | ' if date_name else ''}{block}"

    if SAVE_PKLS and out_pkl.exists() and not OVERWRITE:
        print(f"{label}: output exists -> skip")
        return {"status": "skipped_existing", "label": label, "path": str(pkl_path)}
    if SAVE_PKLS and out_pkl.exists() and OVERWRITE:
        print(f"{label}: output exists -> overwrite")

    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        print(f"{label}: could not read {pkl_path.name} -> skip")
        print(f"  {type(e).__name__}: {e}")
        return {"status": "interrupted", "label": label, "path": str(pkl_path), "reason": f"{type(e).__name__}: {e}"}

    fs = float(d["sample_rate"])
    t = d["time"]

    cam = d["channels"]["cam_frame"]
    lfp = d["channels"]["LFP"]
    stim = d["channels"]["stim"]

    phase_a = d["channels"]["phase_a"]
    phase_b = d["channels"]["phase_b"]
    phase_z = d["channels"]["phase_z"]

    # --- cam threshold (5/95) + rising edges ---
    thr = 0.5 * (np.percentile(cam, 5) + np.percentile(cam, 95))
    cam_bin = cam > thr
    edges = np.flatnonzero((~cam_bin[:-1]) & (cam_bin[1:])) + 1
    if len(edges) == 0:
        print(f"{date_name + ' | ' if date_name else ''}{block}: no cam edges -> skip")
        return

    # --- trim to first rising edge ---
    cut0 = int(edges[0])
    t = t[cut0:] - t[cut0]
    cam = cam[cut0:]
    lfp = lfp[cut0:]
    stim = stim[cut0:]
    phase_a = phase_a[cut0:]
    phase_b = phase_b[cut0:]
    phase_z = phase_z[cut0:]

    # --- redo edges on trimmed cam ---
    thr = 0.5 * (np.percentile(cam, 5) + np.percentile(cam, 95))
    cam_bin = cam > thr
    edges = np.flatnonzero((~cam_bin[:-1]) & (cam_bin[1:])) + 1
    if len(edges) == 0:
        print(f"{date_name + ' | ' if date_name else ''}{block}: no cam edges after trim -> skip")
        return

    # --- trial starts by gap > 0.2 s ---
    gap_sec = 0.2
    gap_samp = int(round(gap_sec * fs))
    gaps = np.diff(edges)
    breaks = np.flatnonzero(gaps > gap_samp)

    trial_start = edges[np.r_[0, breaks + 1]]
    trial_end = np.r_[trial_start[1:], len(t)]

    print(f"\n{date_name + ' | ' if date_name else ''}{block}: trials={len(trial_start)}")
    print("Trial | cam_on(s) | stim_on(s) | diff(s) | cam_fps(Hz) | n_pulses | median_IPI(ms)")
    print("--------------------------------------------------------------------------------------")

    all_trials = {}

    for i in range(len(trial_start)):
        a = int(trial_start[i])
        b = int(trial_end[i])

        cam_on = float(t[a])

        pulse_idx = stim_pulse_samples(stim[a:b], fs=fs, v_min=STIM_PULSE_THRESHOLD_V)
        j = int(pulse_idx[0]) if len(pulse_idx) else None

        stim_on = float(t[a + j]) if j is not None else np.nan
        stim_off = float(t[a + pulse_idx[-1]]) if len(pulse_idx) else np.nan
        diff = stim_on - cam_on if np.isfinite(stim_on) else np.nan

        e_tr = edges[(edges >= a) & (edges < b)]
        if len(e_tr) >= 3:
            med_gap_samp = np.median(np.diff(e_tr))
            cam_fps = fs / med_gap_samp
        else:
            cam_fps = np.nan

        # t=0 at stim onset (if present)
        t0 = float(t[a + j]) if j is not None else float(t[a])
        t_trial = (t[a:b] - t0).astype(np.float64)
        cam_frame_times_s = (t[e_tr] - t0).astype(np.float64)
        if len(cam_frame_times_s):
            stim_on_frame_idx = int(np.searchsorted(cam_frame_times_s, 0.0, side="left"))
            if np.isfinite(stim_off):
                stim_off_rel = stim_off - t0
                stim_off_frame_idx = int(np.searchsorted(cam_frame_times_s, stim_off_rel, side="right") - 1)
                stim_off_frame_idx = max(stim_off_frame_idx, stim_on_frame_idx)
            else:
                stim_off_frame_idx = -1
        else:
            stim_on_frame_idx = -1
            stim_off_frame_idx = -1

        # -------------------------
        # VELOCITY (computed on same segment, then shifted to t_trial axis)
        # -------------------------
        vel_t_s, vel_mps = velocity_from_phase_a(
            phase_a[a:b],
            fs=fs,
            bin_duration_s=BIN_DURATION_S,
            ticks_per_rev=TICKS_PER_REV,
            wheel_diameter_m=WHEEL_DIAM_M,
        )
        # vel_t_s is relative to segment start (= t[a]); shift to be relative to stim onset (= t0)
        vel_t_trial_s = vel_t_s + (t[a] - t0)

        # median IPI
        if len(pulse_idx) >= 2:
            pulse_times_s = t_trial[pulse_idx]
            ipi_s = np.diff(pulse_times_s)
            median_ipi_ms = 1000.0 * float(np.median(ipi_s))
        else:
            median_ipi_ms = np.nan

        print(f"{i+1:>5} | {cam_on:>8.3f} | {stim_on:>9.3f} | {diff:>7.3f} | {cam_fps:>10.1f} |"
              f" {len(pulse_idx):>8} | {median_ipi_ms:>13.3f}")

        trial_key = f"{block}_{i+1}"
        trial_dict = {
            "trial_key": trial_key,
            "block": block,
            "trial_num": int(i + 1),
            "sample_rate": fs,

            "cam_on_s_block": cam_on,
            "stim_on_s_block": stim_on,
            "stim_off_s_block": stim_off,
            "stim_minus_cam_s": diff,
            "cam_fps_hz": cam_fps,

            "t0_block_s": t0,
            "start_sample_in_trimmed": a,
            "end_sample_in_trimmed": b,

            "t_stim_s": t_trial,
            "cam_frame_times_stim_s": cam_frame_times_s,
            "stim_on_frame_idx": stim_on_frame_idx,
            "stim_off_frame_idx": stim_off_frame_idx,

            "median_ipi_ms": float(median_ipi_ms) if np.isfinite(median_ipi_ms) else np.nan,

            # stim pulses
            "stim_pulse_samples_in_trial": pulse_idx.astype(int),
            "stim_pulse_times_s": t_trial[pulse_idx].astype(np.float64) if len(pulse_idx) else np.array([], dtype=np.float64),

            # velocity (aligned to t_trial axis)
            "vel_bin_t_s": vel_t_trial_s,
            "vel_bin_mps": vel_mps,
            "vel_bin_cmps": 100.0 * vel_mps,

            "vel_bin_duration_s": float(BIN_DURATION_S),
            "vel_ticks_per_rev": int(TICKS_PER_REV),
            "vel_wheel_diameter_m": float(WHEEL_DIAM_M),

            "channels": {
                "cam_frame": cam[a:b],
                "stim": stim[a:b],
                "LFP": lfp[a:b],
                "phase_a": phase_a[a:b],
                "phase_b": phase_b[a:b],
                "phase_z": phase_z[a:b],
            },
        }

        all_trials[trial_key] = trial_dict

    # plots (first N_PLOT trials) â€” now optional
    if PLOT_FIRST:
        n_plot = min(N_PLOT, len(trial_start))
        for i in range(n_plot):
            a = int(trial_start[i])
            b = int(trial_end[i])

            pulse_idx = stim_pulse_samples(stim[a:b], fs=fs, v_min=STIM_PULSE_THRESHOLD_V)
            j = int(pulse_idx[0]) if len(pulse_idx) else None
            t0 = float(t[a + j]) if j is not None else float(t[a])
            t_trial = t[a:b] - t0  # time from stim onset

            vel_t_s, vel_mps = velocity_from_phase_a(
                phase_a[a:b],
                fs=fs,
                bin_duration_s=BIN_DURATION_S,
                ticks_per_rev=TICKS_PER_REV,
                wheel_diameter_m=WHEEL_DIAM_M,
            )
            vel_t_trial_s = vel_t_s + (t[a] - t0)

            fig, ax = plt.subplots(7, 1, sharex=True, figsize=(14, 12))

            ax[0].plot(t_trial, cam[a:b]);     ax[0].set_title(f"{block}_{i+1} cam_frame"); ax[0].set_ylabel("V")
            ax[1].plot(t_trial, lfp[a:b]);     ax[1].set_title(f"{block}_{i+1} LFP");      ax[1].set_ylabel("V")
            ax[2].plot(t_trial, stim[a:b]);    ax[2].set_title(f"{block}_{i+1} stim");     ax[2].set_ylabel("V")
            ax[3].plot(t_trial, phase_a[a:b]); ax[3].set_title(f"{block}_{i+1} phase_a");  ax[3].set_ylabel("V")
            ax[4].plot(t_trial, phase_b[a:b]); ax[4].set_title(f"{block}_{i+1} phase_b");  ax[4].set_ylabel("V")
            ax[5].plot(t_trial, phase_z[a:b]); ax[5].set_title(f"{block}_{i+1} phase_z");  ax[5].set_ylabel("V")

            if len(vel_t_trial_s) > 0:
                ax[6].plot(vel_t_trial_s, 100.0 * vel_mps)
            ax[6].set_title(f"{block}_{i+1} velocity (from phase_a, {int(BIN_DURATION_S*1000)} ms bins)")
            ax[6].set_ylabel("cm/s")
            ax[6].set_xlabel("time from stim onset (s)")

            plt.tight_layout()
            plt.show()

    if SAVE_PKLS:
        block_dict = {
            "block": block,
            "sample_rate": fs,
            "n_trials": int(len(all_trials)),
            "trials": all_trials,
        }
        with open(out_pkl, "wb") as f:
            pickle.dump(block_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved {len(all_trials)} epoched trials to:\n{out_pkl}")
        return {"status": "saved", "label": label, "path": str(out_pkl), "n_trials": int(len(all_trials))}

    else:
        print("\nSAVE_PKLS=False (no files written)")
        return {"status": "processed", "label": label, "path": str(pkl_path), "n_trials": int(len(all_trials))}


def run_single_date(mouse_name: str, date_name: str) -> list[dict]:
    import re

    results = []
    open_ephys_root = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / date_name
    block_pattern = re.compile(r"^R\d+$")
    if not open_ephys_root.exists():
        print(f"Date folder not found: {open_ephys_root}")
        return [{"status": "failed", "label": f"{date_name}", "reason": "date folder not found"}]

    for block_dir in sorted(open_ephys_root.iterdir()):
        if not block_dir.is_dir():
            continue
        if not block_pattern.match(block_dir.name):
            continue

        pkl_path = block_dir / f"{block_dir.name}_ephys.pkl"
        if not pkl_path.exists():
            continue

        try:
            result = process_one_pkl(pkl_path, date_name)
        except Exception as e:
            label = f"{date_name} | {block_dir.name}"
            print(f"{label}: unexpected error -> skip")
            print(f"  {type(e).__name__}: {e}")
            result = {"status": "interrupted", "label": label, "path": str(pkl_path), "reason": f"{type(e).__name__}: {e}"}

        if result is not None:
            results.append(result)
    return results


def run_all_dates(mouse_name: str) -> list[dict]:
    import re

    open_ephys_root = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys"
    block_pattern = re.compile(r"^R\d+$")
    if not open_ephys_root.exists():
        print(f"Mouse folder not found: {open_ephys_root}")
        return []

    batch_results = []
    for date_dir in sorted(open_ephys_root.iterdir()):
        if not date_dir.is_dir():
            continue

        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir():
                continue
            if not block_pattern.match(block_dir.name):
                continue

            pkl_path = block_dir / f"{block_dir.name}_ephys.pkl"
            if not pkl_path.exists():
                continue

            try:
                result = process_one_pkl(pkl_path, date_dir.name)
            except KeyboardInterrupt:
                label = f"{date_dir.name} | {block_dir.name}"
                print(f"{label}: interrupted by user")
                batch_results.append({"status": "interrupted", "label": label, "path": str(pkl_path), "reason": "KeyboardInterrupt"})
                raise
            except Exception as e:
                label = f"{date_dir.name} | {block_dir.name}"
                print(f"{label}: unexpected error -> skip")
                print(f"  {type(e).__name__}: {e}")
                result = {"status": "interrupted", "label": label, "path": str(pkl_path), "reason": f"{type(e).__name__}: {e}"}

            if result is not None:
                batch_results.append(result)
    return batch_results


def run_single_block(mouse_name: str, date_name: str, block_name: str):
    pkl_path = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / date_name / block_name / f"{block_name}_ephys.pkl"
    if not pkl_path.exists():
        print(f"Block folder not found: {pkl_path.parent}")
        return None
    return process_one_pkl(pkl_path, date_name)


def main() -> None:
    mouse_names = resolve_mouse_names(MOUSE_NAME)
    if not mouse_names:
        print("No mice found to process.")
        return

    batch_results = []
    if not RUN_BATCH:
        if SINGLE_DATE is not None and SINGLE_BLOCK is None:
            for mouse_name in mouse_names:
                batch_results.extend(run_single_date(mouse_name, SINGLE_DATE))
        elif SINGLE_DATE is not None and SINGLE_BLOCK is not None:
            for mouse_name in mouse_names:
                result = run_single_block(mouse_name, SINGLE_DATE, SINGLE_BLOCK)
                if result is not None:
                    batch_results.append(result)
        else:
            print("Set SINGLE_BLOCK for single-block mode, or set SINGLE_DATE to run all blocks from one date.")
            return
    else:
        for mouse_name in mouse_names:
            batch_results.extend(run_all_dates(mouse_name))

    if batch_results:
        saved = [r for r in batch_results if r["status"] in {"saved", "processed"}]
        skipped = [r for r in batch_results if r["status"] == "skipped_existing"]
        interrupted = [r for r in batch_results if r["status"] == "interrupted"]

        print("\nBatch summary")
        print(f"  completed: {len(saved)}")
        print(f"  skipped existing: {len(skipped)}")
        print(f"  interrupted: {len(interrupted)}")

        if skipped:
            print("\nSkipped existing blocks:")
            for r in skipped:
                print(f"  {r['label']}")

        if interrupted:
            print("\nInterrupted blocks:")
            for r in interrupted:
                print(f"  {r['label']} | {r.get('reason', '')}")


if __name__ == "__main__":
    main()


