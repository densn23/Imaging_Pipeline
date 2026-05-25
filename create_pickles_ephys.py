from pathlib import Path
import pickle
import re

import numpy as np
from open_ephys.analysis import Session
from config import DATA_ANALYSIS_ROOT


TARGET_MOUSE = None   # set to None to process all mice
SINGLE_DATE = None      # e.g. "23-12-25"
SINGLE_BLOCK = None        # e.g. "R4"
SELECT_BLOCKS = []         # e.g. [("23-12-25", "R4"), ("08-01-26", "R7")]
OVERWRITE = False



BLOCK_PATTERN = re.compile(r"^R\d+$")


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


def process_block(block_dir: Path, mouse_name: str, date_name: str) -> dict:
    block_name = block_dir.name
    label = f"{mouse_name} | {date_name} | {block_name}"
    print(f"    Block: {block_name}")

    out_pkl = block_dir / f"{block_name}_ephys.pkl"
    if out_pkl.exists() and not OVERWRITE:
        print(f"      {out_pkl.name} already exists -> SKIPPING")
        return {"status": "skipped_existing", "label": label}
    if out_pkl.exists() and OVERWRITE:
        print(f"      {out_pkl.name} already exists -> OVERWRITING")

    structure_path = block_dir / "Record Node 104" / "structure.openephys"
    if not structure_path.exists():
        print("      WARNING: structure.openephys is missing in Record Node 104.")
        print("               This block may have raw data files but incomplete Open Ephys metadata.")

    try:
        print(f"      Recording folder: {block_dir}")
        session = Session(block_dir)
    except Exception as e:
        print(f"      WARNING: could not load Session: {e}")
        return {"status": "failed", "label": label, "reason": f"could not load Session: {e}"}

    if not session.recordnodes:
        print("      WARNING: no record nodes found, skipping.")
        return {"status": "failed", "label": label, "reason": "no record nodes found"}

    rn = session.recordnodes[0]
    if not rn.recordings:
        print("      WARNING: no recordings in record node, skipping.")
        return {"status": "failed", "label": label, "reason": "no recordings in record node"}

    rec = rn.recordings[0]
    if not rec.continuous:
        print("      WARNING: no continuous streams, skipping.")
        return {"status": "failed", "label": label, "reason": "no continuous streams"}

    cont = rec.continuous[0]
    meta = cont.metadata
    sample_rate = float(meta.sample_rate)
    channel_names = list(meta.channel_names)
    bit_volts = list(meta.bit_volts)

    print(f"      sample_rate: {sample_rate}")
    print(f"      channel_names: {channel_names}")

    try:
        samples = cont.samples
        timestamps = cont.timestamps
    except FileNotFoundError as e:
        print("      WARNING: could not load continuous data for this block.")
        print(f"               {e}")
        print("               Skipping this block.")
        return {"status": "failed", "label": label, "reason": f"could not load continuous data: {e}"}

    if not isinstance(samples, np.ndarray):
        samples = np.asarray(samples)
    if not isinstance(timestamps, np.ndarray):
        timestamps = np.asarray(timestamps)

    n_samples, n_channels = samples.shape
    print(f"      samples shape: {samples.shape}")
    print(f"      timestamps shape: {timestamps.shape}")

    name_to_idx = {name: i for i, name in enumerate(channel_names)}

    def get_chan(name: str) -> np.ndarray:
        idx = name_to_idx[name]
        return samples[:, idx].astype(np.float32) * float(bit_volts[idx])

    try:
        lfp = get_chan("CH11")
        adc5 = get_chan("ADC5")
        adc6 = get_chan("ADC6")
        trial_sig = get_chan("ADC7")
        cam_frame = get_chan("ADC1")
        phase_z = get_chan("ADC2")
        phase_b = get_chan("ADC3")
        phase_a = get_chan("ADC4")
    except KeyError as e:
        print(f"      ERROR: missing channel {e}, skipping this block.")
        return {"status": "failed", "label": label, "reason": f"missing channel {e}"}

    stim = adc5 + adc6
    time = timestamps - timestamps[0]

    block_ephys = {
        "mouse": mouse_name,
        "date": date_name,
        "block": block_name,
        "sample_rate": sample_rate,
        "n_samples": int(n_samples),
        "time": time.astype(np.float64),
        "channel_names": channel_names,
        "name_to_idx": name_to_idx,
        "bit_volts": bit_volts,
        "channels": {
            "LFP": lfp,
            "trial": trial_sig,
            "cam_frame": cam_frame,
            "stim": stim,
            "phase_a": phase_a,
            "phase_b": phase_b,
            "phase_z": phase_z,
        },
    }

    print("      CHANNELS SAVED:", list(block_ephys["channels"].keys()))
    print(f"      Saving: {out_pkl}")
    with open(out_pkl, "wb") as f:
        pickle.dump(block_ephys, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("      Done with this block.")
    return {"status": "saved", "label": label}


def run_single_block(mouse_name: str, date_name: str, block_name: str) -> list[dict]:
    block_dir = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / date_name / block_name
    if not block_dir.exists():
        print(f"Block folder not found: {block_dir}")
        return [{"status": "failed", "label": f"{mouse_name} | {date_name} | {block_name}", "reason": "block folder not found"}]

    print(f"\nMouse: {mouse_name}")
    print(f"  Date: {date_name}")
    return [process_block(block_dir, mouse_name, date_name)]


def run_single_date(mouse_name: str, date_name: str) -> list[dict]:
    results = []
    date_dir = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / date_name
    if not date_dir.exists():
        print(f"Date folder not found: {date_dir}")
        return [{"status": "failed", "label": f"{mouse_name} | {date_name}", "reason": "date folder not found"}]

    print(f"\nMouse: {mouse_name}")
    print(f"  Date: {date_name}")
    for block_dir in sorted(date_dir.iterdir()):
        if not block_dir.is_dir():
            continue
        if not BLOCK_PATTERN.match(block_dir.name):
            continue
        try:
            results.append(process_block(block_dir, mouse_name, date_name))
        except Exception as e:
            print(f"      WARNING: unexpected error, skipping block: {e}")
            results.append({
                "status": "failed",
                "label": f"{mouse_name} | {date_name} | {block_dir.name}",
                "reason": f"unexpected error: {type(e).__name__}: {e}",
            })
    return results


def run_selected_blocks(mouse_name: str, selected_blocks: list[tuple[str, str]]) -> list[dict]:
    results = []
    print(f"\nMouse: {mouse_name}")
    for date_name, block_name in selected_blocks:
        block_dir = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / date_name / block_name
        if not block_dir.exists():
            print(f"  Missing block folder: {block_dir}")
            results.append({
                "status": "failed",
                "label": f"{mouse_name} | {date_name} | {block_name}",
                "reason": "block folder not found",
            })
            continue

        print(f"  Date: {date_name}")
        try:
            results.append(process_block(block_dir, mouse_name, date_name))
        except Exception as e:
            print(f"      WARNING: unexpected error, skipping block: {e}")
            results.append({
                "status": "failed",
                "label": f"{mouse_name} | {date_name} | {block_name}",
                "reason": f"unexpected error: {type(e).__name__}: {e}",
            })
    return results


def run_all(mouse_names: list[str] | None = None) -> list[dict]:
    results = []
    names = mouse_names if mouse_names is not None else resolve_mouse_names(TARGET_MOUSE)
    for mouse_name in names:
        open_ephys_root = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys"
        if not open_ephys_root.exists():
            continue

        print(f"\nMouse: {mouse_name}")
        for date_dir in sorted(open_ephys_root.iterdir()):
            if not date_dir.is_dir():
                continue

            print(f"  Date: {date_dir.name}")
            for block_dir in sorted(date_dir.iterdir()):
                if not block_dir.is_dir():
                    continue
                if not BLOCK_PATTERN.match(block_dir.name):
                    continue
                try:
                    results.append(process_block(block_dir, mouse_name, date_dir.name))
                except Exception as e:
                    print(f"      WARNING: unexpected error, skipping block: {e}")
                    results.append({
                        "status": "failed",
                        "label": f"{mouse_name} | {date_dir.name} | {block_dir.name}",
                        "reason": f"unexpected error: {type(e).__name__}: {e}",
                    })
    return results


def print_summary(results: list[dict]) -> None:
    if not results:
        return

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("\nRun summary")
    for key in sorted(counts.keys()):
        print(f"  {key}: {counts[key]}")

    not_processed = [r for r in results if r["status"] != "saved"]
    if not_processed:
        print("\nBlocks not processed:")
        for r in not_processed:
            reason = r.get("reason", r["status"])
            print(f"  {r['label']} | {reason}")


def main() -> None:
    mouse_names = resolve_mouse_names(TARGET_MOUSE)
    if not mouse_names:
        print("No mice found to process.")
        return

    summary_results = []
    if SELECT_BLOCKS:
        for mouse_name in mouse_names:
            summary_results.extend(run_selected_blocks(mouse_name, SELECT_BLOCKS))
    elif SINGLE_DATE is not None and SINGLE_BLOCK is not None:
        for mouse_name in mouse_names:
            summary_results.extend(run_single_block(mouse_name, SINGLE_DATE, SINGLE_BLOCK))
    elif SINGLE_DATE is not None:
        for mouse_name in mouse_names:
            summary_results.extend(run_single_date(mouse_name, SINGLE_DATE))
    else:
        summary_results = run_all(mouse_names)
    print_summary(summary_results)


if __name__ == "__main__":
    main()
