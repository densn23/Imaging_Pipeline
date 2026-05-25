from pathlib import Path
import pickle
import re

import numpy as np
from tifffile import imread
from config import DATA_ANALYSIS_ROOT


TARGET_MOUSE = None  # set to None to process all mice
SINGLE_DATE = None # e.g. "23-12-25"
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
        if not (mouse_dir / "Imaging_Data").exists():
            continue
        out.append(mouse_dir.name)
    return out


def resolve_mouse_names(raw: str | None) -> list[str]:
    names = parse_mouse_names(raw)
    return names if names else discover_mouse_names()


def find_tiff(trial_dir: Path) -> Path | None:
    mmstack_files = sorted(trial_dir.glob("*MMStack*.ome.tif"))
    if mmstack_files:
        return mmstack_files[0]

    ome_files = sorted(trial_dir.glob("*.ome.tif"))
    if ome_files:
        return ome_files[0]

    return None


def process_block(block_dir: Path, mouse_name: str, date_name: str) -> dict:
    block_name = block_dir.name
    label = f"{mouse_name} | {date_name} | {block_name}"
    print(f"    Block: {block_name}")

    out_path = block_dir / f"{block_name}_traces.pkl"
    if out_path.exists() and not OVERWRITE:
        print(f"      {out_path.name} already exists -> SKIPPING")
        return {"status": "skipped_existing", "label": label}
    if out_path.exists() and OVERWRITE:
        print(f"      {out_path.name} already exists -> OVERWRITING")

    block_data = {
        "mouse": mouse_name,
        "date": date_name,
        "block": block_name,
        "trials": {},
    }

    for trial_dir in sorted(block_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        trial_name = trial_dir.name
        print(f"      Trial: {trial_name}")

        tiff_path = find_tiff(trial_dir)
        if tiff_path is None:
            print(f"        WARNING: no .ome.tif found in {trial_dir}")
            continue

        arr = imread(tiff_path)
        if arr.ndim != 3:
            print(f"        WARNING: expected 3D array, got shape {arr.shape}")
            continue

        n_frames, y_px, x_px = arr.shape
        trace = arr.mean(axis=(1, 2)).astype(np.float32)

        block_data["trials"][trial_name] = {
            "tiff_path": str(tiff_path),
            "n_frames": int(n_frames),
            "frame_shape": (int(y_px), int(x_px)),
            "trace_raw": trace,
        }

    if not block_data["trials"]:
        print(f"      No trials with data found for block {block_name}, skipping pickle.")
        return {"status": "failed", "label": label, "reason": "no trials with valid TIFF data"}

    print(f"      Saving: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(block_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"status": "saved", "label": label}


def run_selected_blocks(mouse_name: str, selected_blocks: list[tuple[str, str]]) -> list[dict]:
    results = []
    print(f"\nMouse: {mouse_name}")
    for date_name, block_name in selected_blocks:
        block_dir = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data" / date_name / block_name
        if not block_dir.exists():
            print(f"  Missing block folder: {block_dir}")
            results.append({
                "status": "failed",
                "label": f"{mouse_name} | {date_name} | {block_name}",
                "reason": "block folder not found",
            })
            continue

        print(f"  Date: {date_name}")
        results.append(process_block(block_dir, mouse_name, date_name))
    return results


def run_single_date(mouse_name: str, date_name: str) -> list[dict]:
    results = []
    date_dir = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data" / date_name
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
        results.append(process_block(block_dir, mouse_name, date_name))
    return results


def run_all(mouse_names: list[str] | None = None) -> list[dict]:
    results = []
    names = mouse_names if mouse_names is not None else resolve_mouse_names(TARGET_MOUSE)
    for mouse_name in names:
        imaging_root = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data"
        if not imaging_root.exists():
            continue

        print(f"\nMouse: {mouse_name}")
        for date_dir in sorted(imaging_root.iterdir()):
            if not date_dir.is_dir():
                continue

            print(f"  Date: {date_dir.name}")
            for block_dir in sorted(date_dir.iterdir()):
                if not block_dir.is_dir():
                    continue
                if not BLOCK_PATTERN.match(block_dir.name):
                    continue
                results.append(process_block(block_dir, mouse_name, date_dir.name))
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
            summary_results.extend(run_selected_blocks(mouse_name, [(SINGLE_DATE, SINGLE_BLOCK)]))
    elif SINGLE_DATE is not None:
        for mouse_name in mouse_names:
            summary_results.extend(run_single_date(mouse_name, SINGLE_DATE))
    else:
        summary_results = run_all(mouse_names)
    print_summary(summary_results)


if __name__ == "__main__":
    main()
