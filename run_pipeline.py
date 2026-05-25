from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = "Vinnie1"
SINGLE_DATE = None
SINGLE_BLOCK = None
OVERWRITE = False
PIPELINE_STAGES = "create_data, create_ephys, preprocess_data, preprocess_ephys, filter, pta, pta_mean, pulsogram, summarize, entrain_anal"  # None = run all; e.g. "filter, pta_mean, summarize"


BLOCK_RE = __import__("re").compile(r"^R\d+$")


STAGE_ORDER = [
    "create_data",
    "create_ephys",
    "preprocess_ephys",
    "preprocess_data",
    "filter",
    "pta",
    "pta_mean",
    "pulsogram",
    "summarize",
    "entrain_anal",
]


STAGE_ALIASES = {
    "create": ["create_data", "create_ephys"],
    "create_data": ["create_data"],
    "create_imaging": ["create_data"],
    "create_ephys": ["create_ephys"],
    "preprocess": ["preprocess_ephys", "preprocess_data"],
    "preprocess_ephys": ["preprocess_ephys"],
    "preprocess_data": ["preprocess_data"],
    "filter": ["filter"],
    "pta": ["pta"],
    "pta_first": ["pta"],
    "pta_mean": ["pta_mean"],
    "pta_train": ["pta_mean"],
    "pulsogram": ["pulsogram"],
    "summarize": ["summarize"],
    "summary": ["summarize"],
    "entrain_anal": ["entrain_anal"],
    "entrainment": ["entrain_anal"],
    "stats": ["entrain_anal"],
}


def parse_mice_arg(mouse_arg: str) -> list[str]:
    if mouse_arg is None:
        return []
    text = str(mouse_arg).strip()
    if text == "" or text.lower() in {"none", "all"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_stages_arg(stage_arg: str | None) -> list[str]:
    if stage_arg is None:
        return []
    text = str(stage_arg).strip()
    if text == "" or text.lower() in {"none", "all"}:
        return []

    selected: list[str] = []
    seen = set()
    for part in [p.strip().lower() for p in text.split(",") if p.strip()]:
        if part not in STAGE_ALIASES:
            valid = ", ".join(STAGE_ORDER)
            raise ValueError(f"Unknown pipeline stage: {part}. Valid stages: {valid}")
        for stage_name in STAGE_ALIASES[part]:
            if stage_name not in seen:
                selected.append(stage_name)
                seen.add(stage_name)
    return selected


def resolve_stage_names(stage_arg: str | None) -> list[str]:
    selected = parse_stages_arg(stage_arg)
    if not selected:
        return list(STAGE_ORDER)
    selected_set = set(selected)
    return [stage_name for stage_name in STAGE_ORDER if stage_name in selected_set]


def discover_mice() -> list[str]:
    out = []
    for mouse_dir in sorted(DATA_ANALYSIS_ROOT.iterdir()):
        if not mouse_dir.is_dir():
            continue
        if not (mouse_dir / "Imaging_Data").exists():
            continue
        out.append(mouse_dir.name)
    return out


def block_paths(mouse: str, date: str, block: str) -> dict[str, Path]:
    img_block = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data" / date / block
    eph_block = DATA_ANALYSIS_ROOT / mouse / "Open_Ephys" / date / block
    return {
        "img_block": img_block,
        "eph_block": eph_block,
        "traces": img_block / f"{block}_traces.pkl",
        "ephys": eph_block / f"{block}_ephys.pkl",
        "epoched_ephys": eph_block / f"{block}_epoched_ephys.pkl",
        "processed": img_block / f"{block}_traces_processed.pkl",
        "notched": img_block / f"{block}_traces_processed_notched.pkl",
        "pta_first": img_block / f"{block}_traces_processed_notched_pta_first_pulse.pkl",
        "pta_train": img_block / f"{block}_traces_processed_notched_pta_train.pkl",
        "pulsogram": img_block / f"{block}_traces_processed_notched_pulsogram.pkl",
        "summary": img_block / f"{block}_summary.pkl",
        "entrain_anal": img_block / f"{block}_entrainment_analysis.pkl",
    }


def discover_blocks(mouse: str, date: str | None = None, block: str | None = None) -> list[tuple[str, str]]:
    imaging_root = DATA_ANALYSIS_ROOT / mouse / "Imaging_Data"
    if not imaging_root.exists():
        raise FileNotFoundError(f"Imaging root not found: {imaging_root}")

    if date is not None and block is not None:
        return [(date, block)]

    if date is not None:
        date_dir = imaging_root / date
        if not date_dir.exists():
            raise FileNotFoundError(f"Date folder not found: {date_dir}")
        return [(date, d.name) for d in sorted(date_dir.iterdir()) if d.is_dir() and BLOCK_RE.match(d.name)]

    out: list[tuple[str, str]] = []
    for date_dir in sorted(imaging_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if block_dir.is_dir() and BLOCK_RE.match(block_dir.name):
                out.append((date_dir.name, block_dir.name))
    return out


def print_stage(label: str, status: str) -> None:
    print(f"  [{status}] {label}")


def run_stage(stage_name: str, fn, *args, **kwargs) -> str:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        status = f"failed: {type(e).__name__}: {e}"
        print_stage(stage_name, status)
        return status


def stage_create_data(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import create_pickles_data
    p = block_paths(mouse, date, block)
    if p["img_block"].exists():
        create_pickles_data.OVERWRITE = overwrite
        r = create_pickles_data.process_block(p["img_block"], mouse, date)
        return r["status"]
    return "missing_imaging_block"


def stage_create_ephys(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import create_pickles_ephys

    p = block_paths(mouse, date, block)
    if p["eph_block"].exists():
        create_pickles_ephys.OVERWRITE = overwrite
        r = create_pickles_ephys.process_block(p["eph_block"], mouse, date)
        return r["status"]
    return "missing_ephys_block"


def stage_preprocess_ephys(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import preprocess_ephys

    p = block_paths(mouse, date, block)
    if not p["ephys"].exists():
        return "missing_input"
    preprocess_ephys.SAVE_PKLS = True
    preprocess_ephys.PLOT_FIRST = False
    if overwrite and p["epoched_ephys"].exists():
        p["epoched_ephys"].unlink()
    r = preprocess_ephys.process_one_pkl(p["ephys"], date)
    return r["status"]


def stage_preprocess_data(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import preprocess_data

    p = block_paths(mouse, date, block)
    if not p["traces"].exists() or not p["epoched_ephys"].exists():
        return "missing_input"
    if p["processed"].exists() and not overwrite:
        return "skipped_existing"
    preprocess_data.run_single_block(p["traces"], p["epoched_ephys"], save_output=True, show_plots=False)
    return "saved"


def stage_filter(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import process_data_filter

    p = block_paths(mouse, date, block)
    if not p["processed"].exists():
        return "missing_input"
    if p["notched"].exists() and not overwrite:
        return "skipped_existing"
    process_data_filter.SAVE_OUTPUT = True
    process_data_filter.SHOW_PLOTS = False
    process_data_filter.SHOW_POST_NOTCH_PSD = False
    process_data_filter.SHOW_POST_NOTCH_BAND_POWER = False
    process_data_filter.run_single(p["processed"])
    return "saved"


def stage_pta_first(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import process_data_PTA

    p = block_paths(mouse, date, block)
    if not p["notched"].exists() or not p["epoched_ephys"].exists():
        return "missing_input"
    if p["pta_first"].exists() and not overwrite:
        return "skipped_existing"
    process_data_PTA.SAVE_OUTPUT = True
    process_data_PTA.SHOW_PLOTS = False
    process_data_PTA.run_single(p["notched"], p["epoched_ephys"])
    return "saved"


def stage_pta_train(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import process_data_PTA_mean

    p = block_paths(mouse, date, block)
    if not p["notched"].exists() or not p["epoched_ephys"].exists():
        return "missing_input"
    if p["pta_train"].exists() and not overwrite:
        return "skipped_existing"
    process_data_PTA_mean.SAVE_OUTPUT = True
    process_data_PTA_mean.SHOW_PLOTS = False
    process_data_PTA_mean.run_single(p["notched"], p["epoched_ephys"])
    return "saved"


def stage_pulsogram(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import process_data_pulsogram

    p = block_paths(mouse, date, block)
    if not p["notched"].exists() or not p["epoched_ephys"].exists():
        return "missing_input"
    if p["pulsogram"].exists() and not overwrite:
        return "skipped_existing"
    process_data_pulsogram.SAVE_OUTPUT = True
    process_data_pulsogram.SHOW_PLOTS = False
    process_data_pulsogram.run_single(p["notched"], p["epoched_ephys"])
    return "saved"


def stage_summary(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import summarize

    summarize.OVERWRITE = overwrite
    summarize.SAVE_OUTPUT = True
    r = summarize.process_block(mouse, date, block)
    return r["status"]


def stage_entrain_anal(mouse: str, date: str, block: str, overwrite: bool) -> str:
    import entrain_anal

    p = block_paths(mouse, date, block)
    if not p["notched"].exists() or not p["epoched_ephys"].exists():
        return "missing_input"
    if p["entrain_anal"].exists() and not overwrite:
        return "skipped_existing"

    entrain_anal.ANALYSIS_MODE = "block"
    entrain_anal.MOUSE_NAME = mouse
    entrain_anal.SINGLE_DATE = date
    entrain_anal.SINGLE_BLOCK = block
    entrain_anal.ONLY_TRIAL = None
    entrain_anal.RUN_BATCH = False
    entrain_anal.SHOW_FIGURES = False
    entrain_anal.PRINT_TRIAL_TABLE = False
    entrain_anal.SAVE_OUTPUT = True
    entrain_anal.SHOW_EPHYS_CHECK_SUMMARY = False

    try:
        entrain_anal.run_current_block(show_figures=False)
    except SystemExit as e:
        return str(e) or "no_usable_trials"
    return "saved"


def run_block(mouse: str, date: str, block: str, overwrite: bool, stage_names: list[str]) -> dict[str, str]:
    label = f"{date} | {block}"
    print(f"\n[BLOCK] {label}")

    results: dict[str, str] = {}

    stage_defs = [
        ("create_data", "create_pickles_data", stage_create_data),
        ("create_ephys", "create_pickles_ephys", stage_create_ephys),
        ("preprocess_ephys", "preprocess_ephys", stage_preprocess_ephys),
        ("preprocess_data", "preprocess_data", stage_preprocess_data),
        ("filter", "process_data_filter", stage_filter),
        ("pta", "process_data_PTA", stage_pta_first),
        ("pta_mean", "process_data_PTA_mean", stage_pta_train),
        ("pulsogram", "process_data_pulsogram", stage_pulsogram),
        ("summarize", "summarize", stage_summary),
        ("entrain_anal", "entrain_anal", stage_entrain_anal),
    ]

    for key, label_name, fn in stage_defs:
        if key not in stage_names:
            continue
        results[key] = run_stage(label_name, fn, mouse, date, block, overwrite)
        print_stage(label_name, results[key])

    return results


def print_final_summary(all_results: dict[str, dict[str, str]], stage_names: list[str]) -> None:
    print("\nPipeline summary")
    n_blocks = len(all_results)
    print(f"  blocks processed: {n_blocks}")
    if not all_results:
        return

    stage_counts: dict[str, dict[str, int]] = {}
    for block_res in all_results.values():
        for stage, status in block_res.items():
            stage_counts.setdefault(stage, {})
            stage_counts[stage][status] = stage_counts[stage].get(status, 0) + 1

    for stage in stage_names:
        counts = stage_counts.get(stage, {})
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        print(f"  {stage}: " + (", ".join(parts) if parts else "none"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full data-analysis pipeline end-to-end.")
    parser.add_argument("--mouse", default=MOUSE_NAME, help='Mouse name or comma-separated list, e.g. "Jamie11, Jamie12"')
    parser.add_argument("--date", default=SINGLE_DATE, help="Optional single date, e.g. 15-12-25")
    parser.add_argument("--block", default=SINGLE_BLOCK, help="Optional single block, e.g. R6")
    parser.add_argument(
        "--stages",
        default=PIPELINE_STAGES,
        help='Comma-separated stage names, e.g. "filter, pta_mean, summarize"; use None for all stages',
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs even if they already exist")
    args = parser.parse_args()

    mice = parse_mice_arg(args.mouse)
    if not mice:
        mice = discover_mice()
    if not mice:
        print("[SKIP] no mouse names provided")
        return

    stage_names = resolve_stage_names(args.stages)
    print("Stages:", ", ".join(stage_names))

    grand_total = 0
    processed_mice: list[str] = []
    for mouse in mice:
        print(f"\n=== Mouse: {mouse} ===")
        blocks = discover_blocks(mouse, args.date, args.block)
        if not blocks:
            print("[SKIP] no matching blocks found")
            continue

        processed_mice.append(mouse)

        all_results: dict[str, dict[str, str]] = {}
        for date, block in blocks:
            all_results[f"{date} | {block}"] = run_block(
                mouse,
                date,
                block,
                overwrite=bool(args.overwrite or OVERWRITE),
                stage_names=stage_names,
            )

        print_final_summary(all_results, stage_names)
        grand_total += len(all_results)

    if len(mice) > 1:
        print(f"\nProcessed {grand_total} blocks across {len(mice)} mice.")
    if processed_mice:
        print("Mice processed: " + ", ".join(processed_mice))
    else:
        print("Mice processed: none")


if __name__ == "__main__":
    main()
