from __future__ import annotations

import re
from pathlib import Path

from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = "Vinnie1"
SINGLE_DATE = None         # e.g. "01-04-26"
RUN_BATCH = False           # True = process all dates for this mouse
DRY_RUN = False              # True = preview only, False = actually rename/move
ONLY_NEW_DATA = True        # True = leave existing R-block folders untouched and only compile still-raw inputs

# Which roots to process
COMPILE_IMAGING = True      # Imaging_Data: group flat trial folders into block folders
COMPILE_EPHYS = True        # Open_Ephys: rename recording folders chronologically to R1, R2, ...


TRIAL_PATTERN = re.compile(r"^(R\d+)_(\d+)$")
BLOCK_PATTERN = re.compile(r"^R\d+$")


def block_number(name: str) -> int | None:
    m = BLOCK_PATTERN.match(str(name).strip())
    if m is None:
        return None
    try:
        return int(str(name).strip()[1:])
    except ValueError:
        return None


def parse_mouse_names(raw: str) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "all"}:
        return []
    names = [part.strip() for part in text.split(",")]
    return [name for name in names if name]


def discover_mouse_names() -> list[str]:
    out = []
    for mouse_dir in sorted(DATA_ANALYSIS_ROOT.iterdir()):
        if not mouse_dir.is_dir():
            continue
        if not (mouse_dir / "Imaging_Data").exists() and not (mouse_dir / "Open_Ephys").exists():
            continue
        out.append(mouse_dir.name)
    return out


def resolve_mouse_names(raw: str | None) -> list[str]:
    names = parse_mouse_names(raw)
    return names if names else discover_mouse_names()


def move_trial_folder(src: Path, dest: Path, dry_run: bool) -> tuple[str, str | None]:
    if dest.exists():
        return "skipped_existing", f"destination exists: {dest.name}"

    if dry_run:
        print(f"    WOULD MOVE: {src.name} -> {dest.parent.name}\\{dest.name}")
        return "dry_run", None

    src.rename(dest)
    print(f"    MOVED: {src.name} -> {dest.parent.name}\\{dest.name}")
    return "moved", None


def rename_folder(src: Path, dest: Path, dry_run: bool) -> tuple[str, str | None]:
    if src == dest:
        return "already_named", None
    if dest.exists():
        return "skipped_existing", f"destination exists: {dest.name}"

    if dry_run:
        print(f"    WOULD RENAME: {src.name} -> {dest.name}")
        return "dry_run", None

    src.rename(dest)
    print(f"    RENAMED: {src.name} -> {dest.name}")
    return "renamed", None


def ephys_recording_files(block_dir: Path) -> list[Path]:
    recnode = block_dir / "Record Node 104"
    if not recnode.exists():
        return []

    files = []
    for p in recnode.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() == "settings.xml":
            continue
        files.append(p)
    return files


def ephys_sort_key(block_dir: Path):
    files = ephys_recording_files(block_dir)
    if not files:
        return None
    mtimes = [p.stat().st_mtime for p in files]
    return (min(mtimes), max(mtimes), block_dir.name)


def compile_imaging_date(date_dir: Path, mouse_name: str, dry_run: bool) -> dict:
    label = f"{mouse_name} | {date_dir.name} | Imaging_Data"
    print("  Imaging_Data")

    trial_dirs = []
    for item in sorted(date_dir.iterdir()):
        if item.is_dir() and TRIAL_PATTERN.match(item.name):
            trial_dirs.append(item)

    if not trial_dirs:
        print("    No new flat trial folders found." if ONLY_NEW_DATA else "    No flat trial folders found.")
        return {
            "status": "skipped_no_flat_trials",
            "label": label,
            "handled": 0,
            "skipped_existing": 0,
            "failed": 0,
        }

    handled = 0
    skipped_existing = 0
    failed = 0

    for trial_dir in trial_dirs:
        m = TRIAL_PATTERN.match(trial_dir.name)
        if m is None:
            continue

        block_name = m.group(1)
        block_dir = date_dir / block_name
        dest = block_dir / trial_dir.name

        if not block_dir.exists():
            if dry_run:
                print(f"    WOULD CREATE BLOCK FOLDER: {block_name}")
            else:
                block_dir.mkdir(exist_ok=True)
                print(f"    CREATED BLOCK FOLDER: {block_name}")

        try:
            status, reason = move_trial_folder(trial_dir, dest, dry_run)
        except Exception as e:
            failed += 1
            print(f"    FAILED: {trial_dir.name} -> {type(e).__name__}: {e}")
            continue

        if status in {"moved", "dry_run"}:
            handled += 1
        elif status == "skipped_existing":
            skipped_existing += 1
            print(f"    SKIPPED: {trial_dir.name} | {reason}")

    if failed:
        status = "partial_failed"
    elif handled == 0 and skipped_existing > 0:
        status = "skipped_existing"
    else:
        status = "done"

    return {
        "status": status,
        "label": label,
        "handled": handled,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


def compile_ephys_date(date_dir: Path, mouse_name: str, dry_run: bool) -> dict:
    label = f"{mouse_name} | {date_dir.name} | Open_Ephys"
    print("  Open_Ephys")

    candidates: list[tuple[Path, tuple]] = []
    for item in sorted(date_dir.iterdir()):
        if not item.is_dir():
            continue
        key = ephys_sort_key(item)
        if key is not None:
            candidates.append((item, key))

    if not candidates:
        print("    No usable ephys recording folders found.")
        return {
            "status": "skipped_no_rename_needed",
            "label": label,
            "handled": 0,
            "skipped_existing": 0,
            "failed": 0,
        }

    ordered_dirs = [item for item, _ in sorted(candidates, key=lambda x: x[1])]

    if ONLY_NEW_DATA:
        existing_block_dirs = [item for item in ordered_dirs if BLOCK_PATTERN.match(item.name)]
        raw_dirs = [item for item in ordered_dirs if not BLOCK_PATTERN.match(item.name)]
        existing_nums = [block_number(item.name) for item in existing_block_dirs]
        existing_nums = [n for n in existing_nums if n is not None]
        next_block_num = max(existing_nums, default=0) + 1

        if existing_block_dirs:
            print(f"    Existing compiled block folders left untouched: {len(existing_block_dirs)}")
        if not raw_dirs:
            print("    No new raw ephys folders found.")
            return {
                "status": "skipped_existing",
                "label": label,
                "handled": 0,
                "skipped_existing": len(existing_block_dirs),
                "failed": 0,
            }

        desired_names = {}
        for src in raw_dirs:
            desired_names[src] = f"R{next_block_num}"
            next_block_num += 1
        candidate_set = set(raw_dirs)
    else:
        desired_names = {src: f"R{i+1}" for i, src in enumerate(ordered_dirs)}
        candidate_set = set(ordered_dirs)

    # Check for collisions with unrelated folders first.
    conflicts = []
    for src, desired_name in desired_names.items():
        dest = date_dir / desired_name
        if dest.exists() and dest not in candidate_set:
            conflicts.append((src.name, desired_name, dest.name))

    if conflicts:
        for src_name, desired_name, dest_name in conflicts:
            print(f"    CONFLICT: {src_name} wants {desired_name}, but {dest_name} already exists and is not a recording folder")
        return {
            "status": "failed",
            "label": label,
            "handled": 0,
            "skipped_existing": 0,
            "failed": len(conflicts),
        }

    rename_sources = list(desired_names.keys())
    rename_plan = [(src, date_dir / desired_names[src]) for src in rename_sources if src.name != desired_names[src]]

    if not rename_plan:
        if ONLY_NEW_DATA:
            print("    No new raw ephys folders need renaming.")
        else:
            print("    Ephys folders already match chronological R-order.")
        return {
            "status": "skipped_existing",
            "label": label,
            "handled": 0,
            "skipped_existing": len(rename_sources) if ONLY_NEW_DATA else len(ordered_dirs),
            "failed": 0,
        }

    handled = 0
    skipped_existing = 0
    failed = 0

    if dry_run:
        for src, dest in rename_plan:
            print(f"    WOULD RENAME: {src.name} -> {dest.name}")
            handled += 1
    else:
        temp_map: list[tuple[Path, Path]] = []
        for i, (src, _) in enumerate(rename_plan, 1):
            tmp = date_dir / f"__TMP_COMPILE_EPHYS__{i}"
            try:
                src.rename(tmp)
                temp_map.append((tmp, src))
            except Exception as e:
                failed += 1
                print(f"    FAILED TEMP RENAME: {src.name} -> {type(e).__name__}: {e}")

        for tmp, original_src in temp_map:
            desired_name = desired_names[original_src]
            dest = date_dir / desired_name
            try:
                tmp.rename(dest)
                print(f"    RENAMED: {original_src.name} -> {dest.name}")
                handled += 1
            except Exception as e:
                failed += 1
                print(f"    FAILED FINAL RENAME: {original_src.name} -> {type(e).__name__}: {e}")

    if failed:
        status = "partial_failed"
    elif handled == 0 and skipped_existing > 0:
        status = "skipped_existing"
    else:
        status = "done"

    return {
        "status": status,
        "label": label,
        "handled": handled,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


def compile_date(mouse_name: str, date_name: str, dry_run: bool) -> list[dict]:
    print(f"\nDate: {date_name}")
    results: list[dict] = []

    if COMPILE_IMAGING:
        imaging_date_dir = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data" / date_name
        if imaging_date_dir.exists():
            results.append(compile_imaging_date(imaging_date_dir, mouse_name, dry_run))
        else:
            print("  Imaging_Data")
            print(f"    Date folder not found: {imaging_date_dir}")
            results.append({
                "status": "failed",
                "label": f"{mouse_name} | {date_name} | Imaging_Data",
                "handled": 0,
                "skipped_existing": 0,
                "failed": 1,
            })

    if COMPILE_EPHYS:
        ephys_date_dir = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys" / date_name
        if ephys_date_dir.exists():
            results.append(compile_ephys_date(ephys_date_dir, mouse_name, dry_run))
        else:
            print("  Open_Ephys")
            print(f"    Date folder not found: {ephys_date_dir}")
            results.append({
                "status": "failed",
                "label": f"{mouse_name} | {date_name} | Open_Ephys",
                "handled": 0,
                "skipped_existing": 0,
                "failed": 1,
            })

    return results


def run_single_date(mouse_name: str, date_name: str, dry_run: bool) -> list[dict]:
    print(f"Mouse: {mouse_name}")
    return compile_date(mouse_name, date_name, dry_run)


def run_all_dates(mouse_name: str, dry_run: bool) -> list[dict]:
    candidate_dates: set[str] = set()

    if COMPILE_IMAGING:
        imaging_root = DATA_ANALYSIS_ROOT / mouse_name / "Imaging_Data"
        if imaging_root.exists():
            candidate_dates.update(d.name for d in imaging_root.iterdir() if d.is_dir())

    if COMPILE_EPHYS:
        ephys_root = DATA_ANALYSIS_ROOT / mouse_name / "Open_Ephys"
        if ephys_root.exists():
            candidate_dates.update(d.name for d in ephys_root.iterdir() if d.is_dir())

    if not candidate_dates:
        print(f"No usable date folders found for {mouse_name}.")
        return [{
            "status": "failed",
            "label": f"{mouse_name}",
            "handled": 0,
            "skipped_existing": 0,
            "failed": 1,
        }]

    print(f"Mouse: {mouse_name}")
    results: list[dict] = []
    for date_name in sorted(candidate_dates):
        results.extend(compile_date(mouse_name, date_name, dry_run))
    return results


def print_summary(results: list[dict], dry_run: bool) -> None:
    if not results:
        return

    total_handled = sum(r.get("handled", 0) for r in results)
    total_skipped = sum(r.get("skipped_existing", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)

    print("\nCompile summary")
    print(f"  mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  selection: {'ONLY NEW DATA' if ONLY_NEW_DATA else 'PROCESS ALL'}")
    print(f"  roots checked: {len(results)}")
    print(f"  folders handled: {total_handled}")
    print(f"  skipped existing: {total_skipped}")
    print(f"  failures: {total_failed}")

    issues = [r for r in results if r["status"] in {"partial_failed", "failed"}]
    if issues:
        print("\nRoots needing attention:")
        for r in issues:
            print(
                f"  {r['label']} | status={r['status']} | "
                f"handled={r.get('handled', 0)} | skipped={r.get('skipped_existing', 0)} | failed={r.get('failed', 0)}"
            )


def main() -> None:
    mouse_names = resolve_mouse_names(MOUSE_NAME)
    results: list[dict] = []

    if not mouse_names:
        print("No mice found to compile.")
        return

    for i, mouse_name in enumerate(mouse_names):
        if i > 0:
            print("\n" + "=" * 60)

        if RUN_BATCH:
            results.extend(run_all_dates(mouse_name, DRY_RUN))
        elif SINGLE_DATE is not None:
            results.extend(run_single_date(mouse_name, SINGLE_DATE, DRY_RUN))
        else:
            results.extend(run_all_dates(mouse_name, DRY_RUN))

    print_summary(results, DRY_RUN)


if __name__ == "__main__":
    main()
