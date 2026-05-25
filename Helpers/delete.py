from pathlib import Path
import shutil


ROOT = Path(r"D:\Data_Analysis")
TARGET_ROOTS = [
    ROOT / "Jamie6" / "Imaging_Data",
    ROOT / "Jamie6" / "Open_Ephys",
    ROOT / "Jamie11" / "Imaging_Data",
    ROOT / "Jamie11" / "Open_Ephys",
    ROOT / "Jamie12" / "Imaging_Data",
    ROOT / "Jamie12" / "Open_Ephys"
]
DRY_RUN = False
DELETE_EMPTY_DIRS = False


def should_keep(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()

    if path.is_dir():
        return False

    if name.endswith(".pkl"):
        return True
    if name.endswith(".txt") and "rec" in stem:
        return True

    return False


def delete_path(path: Path):
    if DRY_RUN:
        print(f"[DRY] delete {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"[DEL] {path}")


def remove_empty_dirs(root: Path):
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir():
            continue
        if path == root:
            continue
        try:
            is_empty = not any(path.iterdir())
        except OSError:
            continue
        if is_empty:
            if DRY_RUN:
                print(f"[DRY] remove empty dir {path}")
            else:
                path.rmdir()
                print(f"[RMDIR] {path}")


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Root not found: {ROOT}")

    deleted = 0
    kept = 0

    for target_root in TARGET_ROOTS:
        if not target_root.exists():
            print(f"[SKIP] missing target root: {target_root}")
            continue

        print(f"[ROOT] {target_root}")
        for path in sorted(target_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path == target_root:
                continue
            if path.is_dir():
                continue

            if should_keep(path):
                kept += 1
                continue

            delete_path(path)
            deleted += 1

        if DELETE_EMPTY_DIRS:
            remove_empty_dirs(target_root)

    print()
    print(f"Kept: {kept}")
    print(f"Marked for deletion: {deleted}")
    print(f"Dry run: {DRY_RUN}")


if __name__ == "__main__":
    main()
