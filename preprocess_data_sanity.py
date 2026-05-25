from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import DATA_ANALYSIS_ROOT


MOUSE_NAME = "Jamie6"
IMAGING_ROOT = DATA_ANALYSIS_ROOT / MOUSE_NAME / "Imaging_Data"
BLOCK_RE = re.compile(r"^R\d+$")
K = 1  # Matches preprocess_data.py

# -------------------------
# SIMPLE TOGGLES
# -------------------------
SHOW_TABLE = True
SHOW_PLOTS = False
CSV_OUT = None  # e.g. Path("trim_summary.csv")
TRIAL_FILTERS = None  # e.g. ["R1_1", "29.10.25"]


def trial_key(name: str) -> int:
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


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


def normalize_date_token(token: str) -> str:
    # Accept 29-10-25, 29.10.25, 29/10/25 (and optional trailing dot)
    t = token.strip().rstrip(".")
    t = t.replace(".", "-").replace("/", "-")
    return t


def parse_trial_date_tokens(tokens: list[str] | None) -> tuple[set[str] | None, set[str] | None]:
    if not tokens:
        return None, None

    trial_pat = re.compile(r"^R\d+_\d+$", re.IGNORECASE)
    date_pat = re.compile(r"^\d{2}[-./]\d{2}[-./]\d{2}\.?$")

    trials: set[str] = set()
    dates: set[str] = set()
    for tok in tokens:
        if date_pat.match(tok):
            dates.add(normalize_date_token(tok))
        elif trial_pat.match(tok):
            trials.add(tok)

    return (trials or None), (dates or None)


def collect_trimmed_trials(imaging_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not imaging_root.exists():
        raise FileNotFoundError(f"Imaging root not found: {imaging_root}")

    for date_dir in sorted(imaging_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for block_dir in sorted(date_dir.iterdir()):
            if not block_dir.is_dir() or not BLOCK_RE.match(block_dir.name):
                continue
            pkl_path = block_dir / f"{block_dir.name}_traces.pkl"
            if not pkl_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            trials = data.get("trials", {})
            for trial_name in sorted(trials.keys(), key=trial_key):
                trace_raw = trials[trial_name].get("trace_raw")
                if trace_raw is None:
                    continue

                F = np.asarray(trace_raw, float)
                n_full = int(len(F))
                if n_full == 0:
                    continue

                _, s, e = trim_by_level(F, K)
                trimmed_left = int(s)
                trimmed_right = int(max(0, n_full - e))
                if trimmed_left == 0 and trimmed_right == 0:
                    continue

                rows.append(
                    {
                        "pkl_path": str(pkl_path),
                        "date": date_dir.name,
                        "block": block_dir.name,
                        "trial": trial_name,
                        "n_full": n_full,
                        "trimmed_left": trimmed_left,
                        "trimmed_right": trimmed_right,
                        "kept_start": int(s),
                        "kept_end": int(e),
                        "n_kept": int(max(0, e - s)),
                    }
                )
    return rows


def filter_rows(
    rows: list[dict], only_trials: set[str] | None = None, only_dates: set[str] | None = None
) -> list[dict]:
    out = rows
    if only_trials:
        out = [r for r in out if r["trial"] in only_trials]
    if only_dates:
        out = [r for r in out if normalize_date_token(r["date"]) in only_dates]
    return out


def plot_trimmed_trials(
    rows: list[dict], only_trials: set[str] | None = None, only_dates: set[str] | None = None
) -> None:
    if not rows:
        print("No trimmed trials to plot.")
        return

    rows = filter_rows(rows, only_trials=only_trials, only_dates=only_dates)
    if only_trials or only_dates:
        if not rows:
            print("No trimmed trials matched the requested filters.")
            return

    print(f"Plotting {len(rows)} trimmed trial(s)...")
    for row in rows:
        pkl_path = Path(row["pkl_path"])
        trial_name = row["trial"]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        trials = data.get("trials", {})
        if trial_name not in trials:
            continue

        F_full = np.asarray(trials[trial_name].get("trace_raw"), float)
        if F_full.size == 0:
            continue

        s = int(row["kept_start"])
        e = int(row["kept_end"])
        F_keep = F_full[s:e]
        x_full = np.arange(len(F_full))
        x_keep = np.arange(s, e)

        plt.figure(figsize=(12, 3.5))
        plt.plot(x_full, F_full, color="0.75", lw=1.0, label="raw full trace")
        plt.plot(x_keep, F_keep, color="tab:blue", lw=1.2, label="kept after trim")

        if s > 0:
            plt.axvspan(0, s - 1, color="tab:red", alpha=0.12, label="trimmed start")
        if e < len(F_full):
            plt.axvspan(e, len(F_full) - 1, color="tab:orange", alpha=0.12, label="trimmed end")
        plt.axvline(s, color="tab:red", ls="--", lw=1.0)
        plt.axvline(e, color="tab:orange", ls="--", lw=1.0)

        plt.title(
            f"{row['date']} | {row['block']} | {trial_name} | "
            f"trimmed_left={row['trimmed_left']} trimmed_right={row['trimmed_right']}"
        )
        plt.xlabel("frame")
        plt.ylabel("F (a.u.)")
        plt.legend(loc="lower left", fontsize=8, framealpha=0.85)
        plt.tight_layout()

    plt.show(block=True)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No trimmed trials found.")
        return

    headers = [
        "date",
        "block",
        "trial",
        "n_full",
        "trimmed_left",
        "trimmed_right",
        "kept_start",
        "kept_end",
        "n_kept",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


def maybe_write_csv(rows: list[dict], csv_path: Path | None) -> None:
    if csv_path is None:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "date",
        "block",
        "trial",
        "n_full",
        "trimmed_left",
        "trimmed_right",
        "kept_start",
        "kept_end",
        "n_kept",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan all Jamie5 imaging trials and list only trimmed trials."
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Print the trimmed-trials table (off by default).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the trimmed-trials table.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting (table/CSV only).",
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        default=None,
        help="Optional filters for plotting: trial IDs (R1_1) and/or date token (29.10.25).",
    )
    args = parser.parse_args()

    rows = collect_trimmed_trials(IMAGING_ROOT)

    trial_tokens = TRIAL_FILTERS if TRIAL_FILTERS is not None else args.trials
    only_trials, only_dates = parse_trial_date_tokens(trial_tokens)
    filtered_rows = filter_rows(rows, only_trials=only_trials, only_dates=only_dates)

    show_table = SHOW_TABLE if SHOW_TABLE is not None else args.show_table
    show_plots = SHOW_PLOTS if SHOW_PLOTS is not None else (not args.no_plot)
    csv_out = CSV_OUT if CSV_OUT is not None else args.csv

    if show_table:
        print_table(filtered_rows)
    elif not filtered_rows:
        print("No trimmed trials found for the requested filters.")
    maybe_write_csv(filtered_rows, csv_out)
    if show_plots:
        plot_trimmed_trials(filtered_rows)


if __name__ == "__main__":
    main()


