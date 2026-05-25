from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from config import DATA_ANALYSIS_ROOT


MOUSE = "Vinnie1"
DATE = "15-05-26"
BLOCK = "R16"
FS = 3600.0

ONLY_TRIAL = None          # None = average all trials, e.g. "R16_3" = one trial
PLOT_ALL_TRIALS = False    # True = show one spectrogram per trial
PLOT_RAW_TRACE = False     # Only active when PLOT_ALL_TRIALS is True

NPERSEG = 4096
NOVERLAP = 3840
FMAX = 250.0
STIM_START_SEC = 5.0
STIM_END_SEC = 15.0
SHARED_COLOR_SCALE = True
COLOR_PERCENTILES = (5.0, 99.0)

# Cut trace before FFT/spectrogram.
# Example: CUT_END_SEC = 3.0 removes the last 3 seconds.
CUT_START_SEC = 0.0
CUT_END_SEC = 0.0


def load_traces():
    path = DATA_ANALYSIS_ROOT / MOUSE / "Imaging_Data" / DATE / BLOCK / f"{BLOCK}_traces.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    traces = {}
    for name, trial in sorted(data["trials"].items()):
        traces[name] = np.asarray(trial["trace_raw"], dtype=float)
    return path, traces


def compute_spec(x):
    x = np.asarray(x, dtype=float)
    start = int(round(CUT_START_SEC * FS))
    stop = len(x) - int(round(CUT_END_SEC * FS))
    x = x[max(0, start):max(0, stop)]
    if len(x) < 32:
        raise ValueError("Trace too short after cutting start/end.")

    x = x - np.nanmean(x)
    f, t, p = spectrogram(
        x,
        fs=FS,
        window="hann",
        nperseg=min(NPERSEG, len(x)),
        noverlap=min(NOVERLAP, len(x) - 1),
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    return f, t, p


def spec_db_for_plot(f, p):
    keep = f <= FMAX
    p_db = 10 * np.log10(np.maximum(p[keep], 1e-30))
    return keep, p_db


def color_limits(spec_items):
    vals = []
    for f, _, p in spec_items:
        _, p_db = spec_db_for_plot(f, p)
        vals.append(p_db[np.isfinite(p_db)])
    vals = np.concatenate([v for v in vals if len(v)])
    if len(vals) == 0:
        return None
    lo, hi = np.percentile(vals, COLOR_PERCENTILES)
    return float(lo), float(hi)


def plot_spec(ax, f, t, p, title, clim=None):
    keep, p_db = spec_db_for_plot(f, p)
    im = ax.imshow(
        p_db,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], f[keep][0], f[keep][-1]],
        cmap="magma",
    )
    if clim is not None:
        im.set_clim(*clim)
    ax.axvline(STIM_START_SEC, color="cyan", ls="--", lw=1.0)
    ax.axvline(STIM_END_SEC, color="cyan", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    plt.colorbar(im, ax=ax, label="PSD (dB)")


def plot_raw_trace(ax, x, title):
    x = np.asarray(x, dtype=float)
    start = int(round(CUT_START_SEC * FS))
    stop = len(x) - int(round(CUT_END_SEC * FS))
    x = x[max(0, start):max(0, stop)]
    t = np.arange(len(x), dtype=float) / FS

    ax.plot(t, x, color="black", lw=0.8)
    ax.axvline(STIM_START_SEC, color="red", ls="--", lw=1.0)
    ax.axvline(STIM_END_SEC, color="red", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("raw trace")


def main():
    path, traces = load_traces()
    print(f"Loaded: {path}")
    print(f"Trials: {', '.join(traces)}")

    if ONLY_TRIAL is not None:
        f, t, p = compute_spec(traces[ONLY_TRIAL])
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_spec(ax, f, t, p, ONLY_TRIAL)
        plt.show()
        return

    if PLOT_ALL_TRIALS:
        spec_items = [(name, *compute_spec(x)) for name, x in traces.items()]
        clim = color_limits([(f, t, p) for _, f, t, p in spec_items]) if SHARED_COLOR_SCALE else None

        n_cols = 2 if PLOT_RAW_TRACE else 1
        fig_width = 13 if PLOT_RAW_TRACE else 9
        fig, axes = plt.subplots(len(traces), n_cols, figsize=(fig_width, 3 * len(traces)), squeeze=False)
        for row, (name, f, t, p) in enumerate(spec_items):
            if PLOT_RAW_TRACE:
                x = traces[name]
                plot_raw_trace(axes[row, 0], x, f"{name} raw trace")
                ax_spec = axes[row, 1]
            else:
                ax_spec = axes[row, 0]
            plot_spec(ax_spec, f, t, p, name, clim=clim)
        plt.tight_layout()
        plt.show()
        return

    specs = []
    for x in traces.values():
        f, t, p = compute_spec(x)
        specs.append(p)

    p_mean = np.mean(np.stack(specs), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_spec(ax, f, t, p_mean, f"{BLOCK} mean spectrogram, n={len(traces)}")
    plt.show()


if __name__ == "__main__":
    main()
