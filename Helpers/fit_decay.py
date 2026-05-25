from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tifffile import TiffFile


TRIAL_DIR = Path(r"D:\Data_Analysis\Jamie11\Imaging_Data\01-05-26\R13\R13_1")
OUTPUT_DIR = Path(r"D:\Data_Analysis\tables\photobleach_fit")
CACHE_PATH = OUTPUT_DIR / "Jamie11_01-05-26_R13_1_trace_500hz.npz"

FPS_HZ = 500.0
BIN_SECONDS = 1.0
REFRESH_TRACE_CACHE = False


def tiff_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.endswith("_Default"):
        return (0, stem)
    suffix = stem.rsplit("_", 1)[-1]
    return (int(suffix) + 1, stem) if suffix.isdigit() else (10**9, stem)


def stack_trace(tiff_path: Path) -> np.ndarray:
    trace = []
    with TiffFile(tiff_path) as tif:
        for page in tif.pages:
            frame = page.asarray()
            if frame.ndim != 2:
                raise ValueError(f"Expected 2D TIFF page, got shape {frame.shape} in {tiff_path}")
            trace.append(float(np.mean(frame)))
    return np.asarray(trace, dtype=np.float32)


def load_or_extract_trace() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not REFRESH_TRACE_CACHE:
        data = np.load(CACHE_PATH)
        print(f"Loaded cached trace: {CACHE_PATH}")
        return data["time_s"], data["fluorescence"], data["boundaries"]

    tiff_paths = sorted(TRIAL_DIR.glob("*MMStack*.ome.tif"), key=tiff_sort_key)
    if not tiff_paths:
        raise FileNotFoundError(f"No MMStack OME-TIFF files found in {TRIAL_DIR}")

    traces = []
    boundaries = [0]
    for path in tiff_paths:
        print(f"Reading {path.name}")
        trace = stack_trace(path)
        print(f"  frames={len(trace)} first={trace[0]:.2f} last={trace[-1]:.2f}")
        traces.append(trace)
        boundaries.append(boundaries[-1] + len(trace))

    fluorescence = np.concatenate(traces)
    time_s = np.arange(len(fluorescence), dtype=float) / FPS_HZ
    boundaries = np.asarray(boundaries, dtype=int)
    np.savez_compressed(CACHE_PATH, time_s=time_s, fluorescence=fluorescence, boundaries=boundaries)
    print(f"Cached trace: {CACHE_PATH}")
    return time_s, fluorescence, boundaries


def bin_trace(t: np.ndarray, y: np.ndarray, bin_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    bin_size = max(1, int(round(bin_seconds * FPS_HZ)))
    n_bins = len(y) // bin_size
    y_trim = y[: n_bins * bin_size]
    t_trim = t[: n_bins * bin_size]
    y_bin = y_trim.reshape(n_bins, bin_size).mean(axis=1)
    t_bin = t_trim.reshape(n_bins, bin_size).mean(axis=1)
    return t_bin, y_bin


def lower_envelope(t: np.ndarray, y: np.ndarray, window_s: float = 10.0, percentile: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    bins = np.floor((t - t[0]) / window_s).astype(int)
    out_t = []
    out_y = []
    for b in np.unique(bins):
        keep = bins == b
        if np.sum(keep) < 3:
            continue
        out_t.append(float(np.median(t[keep])))
        out_y.append(float(np.percentile(y[keep], percentile)))
    return np.asarray(out_t), np.asarray(out_y)


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = np.median(y[lo:hi])
    return out


def remove_high_transients(t: np.ndarray, y: np.ndarray, window_s: float = 61.0, mad_z: float = 2.5) -> np.ndarray:
    window = max(5, int(round(window_s / BIN_SECONDS)))
    if window % 2 == 0:
        window += 1
    baseline = rolling_median(y, window)
    resid = y - baseline
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    robust_sd = max(1e-9, 1.4826 * mad)
    return resid <= med + mad_z * robust_sd


def single_exp(t: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    return a * np.exp(-t / tau) + c


def double_exp(t: np.ndarray, a_fast: float, tau_fast: float, a_slow: float, tau_slow: float, c: float) -> np.ndarray:
    return a_fast * np.exp(-t / tau_fast) + a_slow * np.exp(-t / tau_slow) + c


def fit_single(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y0 = float(np.median(y[: max(3, min(10, len(y)))]))
    y_end = float(np.median(y[-max(5, len(y) // 20):]))
    drop = max(1e-6, y0 - y_end)
    duration = float(t[-1] - t[0])
    p0 = (drop, max(10.0, 0.5 * duration), max(0.0, y_end - 0.1 * drop))
    lower = (0.0, 5.0, 0.0)
    upper = (np.inf, 10.0 * duration, max(y0, y_end))
    popt, _ = curve_fit(single_exp, t, y, p0=p0, bounds=(lower, upper), maxfev=50000)
    return popt, single_exp(t, *popt)


def fit_double(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y0 = float(np.median(y[: max(3, min(10, len(y)))]))
    y_end = float(np.median(y[-max(5, len(y) // 20):]))
    drop = max(1e-6, y0 - y_end)
    duration = float(t[-1] - t[0])
    p0 = (
        0.25 * drop,
        max(10.0, 0.05 * duration),
        0.75 * drop,
        max(60.0, 1.5 * duration),
        max(0.0, y_end - 0.1 * drop),
    )
    lower = (0.0, 5.0, 0.0, 30.0, 0.0)
    upper = (np.inf, duration, np.inf, 10.0 * duration, max(y0, y_end))
    popt, _ = curve_fit(double_exp, t, y, p0=p0, bounds=(lower, upper), maxfev=100000)
    if popt[1] > popt[3]:
        popt = np.asarray([popt[2], popt[3], popt[0], popt[1], popt[4]], dtype=float)
    return popt, double_exp(t, *popt)


def model_values(model: str, t: np.ndarray, popt: np.ndarray) -> np.ndarray:
    return single_exp(t, *popt) if model == "single" else double_exp(t, *popt)


def half_times(model: str, popt: np.ndarray, duration: float) -> tuple[float, float]:
    dense_end = max(duration, 10.0 * float(popt[-2] if model == "double" else popt[1]))
    dense_t = np.linspace(0.0, dense_end, 100000)
    dense_y = model_values(model, dense_t, popt)
    y0 = float(model_values(model, np.asarray([0.0]), popt)[0])
    c = float(popt[-1])
    y_obs_end = float(model_values(model, np.asarray([duration]), popt)[0])

    target_plateau = c + 0.5 * (y0 - c)
    target_observed = y_obs_end + 0.5 * (y0 - y_obs_end)

    below_plateau = np.where(dense_y <= target_plateau)[0]
    below_observed = np.where(dense_t <= duration)[0]
    observed_idx = below_observed[np.argmin(np.abs(dense_y[below_observed] - target_observed))]
    plateau_half = float(dense_t[below_plateau[0]]) if len(below_plateau) else np.nan
    observed_half = float(dense_t[observed_idx])
    return plateau_half, observed_half


def fit_variant(name: str, model: str, t: np.ndarray, y: np.ndarray, start_s: float, source: str) -> dict:
    keep = t >= start_s
    t0 = t[keep] - t[keep][0]
    y0 = y[keep]
    label = source

    if source == "transient_mask":
        mask = remove_high_transients(t0, y0)
        t_fit = t0[mask]
        y_fit = y0[mask]
        label = f"transient mask ({np.sum(mask)}/{len(mask)} kept)"
    elif source == "lower_envelope":
        t_fit, y_fit = lower_envelope(t0, y0, window_s=10.0, percentile=20.0)
        label = "10 s lower envelope p20"
    else:
        t_fit = t0
        y_fit = y0

    if model == "single":
        popt, fit_y = fit_single(t_fit, y_fit)
    else:
        popt, fit_y = fit_double(t_fit, y_fit)

    pred = model_values(model, t_fit, popt)
    resid = y_fit - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    k = len(popt)
    n = len(y_fit)
    aic = float(n * np.log(max(sse / n, 1e-12)) + 2 * k)
    plateau_half_s, observed_half_s = half_times(model, popt, float(t_fit[-1] - t_fit[0]))

    return {
        "name": name,
        "model": model,
        "source": label,
        "start_s": float(start_s),
        "t_fit": t_fit,
        "y_fit": y_fit,
        "popt": popt,
        "fit_y": fit_y,
        "rmse": rmse,
        "r2": r2,
        "aic": aic,
        "plateau_half_s": plateau_half_s,
        "observed_half_s": observed_half_s,
    }


def plot_raw(time_s: np.ndarray, fluorescence: np.ndarray, boundaries: np.ndarray, t_bin: np.ndarray, y_bin: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_s, fluorescence, lw=0.35)
    axes[0].set_title("Raw fluorescence, all physical TIFF frames concatenated")
    axes[0].set_ylabel("fluorescence (a.u.)")

    axes[1].plot(t_bin, y_bin, lw=0.8)
    axes[1].set_title(f"Binned view ({BIN_SECONDS:g} s bins)")
    axes[1].set_ylabel("fluorescence (a.u.)")
    axes[1].set_xlabel("time (s)")

    for ax in axes:
        for boundary in boundaries[1:-1]:
            ax.axvline(boundary / FPS_HZ, color="0.25", lw=0.8, alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_raw_concatenated_trace.png", dpi=180)
    plt.close(fig)


def plot_variants(t_bin: np.ndarray, y_bin: np.ndarray, results: list[dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
    axes = axes.ravel()
    for ax, res in zip(axes, results):
        t0 = t_bin[t_bin >= res["start_s"]] - t_bin[t_bin >= res["start_s"]][0]
        y0 = y_bin[t_bin >= res["start_s"]]
        ax.plot(t0, y0, color="0.75", lw=0.8, label="binned raw after start cut")
        ax.scatter(res["t_fit"], res["y_fit"], s=8, color="tab:blue", alpha=0.65, label=res["source"])
        ax.plot(res["t_fit"], res["fit_y"], color="tab:orange", lw=2.0, label="fit")
        ax.set_title(f"{res['name']}\nRMSE={res['rmse']:.2f}, R2={res['r2']:.3f}")
        ax.set_ylabel("fluorescence")
        ax.set_xlabel("time after fit start (s)")
    for ax in axes[len(results):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_DIR / "02_fit_variants.png", dpi=180)
    plt.close(fig)


def plot_overlay(t_bin: np.ndarray, y_bin: np.ndarray, results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t_bin, y_bin, color="0.7", lw=0.8, label="binned raw")
    for res in results:
        abs_t = res["t_fit"] + res["start_s"]
        ax.plot(abs_t, res["fit_y"], lw=2.0, label=res["name"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fluorescence (a.u.)")
    ax.set_title("Photobleach fit overlay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_fit_overlay.png", dpi=180)
    plt.close(fig)


def save_results(results: list[dict]) -> None:
    out = OUTPUT_DIR / "photobleach_fit_results.csv"
    rows = []
    for res in results:
        popt = res["popt"]
        if res["model"] == "single":
            row = {
                "name": res["name"],
                "model": "single",
                "source": res["source"],
                "start_s": res["start_s"],
                "A": popt[0],
                "tau_s": popt[1],
                "tau_min": popt[1] / 60.0,
                "C": popt[2],
                "RMSE": res["rmse"],
                "R2": res["r2"],
                "AIC": res["aic"],
                "plateau_half_s": res["plateau_half_s"],
                "plateau_half_min": res["plateau_half_s"] / 60.0,
                "observed_half_s": res["observed_half_s"],
                "observed_half_min": res["observed_half_s"] / 60.0,
            }
        else:
            row = {
                "name": res["name"],
                "model": "double",
                "source": res["source"],
                "start_s": res["start_s"],
                "A_fast": popt[0],
                "tau_fast_s": popt[1],
                "tau_fast_min": popt[1] / 60.0,
                "A_slow": popt[2],
                "tau_slow_s": popt[3],
                "tau_slow_min": popt[3] / 60.0,
                "C": popt[4],
                "RMSE": res["rmse"],
                "R2": res["r2"],
                "AIC": res["aic"],
                "plateau_half_s": res["plateau_half_s"],
                "plateau_half_min": res["plateau_half_s"] / 60.0,
                "observed_half_s": res["observed_half_s"],
                "observed_half_min": res["observed_half_s"] / 60.0,
            }
        rows.append(row)

    keys = sorted({key for row in rows for key in row.keys()})
    import csv

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: list[dict]) -> None:
    print("\nFit results:")
    for res in results:
        popt = res["popt"]
        print(f"\n{res['name']} | {res['model']} | {res['source']}")
        if res["model"] == "single":
            print(f"  A={popt[0]:.4g}, tau={popt[1]:.2f}s ({popt[1]/60:.2f}min), C={popt[2]:.4g}")
        else:
            print(f"  A_fast={popt[0]:.4g}, tau_fast={popt[1]:.2f}s ({popt[1]/60:.2f}min)")
            print(f"  A_slow={popt[2]:.4g}, tau_slow={popt[3]:.2f}s ({popt[3]/60:.2f}min), C={popt[4]:.4g}")
        print(f"  RMSE={res['rmse']:.3f}, R2={res['r2']:.4f}, AIC={res['aic']:.1f}")
        print(f"  half to plateau={res['plateau_half_s']:.2f}s ({res['plateau_half_s']/60:.2f}min)")
        print(f"  half of observed fitted drop={res['observed_half_s']:.2f}s ({res['observed_half_s']/60:.2f}min)")


def main() -> None:
    time_s, fluorescence, boundaries = load_or_extract_trace()
    t_bin, y_bin = bin_trace(time_s, fluorescence, BIN_SECONDS)
    plot_raw(time_s, fluorescence, boundaries, t_bin, y_bin)

    variants = [
        ("double_all", "double", 0.0, "raw"),
        ("double_skip10s", "double", 10.0, "raw"),
        ("double_skip60s", "double", 60.0, "raw"),
        ("double_skip10s_transient_mask", "double", 10.0, "transient_mask"),
        ("double_skip10s_lower_envelope", "double", 10.0, "lower_envelope"),
        ("single_skip10s_lower_envelope", "single", 10.0, "lower_envelope"),
    ]

    results = []
    for name, model, start_s, source in variants:
        print(f"Fitting {name}")
        results.append(fit_variant(name, model, t_bin, y_bin, start_s=start_s, source=source))

    plot_variants(t_bin, y_bin, results)
    plot_overlay(t_bin, y_bin, results)
    save_results(results)
    print_summary(results)

    print(f"\nframes: {len(fluorescence)}")
    print(f"duration: {time_s[-1] / 60:.2f} min")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
