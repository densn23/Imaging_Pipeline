# DBS-GEVI Fiber Photometry Analysis Pipeline

This repository contains a working analysis pipeline for deep brain stimulation
(DBS) experiments combined with high-speed GEVI fiber photometry, Open Ephys
recordings, LFP, stimulation timing, and treadmill velocity.

The pipeline is designed around block-wise mouse recordings. It converts raw
imaging and Open Ephys exports into Python pickles, aligns imaging and ephys to
DBS onset, preprocesses GEVI traces, computes pulse-triggered and
frequency-domain metrics, and generates summary figures/statistics.

This is a lab analysis pipeline, not a packaged Python library. Most scripts are
configured through variables near the top of the file or through
`run_pipeline.py`.

## Main Features

- Converts Micro-Manager `.ome.tif` imaging stacks into trial-level GEVI traces.
- Converts Open Ephys recordings into block-level ephys pickles.
- Detects DBS pulses and aligns imaging/ephys to stimulation onset.
- Computes wheel velocity from rotary encoder phase signals.
- Applies GEVI photobleaching correction and dF/F normalization.
- Applies notch filtering to stimulation-related narrowband artifacts.
- Computes first-pulse and pulse-train pulse-triggered averages.
- Computes spectrograms, PLV, Hilbert amplitude, pulsograms, and entrainment
  metrics.
- Produces single-block, all-trial, condition-average, and paper-style figures.

## Repository Structure

```text
.
|-- config.py                         # root data path
|-- run_pipeline.py                   # main entry point for block-wise pipeline
|-- compile.py                        # folder organization helper
|-- create_pickles_data.py            # imaging TIFF -> trace pickle
|-- create_pickles_ephys.py           # Open Ephys export -> ephys pickle
|-- preprocess_ephys.py               # pulse detection, epoching, velocity
|-- preprocess_data.py                # imaging/ephys alignment, bleach correction, dF/F
|-- process_data_filter.py            # notch filtering and final GEVI signal
|-- process_data_PTA.py               # first-pulse triggered analysis
|-- process_data_PTA_mean.py          # pulse-train PTA, PLV, Hilbert, spectrograms
|-- process_data_pulsogram.py         # pulse-by-pulse heatmaps
|-- summarize.py                      # combines outputs into *_summary.pkl
|-- entrain_anal.py                   # block/condition statistics
|-- plot.py                           # single-block summary plotting
|-- plot_all.py                       # all-trial overview plotting
|-- table.py                          # condition averages and group summaries
|-- Helpers/                          # metadata and exploratory helper scripts
|-- Plot_Figs/                        # figure-specific plotting scripts
|-- figures/                          # example/output figures
```

## Expected Data Layout

Set the root folder in `config.py`:

```python
DATA_ANALYSIS_ROOT = Path(r"D:\Data_Analysis")
```

Most core processing scripts use this path. Some plotting scripts, especially
`table.py` and scripts in `Plot_Figs/`, define their own `DATA_ANALYSIS_ROOT`
near the top of the file. If you move the project or data folder, check those
paths before plotting.

The pipeline expects data organized approximately like this:

```text
DATA_ANALYSIS_ROOT/
|-- MouseName/
|   |-- Imaging_Data/
|   |   |-- DD-MM-YY/
|   |   |   |-- R1/
|   |   |   |   |-- R1_1/
|   |   |   |   |   |-- *MMStack*.ome.tif
|   |   |   |   |-- R1_2/
|   |   |   |   |-- ...
|   |-- Open_Ephys/
|   |   |-- DD-MM-YY/
|   |   |   |-- R1/
|   |   |   |   |-- Record Node 104/
|   |   |   |   |-- ...
|-- tables/
|   |-- Experimental logbook - Maxime.xlsx
|   |-- stim_table_all_jamie.csv
|-- figures/
```

Each recording block is named `R1`, `R2`, etc. Imaging trials are usually named
`R1_1`, `R1_2`, etc.

## Installation

Use a Python environment with the scientific stack installed. The pipeline was
developed with Python 3.11/3.12.

Minimum practical dependencies:

```bash
pip install numpy scipy matplotlib tifffile openpyxl open-ephys-python-tools
```

The Open Ephys dependency must expose:

```python
from open_ephys.analysis import Session
```

## Running The Pipeline

The easiest entry point is `run_pipeline.py`.

Run all default stages for one block:

```bash
python run_pipeline.py --mouse Vinnie1 --date 12-05-26 --block R1
```

Run selected stages:

```bash
python run_pipeline.py --mouse Vinnie1 --date 12-05-26 --block R1 --stages filter,pta_mean,pulsogram,summarize
```

Run all blocks for one mouse:

```bash
python run_pipeline.py --mouse Jamie11
```

Run several mice:

```bash
python run_pipeline.py --mouse "Jamie10, Jamie11, Vinnie1"
```

Overwrite existing outputs:

```bash
python run_pipeline.py --mouse Vinnie1 --date 12-05-26 --block R1 --overwrite
```

Available stage names:

```text
create_data
create_ephys
preprocess_ephys
preprocess_data
filter
pta
pta_mean
pulsogram
summarize
entrain_anal
```

Short aliases are also supported, for example `create`, `preprocess`,
`summary`, `entrainment`, and `stats`.

## Pipeline Stages

### 1. Raw data conversion

`create_pickles_data.py` reads imaging `.ome.tif` files and stores one mean
fluorescence trace per trial in:

```text
R#_traces.pkl
```

`create_pickles_ephys.py` reads Open Ephys recordings and stores relevant
channels in:

```text
R#_ephys.pkl
```

Expected ephys channels include LFP, stimulation, camera frame pulses, trial
signal, and rotary encoder phase channels.

### 2. Ephys preprocessing

`preprocess_ephys.py` detects DBS pulses, epochs ephys data by trial, extracts
camera frame timing, and computes velocity from the rotary encoder.

Output:

```text
R#_epoched_ephys.pkl
```

### 3. Imaging preprocessing

`preprocess_data.py` aligns GEVI traces to ephys timing and computes dF/F.

Photobleaching is corrected by fitting a monotonic decay model to the
pre-stimulation baseline. If the correction produces excessive positive
post-stimulation runaway, the correction is rejected and dF/F is computed from
the raw trace instead.

Output:

```text
R#_traces_processed.pkl
```

### 4. Filtering

`process_data_filter.py` applies notch filters and saves the downstream GEVI
signal.

Output:

```text
R#_traces_processed_notched.pkl
```

### 5. Pulse-triggered and frequency analyses

`process_data_PTA.py` computes first-pulse responses.

`process_data_PTA_mean.py` computes pulse-train/mPTA responses, PLV, Hilbert
amplitude at the DBS frequency and harmonics, and spectrograms.

`process_data_pulsogram.py` computes pulse-by-pulse response heatmaps.

Outputs:

```text
R#_traces_processed_notched_pta_first_pulse.pkl
R#_traces_processed_notched_pta_train.pkl
R#_traces_processed_notched_pulsogram.pkl
```

### 6. Summary

`summarize.py` combines the processed traces, PTA outputs, pulsograms,
spectrograms, and Hilbert metrics into a single block-level file:

```text
R#_summary.pkl
```

This is the main file used by the plotting scripts.

## Plotting And Statistics

Use these scripts after summary files have been generated.

- `plot.py`: one block summary figure.
- `plot_all.py`: overview of all individual trials within one block.
- `table.py`: condition-level averages across blocks using `stim_table_all_jamie.csv`.
- `entrain_anal.py`: block-level and condition-level statistics, including
  velocity, Vm, Hilbert amplitude, PLV, PTA latency/jitter, and optional theta
  band analyses.
- `Plot_Figs/figs.py`: figure-specific group comparisons.
- `Plot_Figs/figs_pw.py`: pulse-width/TEED-balanced comparisons.
- `Plot_Figs/figs_last.py`: special 10 Hz figure/analysis script.

Most plotting scripts are controlled by variables at the top of the file. For
example, `table.py` uses `MOUSE_NAME`, `FREQUENCY_HZ`, `AMPLITUDE_UA`,
`PHASE`, `IMAGING_SIDE`, and panel toggles such as `PLOT_FULL_TRACE`,
`PLOT_SPECTROGRAM`, and `PLOT_PLV_HISTOGRAMS`.

## Metadata Table

Condition-level plotting depends on a metadata table, usually stored as:

```text
tables/stim_table_all_jamie.csv
```

Depending on the script, this `tables/` folder may be expected either inside
`DATA_ANALYSIS_ROOT` or next to the plotting script. Check the path variables at
the top of the script before running.

The table is generated from the experimental logbook with:

```bash
python Helpers/build_stim_table_from_logbook.py
```

The metadata table stores mouse/date/block labels, stimulation parameters,
phase, imaging side, fiber placement, trial counts, and related annotations.

## Common Outputs

For a block named `R1`, the major outputs are:

```text
R1_traces.pkl
R1_ephys.pkl
R1_epoched_ephys.pkl
R1_traces_processed.pkl
R1_traces_processed_notched.pkl
R1_traces_processed_notched_pta_first_pulse.pkl
R1_traces_processed_notched_pta_train.pkl
R1_traces_processed_notched_pulsogram.pkl
R1_summary.pkl
R1_entrainment_analysis.pkl
```

## Notes And Caveats

- This repository is tuned to a specific lab data structure and Open Ephys
  channel mapping.
- The code is intended for reproducible internal analysis, but it is not yet a
  general-purpose package.
- Paths are Windows-style by default; update `config.py` before running.
- dF/F is stored as a fractional value in processed files and multiplied by 100
  for plotting/reporting in most figure scripts.
- Some scripts in `Plot_Figs/` are figure-specific and may contain hard-coded
  selections from the thesis analysis.

## Citation / Use

If you reuse or adapt this pipeline, please cite the associated project or
contact the repository author for the preferred citation once available.
