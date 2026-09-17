# FYS-3003 Space Physics

Scripts written for the UiT course FYS-3003 (Space Physics), one folder per
assignment plus a small analysis suite for the HEIMDALL riometer/radar
instruments. Each folder holds the final version of that task's script(s) —
earlier drafts and dead ends were left out of this repo.

## Layout

| Folder | Description |
|---|---|
| [`Task_1/`](Task_1) | Atmospheric pressure variation with altitude and scale height, using constant vs. variable gravity. |
| [`Task_2/`](Task_2) | Optical depth and solar flux attenuation through the atmosphere at several solar zenith angles. |
| [`Task_3/`](Task_3) | Photoionization: production rates and ionization profiles from solar irradiance. |
| [`Task_4/`](Task_4) | Ion chemistry model of the ionosphere. |
| [`Task_5/`](Task_5) | Magnetospheric current systems (noon-midnight B-field model). |
| [`HEIMDALL/`](HEIMDALL) | Analysis of HEIMDALL VHF/UHF radar and riometer data (electron density, riometer absorption, correlation with ACE solar wind / magnetometer data). |

## Data requirements

Small reference/data files (`MSIS.dat`, `IRI.dat`, `phot_abs.dat`,
`phot_ion.dat`, riometer `.txt` series, ACE magnetometer `.txt` files) are
included directly in each task folder.

The following inputs were **left out** because they're too large for a git
repo without LFS (irradiance tables are ~9-11 MB each and largely duplicated
between tasks; `.mat` radar files are 6-13 MB). Scripts that need them expect
the file in their own working directory:

- `Task_2/` and `Task_3/` — `irradiance_X{0,15,30,45,60,75,85}.txt` (generated
  by `Task_2/optical_depth_and_flux.py`, then consumed by `Task_3/prog_task_3.py`)
- `Task_5/` — `noon-midnight-B-fields.mat`
- `HEIMDALL/VHF/` — `bella-20250303-20250304.mat`
- `HEIMDALL/UHF/` — `beata-20250303-20250304.mat`
- `HEIMDALL/Riometer/` — `bella-20250303-20250304.mat` (VHF) and
  `beata-20250303-20250304.mat` (UHF)

## Requirements

Scripts use `numpy`, `scipy`, `matplotlib`, and (for the Riometer full
analysis) `pandas` and `statsmodels`.
