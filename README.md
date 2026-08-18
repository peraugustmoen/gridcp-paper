# gridcp-paper

Reproduction material for the software paper describing
[**gridcp**](https://github.com/peraugustmoen/gridcp), a Python package for
online changepoint detection.

Each notebook in `code_for_paper/` reproduces one figure or timing
result from the paper. The filename prefix gives the corresponding section in the paper.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install gridcp==0.1.0 jupyter matplotlib numpy pandas
```

The `gridcp` version 0.1.0 is the first version published on PyPI, fixed to ensure reproducibility.

## Reproducing the results

The table below shows what each notebook produces, and where in the paper the corresponding result is located.

| Notebook | Section | Produces |
|---|---|---|
| `3_1_meancusum_runtime.ipynb` | 3.1 | CUSUM runtime for 10k observations at p=1000 |
| `3_2_well_log.ipynb` | 3.2 | `figures/well_log_stacked.pdf` |
| `4_5_GRB_detection.ipynb` | 4.5 | `figures/GRB_171004857_detection.pdf` |
| `5_1_FAR_simulation.ipynb` | 5.1 | `figures/fa_rate.pdf` |
| `5_2_ARL_simulation.ipynb` | 5.2 | `figures/arl_dist_sim.pdf` |
| `5_4_highd_mean_simulation.ipynb` | 5.4 | High-dimensional Gaussian study: `figures/highd_mean_delay.pdf`, `figures/highd_runtime.pdf` |

## Exact reproducibility requires `N_JOBS = 18`

The simulation notebooks parallelize with `N_JOBS = 18`, which is necessary to obtain the same results as the paper.

Expect the simulation notebooks (`5_1`, `5_2`, `5_4`) to take a long time - approximately 1 hour in total when ran on a MacBook Pro with an M5 Pro CPU. The other notebooks are quite fast.

## Data

`data-NPFOCuS/` contains the gamma-ray burst and simulation inputs:

- `grb171004857.pickle` — the GRB analyzed in section 4.5
- `528653157d4320141.pickle`, `528664172d33464026.pickle`,
  `528820438d371897.pickle`, `528910872d3934581.pickle` — additional GRB records
  used as background for the threshold calibration in section 4.5
- `081101167_bw_*.csv`, `081122614_bw_*.csv` — binned light curves
- `simulations.csv` — simulation inputs

The well-log data is downloaded in `3_2_well_log.ipynb` from the [Turing Change Point Dataset](https://github.com/alan-turing-institute/TCPD)
(Ó Ruanaidh and Fitzgerald, 1996; Van den Burg and Williams, 2020), and consequently that notebook needs network access on the first run.

## Outputs

The plots produced by the notebooks go into `figures/`. These are the exact
files used in the paper, kept to avoid having to do a full rerun.

## License

MIT — see [LICENSE](LICENSE).
