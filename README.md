# gridcp-paper

Reproduction material for the software paper describing
[**gridcp**](https://github.com/peraugustmoen/gridcp), a Python package for
online grid-based changepoint detection in data streams.

Each notebook in `code_for_paper/` reproduces one figure, table, or timing
result from the paper. The filename prefix gives the paper section.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install gridcp==0.1.0 jupyter matplotlib numpy pandas
```

The `gridcp` version is pinned deliberately: the paper's results correspond to
`0.1.0`, and later versions may change numerical output.

## Reproducing the results

Run the notebooks **from inside `code_for_paper/`** — they resolve the data
directory as `../data-NPFOCuS`:

```bash
cd code_for_paper
jupyter lab
```

| Notebook | Section | Produces |
|---|---|---|
| `3_1_meancusum_runtime.ipynb` | 3.1 | CUSUM runtime for 10k observations at p=1000 (inline timings) |
| `3_2_well_log.ipynb` | 3.2 | `well_log_stacked.pdf` |
| `4_5_GRB_detection.ipynb` | 4.5 | `GRB_171004857_detection.pdf` |
| `5_1_FAR_simulation.ipynb` | 5.1 | `fa_rate.pdf` |
| `5_2_ARL_simulation.ipynb` | 5.2 | `arl_dist_sim.pdf` |
| `5_4_highd_mean_simulation.ipynb` | 5.4 | High-dimensional Gaussian study (inline table) |
| `6_4_DD_simulation.ipynb` | 6.4 | `dd_sim.pdf` and `simulation_results/dd_sim_results.npz` |

`grb_bootstrap_detection.py` is a standalone bootstrap analysis for the
gamma-ray burst example. Unlike the notebooks, it resolves `data-NPFOCuS`
relative to the **repository root**, so run it from there:

```bash
python code_for_paper/grb_bootstrap_detection.py
```

## Exact reproducibility requires `N_JOBS = 18`

The simulation notebooks parallelize with `N_JOBS = 18`. This value is part of
the random-number stream: each worker is seeded from the parent generator's
`SeedSequence`, so **changing the worker count changes the results**. Keep it at
18 to match the paper. Changing it still produces statistically equivalent
output, just not identical numbers.

Expect the simulation notebooks (`5_1`, `5_2`, `5_4`, `6_4`) to take a long time.

## Data

`data-NPFOCuS/` contains the gamma-ray burst and simulation inputs:

- `grb171004857.pickle` — the GRB analyzed in section 4.5
- `528653157d4320141.pickle`, `528664172d33464026.pickle`,
  `528820438d371897.pickle`, `528910872d3934581.pickle` — additional GRB records
  used by the bootstrap script
- `081101167_bw_*.csv`, `081122614_bw_*.csv` — binned light curves
- `simulations.csv` — simulation inputs

The well-log data is **not** vendored here. `3_2_well_log.ipynb` downloads it
itself from the [Turing Change Point Dataset](https://github.com/alan-turing-institute/TCPD)
(Ó Ruanaidh and Fitzgerald, 1996; Van den Burg and Williams, 2020) via
`git clone --depth 1`, so that notebook needs network access on first run.

## Committed figures

The PDFs in `code_for_paper/` are the exact figures used in the paper, committed
so the repository documents the published output without requiring a full rerun.

## License

MIT — see [LICENSE](LICENSE).
