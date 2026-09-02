# gridcp-paper

Reproduction material for the software paper describing
[**gridcp**](https://github.com/peraugustmoen/gridcp), a Python package for
online changepoint detection.

All results are reproduced by `replication_materials.py`. Information on how to set up a virtual environment, download requirements and run the script, follows.

## Setup

Requires Python 3.10 or newer. Clone the repository and set it as the root directory:

```bash
git clone https://github.com/peraugustmoen/gridcp-paper.git
cd gridcp-paper
```

Create a virtual environment and install the dependencies from `requirements.txt`, depending on your operating system:

**macOS and Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the results

Run everything with:

```bash
python replication_materials.py
```

Output plots will be saved in a `figures` folder. On a MacBook Pro with an M5 Pro chip, this takes approximately 30 minutes. To reproduce only some results, comment out the calls you do not want from the bottom of `replication_materials.py`.

The table below shows what each function produces, where in the paper the corresponding result is located, and approximate runtime of each function (on a Macbook Pro with M5 Pro chip).

| Function | Section | Produces | Runtime |
|---|---|---|---|
| `section_3_1_meancusum_runtime()` | 3.1 | CUSUM runtime for 10k observations at p=1000 | ~5 seconds |
| `section_3_2_well_log()` | 3.2 | Well-log alarm times and penalized score | $<1$ second |
| `section_4_5_grb_detection()` | 4.5 | Poisson GLR and NPFOCuS detections on GRB 171004857 | ~20 seconds |
| `section_5_1_far_simulation()` | 5.1 | Empirical false alarm rate over stream length | ~20 minutes |
| `section_5_2_arl_simulation()` | 5.2 | Run-length distributions against the target ARL | ~3 minutes |
| `section_5_4_highd_mean_simulation()` | 5.4 | High-dimensional Gaussian study: detection delay and runtime | ~5 minutes |



## Exact reproducibility

The simulations are parallelized with `N_JOBS = 18`, which is necessary to obtain the same results as the paper.

## Data

`data_grb/` contains the gamma-ray burst and simulation inputs:

- `grb171004857.pickle`: the GRB analyzed in section 4.5
- `528653157d4320141.pickle`, `528664172d33464026.pickle`,
  `528820438d371897.pickle`, `528910872d3934581.pickle`: additional GRB records
  used as background for the threshold calibration in section 4.5
- `081101167_bw_*.csv`, `081122614_bw_*.csv`: light curves
- `simulations.csv`: simulation inputs

`data_wellog/` contains the well-log data from the [Turing Change Point Dataset](https://github.com/alan-turing-institute/TCPD)
(Ó Ruanaidh and Fitzgerald, 1996; Van den Burg and Williams, 2020).

## License

MIT, see [LICENSE](LICENSE).
