# gridcp-paper

Reproduction material for the software paper describing
[**gridcp**](https://github.com/peraugustmoen/gridcp), a Python package for
online changepoint detection.

All results are reproduced by `replication_materials.py`. Information on how to set up a virtual environment, download requirements and run the script, follows.

## Setup

Requires Python 3.10 or newer. Create a virtual environment and install necessary dependencies from `requirements.txt` using the following code, depending on your operating system:

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

On Windows `cmd.exe`, use `.venv\Scripts\activate.bat` instead of the
PowerShell activation script.


## Reproducing the results

Run everything with:

```bash
python replication_materials.py
```

To reproduce only some results, comment out the calls you do not
want from the bottom of `replication_materials.py`. Individual sections can also be imported and called directly:

```python
from replication_materials import section_3_2_well_log
section_3_2_well_log()
```

The table below shows what each function produces, where in the paper the corresponding result is located, and approximate runtime of each function (on a Macbook Pro with M5 Pro chip).

| Function | Section | Produces | Runtime |
|---|---|---|---|
| `section_3_1_meancusum_runtime()` | 3.1 | CUSUM runtime for 10k observations at p=1000 | TBD |
| `section_3_2_well_log()` | 3.2 | Well-log alarm times and penalized score | TBD |
| `section_4_5_grb_detection()` | 4.5 | Poisson GLR and NPFOCuS detections on GRB 171004857 | TBD |
| `section_5_1_far_simulation()` | 5.1 | Empirical false alarm rate over stream length | TBD |
| `section_5_2_arl_simulation()` | 5.2 | Run-length distributions against the target ARL | TBD |
| `section_5_4_highd_mean_simulation()` | 5.4 | High-dimensional Gaussian study: detection delay and runtime | TBD |

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
