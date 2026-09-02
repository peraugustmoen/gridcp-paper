"""Replication material for the gridcp software paper.

Each result in the paper is reproduced by one function, named
after the section of the paper it corresponds to:

    section_3_1_meancusum_runtime()      CUSUM runtime, 10k observations at p=1000
    section_3_2_well_log()               Well-log alarm times and penalized score
    section_4_5_grb_detection()          Poisson GLR and NPFOCuS on GRB 171004857
    section_5_1_far_simulation()         Empirical false alarm rate over stream length
    section_5_2_arl_simulation()         Run-length distributions against target ARL
    section_5_4_highd_mean_simulation()  High-dimensional Gaussian delay and runtime

Usage
-----
The ``__main__`` block at the bottom of this file calls each section function
explicitly, one per line. To run only some of them, comment out the calls you
do not want and run::

    python replication_materials.py

Alternatively, import and call a single section directly:

    from replication_materials import section_3_2_well_log
    section_3_2_well_log()

Each section is self-contained: it does its own imports, loads its own data and
prints its own header, so the calls can be run in any order or on their own.

Runtime
-------
Sections 3_1, 3_2 and 4_5 are fast. The simulation sections (5_1, 5_2, 5_4)
take roughly an hour in total, about 20 minutes each. Running this file as-is
runs all six.

Reproducibility
---------------
``N_JOBS = 18`` sets the number of parallel workers to 18. This is required to reproduce the exact numbers in
the paper and must not be changed.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
# Resolved from this file's location rather than the working directory, so the
# script can be run from anywhere.

REPO_ROOT = Path(__file__).resolve().parent
DATA_GRB = REPO_ROOT / "data_grb"
DATA_WELL_LOG = REPO_ROOT / "data_wellog"

# Figures are written here as PDFs, one per figure in the paper, named after
# the section function that produces them. Created on first save.
FIGURES = REPO_ROOT / "figures"

# Number of workers in the parallelized simulation code. Must be 18 to
# reproduce the exact results from the paper.
N_JOBS = 18

# Matplotlib style used by section 5.4. Applied via plt.style.context so it
# does not leak into other sections when several are run in one invocation.
PLOT_STYLE = "seaborn-v0_8-whitegrid"


def _banner(title):
    """Print a section header so output stays readable when running several."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _savefig(fig, name):
    """Write `fig` to ``figures/<name>.pdf`` and report where it went."""
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    print(f"Figure saved: {path.relative_to(REPO_ROOT)}")


# ===========================================================================
# Section 3.1 -- `CUSUM` runtime: processing 10k observations of dimension p=1000
# ===========================================================================
def section_3_1_meancusum_runtime():
    """A minimal benchmark of how long the `GridDetector` + `CUSUM` score takes
    to process 10,000 observations of dimension p=1000, one sample at a time.

    The first call to a Numba-compiled score function pays a one-time
    compilation cost, so we **warm up** the kernels before timing. We then time
    the full 10k-sample loop over several independent repetitions and report
    the mean.
    """
    _banner("Section 3.1 -- CUSUM runtime for 10k observations at p=1000")
    from gridcp import GridDetector
    from gridcp.scores import CUSUM

    N = 10_000  # number of observations per run
    N_REPEATS = 10  # number of timed repetitions to average over
    SEED = 0
    p = 1000

    # A high threshold means the detector never alarms, so every run processes
    # all N observations (we are timing throughput, not detection).
    score = CUSUM(n_features=p)
    detector = GridDetector(score=score, threshold=1e18)

    def process_stream(data):
        """Feed an array through the detector one sample at a time."""
        state = detector.init_state()
        for y in data:
            state, output = detector.update(state, y)
        return output

    # Warm-up: trigger Numba compilation (and disk cache) before timing.
    rng = np.random.default_rng(SEED)
    _ = process_stream(rng.standard_normal((N, p)))
    print("Warm-up done (kernels compiled).")

    # Timed runs: fresh data and a fresh state each repetition.
    times = []
    for r in range(N_REPEATS):
        data = rng.standard_normal((N, p))
        t0 = time.perf_counter()
        process_stream(data)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    print(f"Per-run times (s): {np.round(times, 3)}")
    print(f"Mean over {N_REPEATS} runs: {times.mean():.3f} s  (std {times.std():.3f} s)")
    print(f"Throughput: {N / times.mean():,.0f} observations/second")


# ===========================================================================
# Section 3.2 -- Well log changepoint detection example
# ===========================================================================
def section_3_2_well_log():
    """Online changepoint detection on the `well_log` dataset from the Turing
    Change Point Dataset (TCPD) (O Ruanaidh and Fitzgerald, 1996; Van den Burg
    and Williams, 2020). This shows how to import and define the score and
    detector, run it on the data, resetting upon finding a changepoint, and
    plotting the output.
    """
    _banner("Section 3.2 -- Well-log alarm times and penalized score")
    from gridcp import GridDetector
    from gridcp.scores import GaussianMean

    # -- Load data and annotations -----------------------------------------
    # The TCPD data is stored in the repository under ``data_wellog/``, so no
    # download step is needed. (The original notebook cloned
    # https://github.com/alan-turing-institute/TCPD at this point.)
    with open(DATA_WELL_LOG / "datasets" / "well_log" / "well_log.json") as f:
        d = json.load(f)
    data = np.array(d["series"][0]["raw"], dtype=float)

    with open(DATA_WELL_LOG / "annotations.json") as f:
        annotations = json.load(f)
    annotators = annotations["well_log"]

    # -- Code from the paper -----------------------------------------------
    score = GaussianMean()
    detector = GridDetector(score, threshold=2.8)

    state = detector.init_state()
    alarms = []
    scores = []
    for i, y in enumerate(data):
        state, output = detector.update(state, y)
        scores.append(output["max_score"])
        if output["alarm"]:
            alarms.append(i)
            state = detector.init_state()

    # -- Plot ---------------------------------------------------------------
    # Upper panel: Data with alarm times and annotated changepoints.
    # Bottom panel: The penalized GaussianMean score with threshold.
    colors = plt.cm.tab10.colors
    thr = float(detector.threshold[0])

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    # Top display: raw data with annotators marked by triangles, and alarm times
    ax = axes2[0]
    ax.plot(data, color="steelblue", linewidth=0.8)
    for j, a in enumerate(alarms):
        ax.axvline(
            a,
            color="tomato",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            label="Alarm times" if j == 0 else None,
        )
    cp_stack = {}
    for k, (ann_id, cps) in enumerate(annotators.items()):
        for cp in cps:
            level = cp_stack.get(cp, 0)
            ax.scatter(
                cp,
                0.03 + level * 0.025,
                transform=ax.get_xaxis_transform(),
                s=12,
                marker="^",
                color=colors[k],
                zorder=3,
            )
            cp_stack[cp] = level + 1

    ann_handle = plt.Line2D(
        [],
        [],
        marker="^",
        linestyle="none",
        markersize=5,
        color="grey",
        label="Annotated changepoints",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [ann_handle],
        labels=labels + ["Annotated changepoints"],
        loc="upper left",
        fontsize=9,
    )
    ax.set_ylabel("NMR measurement")

    # Bottom display: running max score with threshold
    ax = axes2[1]
    ax.plot(scores, color="steelblue", linewidth=0.8)
    ax.axhline(
        thr,
        color="lightgreen",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({thr:.2f})",
    )
    ax.set_ylabel("Penalized score")
    ax.set_xlabel("Sample index")
    ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(
        "Alarm times of the GaussianMean detector on the well log dataset",
        fontsize=12,
    )
    plt.tight_layout()
    _savefig(fig2, "3_2_well_log")


# ===========================================================================
# Section 4.5 -- Gamma-ray burst detection
# ===========================================================================
def section_4_5_grb_detection():
    """Poisson GLR and NPFOCuS detectors on GRB 171004857."""
    _banner("Section 4.5 -- Poisson GLR and NPFOCuS detections on GRB 171004857")
    from gridcp.calibration import calibrate_threshold_false_alarm
    from gridcp.detector import GridDetector
    from gridcp.scores import ExponentialFamilyGLR, NPFOCuS

    data_dir = str(DATA_GRB)
    grb_filename = "grb171004857.pickle"
    grb_file = os.path.join(data_dir, grb_filename)

    with open(grb_file, "rb") as fh:
        t, counts_dict, trigger_t, trigger_str = pickle.load(fh)

    t_mid = (t[:-1] + t[1:]) / 2
    channels = list(counts_dict.keys())
    grb_matrix = np.column_stack([counts_dict[c] for c in channels]).astype(np.float64)
    grb_total = grb_matrix.sum(axis=1)

    print(f"Trigger: {trigger_str}")
    print(f"Bins: {len(t_mid)}, Channels: {channels}")

    # -- Plot the total counts across detectors -----------------------------
    fig, axes = plt.subplots(5, 1, figsize=(13, 14), constrained_layout=True)

    all_fnames = sorted(f for f in os.listdir(data_dir) if f.endswith(".pickle"))
    for ax, fname in zip(axes, all_fnames):
        with open(os.path.join(data_dir, fname), "rb") as fh:
            t_f, counts_f, _, trigger_str_f = pickle.load(fh)
        t_mid_f = (t_f[:-1] + t_f[1:]) / 2
        total_f = np.column_stack(list(counts_f.values())).sum(axis=1)
        ax.step(t_mid_f, total_f, where="mid", linewidth=0.8)
        ax.set_title(f"{fname}  -  trigger: {trigger_str_f}", fontsize=9)
        ax.set_ylabel("Total counts")
        ax.ticklabel_format(useOffset=False, axis="x")

    axes[-1].set_xlabel("Time (MET s)")
    fig.suptitle("All pickle files - total counts", fontsize=12)
    # Not saved: this overview of all pickle files does not appear in the paper.

    # -- Defining the score models and calibration --------------------------
    # Thresholds are estimated by simulating null paths from a homogeneous
    # Poisson process. The rate is a robust (median) estimate computed from the
    # first four (background) files' total counts, rather than the sample mean,
    # so that a contaminated bin -- such as one of those files' own onboard-
    # trigger period -- can't pull the estimate up.
    target_fa = 0.05
    n_paths = 10000
    stream_len = len(grb_total)
    n_jobs = N_JOBS

    first_four_files = all_fnames[:4]
    training_data = []
    for fname in first_four_files:
        with open(os.path.join(data_dir, fname), "rb") as fh:
            _, counts_f, _, _ = pickle.load(fh)
        training_data.append(
            np.column_stack(list(counts_f.values())).sum(axis=1).astype(np.float64)
        )

    training_data = np.concatenate(training_data)
    robust_lambda = float(np.median(training_data))
    robust_scale = float(1.4826 * np.median(np.abs(training_data - robust_lambda)))

    poisson_theta_init = np.log(robust_lambda)

    glr_score = ExponentialFamilyGLR.from_family(
        "poisson",
        n_features=1,
        theta_init=poisson_theta_init,
        enable_penalty=False,
    )

    value_grid = np.linspace(
        robust_lambda - robust_scale * 2,
        robust_lambda + robust_scale * 2,
    )

    npfocus_score_obj = NPFOCuS(
        value_grid=value_grid,
        n_features=1,
    )

    print("poisson_theta_init", poisson_theta_init)
    print(f"Robust (median) Poisson rate estimate from first four series: {robust_lambda}")

    def poisson_pre_sampler(rng):
        return rng.poisson(lam=robust_lambda)

    glr_threshold = calibrate_threshold_false_alarm(
        glr_score,
        false_alarm_probability=target_fa,
        stream_len=stream_len,
        n_paths=n_paths,
        pre_sampler=poisson_pre_sampler,
        rng=0,
        n_jobs=n_jobs,
    )

    npfocus_threshold = calibrate_threshold_false_alarm(
        npfocus_score_obj,
        false_alarm_probability=target_fa,
        stream_len=stream_len,
        n_paths=n_paths,
        pre_sampler=poisson_pre_sampler,
        rng=0,
        n_jobs=n_jobs,
    )

    glr_det = GridDetector(
        score=glr_score,
        threshold=glr_threshold,
    )
    npf_det = GridDetector(
        score=npfocus_score_obj,
        threshold=npfocus_threshold,
    )

    print(f"Stream length: {stream_len}, false alarm prob: {target_fa}, n_paths: {n_paths}")
    print(f"GLR threshold:     {glr_threshold}")
    print(f"NPFOCuS threshold: {npfocus_threshold}")

    # -- Run detectors ------------------------------------------------------
    # Both detectors run online on the total counts using the same simple loop
    # used throughout: call `detector.update` on each observation and reset
    # state after every alarm.
    glr_state = glr_det.init_state()
    glr_trace = []
    glr_alarms = []
    glr_cps = []

    npf_state = npf_det.init_state()
    npf_trace = []
    npf_alarms = []
    npf_cps = []
    npf_triggered_stats = []

    for i, y in enumerate(grb_total):
        glr_state, glr_out = glr_det.update(glr_state, y)
        glr_trace.append(glr_out["max_score"].copy())
        if glr_out["alarm"]:
            glr_alarms.append(i)
            glr_cps.append(int(glr_out["max_split_point"][0]))
            glr_state = glr_det.init_state()

        npf_state, npf_out = npf_det.update(npf_state, y)
        npf_trace.append(npf_out["max_score"].copy())
        if npf_out["alarm"]:
            npf_alarms.append(i)
            stat_idx = int(np.flatnonzero(npf_out["max_score"] > npfocus_threshold)[0])
            npf_cps.append(int(npf_out["max_split_point"][stat_idx]))
            npf_triggered_stats.append("sum" if stat_idx == 0 else "max")
            npf_state = npf_det.init_state()

    glr_trace = np.stack(glr_trace)
    npf_trace = np.stack(npf_trace)

    print(f"GLR: detected {len(glr_alarms)} alarm(s). Alarm times: {glr_alarms}. Changepoint grid locations: {glr_cps}")
    print(f"NPFOCuS: detected {len(npf_alarms)} alarm(s). Alarm times: {npf_alarms}. Changepoint grid locations: {npf_cps} (triggered by: {npf_triggered_stats})")

    # -- Plot results -------------------------------------------------------
    threshold_color = "#d62728"
    score_color_glr = "#1f77b4"
    score_color_npf_sum = "#2ca02c"
    score_color_npf_max = "#ff7f0e"
    alarm_color = "#7f7f7f"
    count_color = "#2b6cb0"
    trigger_color = "#000000"

    def draw_alarm_lines(ax, alarm_indices):
        for idx, alarm_idx in enumerate(alarm_indices):
            ax.axvline(
                t_mid[alarm_idx],
                color=alarm_color,
                linewidth=1.8,
                label="Alarm" if idx == 0 else None,
                zorder=4,
            )

    def draw_trigger_line(ax):
        ax.axvline(
            trigger_t,
            color=trigger_color,
            linestyle="--",
            linewidth=1.4,
            label="Onboard trigger",
            zorder=5,
        )

    def draw_count_panel(ax, alarm_indices, title):
        ax.step(t_mid, grb_total, where="mid", color=count_color, linewidth=0.9)
        draw_alarm_lines(ax, alarm_indices)
        draw_trigger_line(ax)
        ax.set_ylabel("Total counts")
        ax.set_title(title, loc="left", fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex="col", constrained_layout=True)

    draw_count_panel(axes[0, 0], glr_alarms, "Poisson GLR on GRB total counts")
    draw_count_panel(axes[1, 0], npf_alarms, "NPFOCuS on GRB total counts")

    axes[0, 1].plot(t_mid, glr_trace[:, 0], color=score_color_glr, linewidth=1.0)
    axes[0, 1].axhline(glr_threshold[0], color=threshold_color, linewidth=1.4)
    draw_alarm_lines(axes[0, 1], glr_alarms)
    draw_trigger_line(axes[0, 1])
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Poisson GLR statistic and threshold", loc="left", fontsize=11)
    axes[0, 1].legend(fontsize=8)

    axes[1, 1].plot(t_mid, npf_trace[:, 0], color=score_color_npf_sum, linewidth=1.0, label="sum stat")
    axes[1, 1].axhline(npfocus_threshold[0], color=threshold_color, linewidth=1.4)
    axes[1, 1].plot(t_mid, npf_trace[:, 1], color=score_color_npf_max, linewidth=1.0, label="max stat")
    axes[1, 1].axhline(npfocus_threshold[1], color=threshold_color, linestyle="--", linewidth=1.4)
    draw_alarm_lines(axes[1, 1], npf_alarms)
    draw_trigger_line(axes[1, 1])
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("NPFOCuS statistics and thresholds", loc="left", fontsize=11)

    for ax in axes.flat:
        ax.ticklabel_format(useOffset=False, axis="x")
    for ax in axes[:, 0]:
        ax.set_xlabel("Time")
    for ax in axes[:, 1]:
        ax.set_xlabel("Time")

    fig.suptitle("Poisson GLR and NPFOCuS detectors on GRB 171004857", fontsize=13)
    _savefig(fig, "4_5_grb_detection")


# ===========================================================================
# Section 5.1 -- False alarm rate simulation
# ===========================================================================
def section_5_1_far_simulation():
    r"""Empirical false alarm rate of the `CUSUM` detector under the null
    (i.i.d. standard normal data) as a function of stream length. The threshold
    is calibrated to a target false alarm probability over a stream of length
    $T$, and we verify that the realized false alarm rate stays controlled as
    the stream grows way beyond this point.
    """
    _banner("Section 5.1 -- Empirical false alarm rate over stream length")
    from gridcp import GridDetector
    from gridcp.calibration import calibrate_threshold_false_alarm, mc_alarm_times
    from gridcp.scores import CUSUM

    # -- Calibrate threshold ------------------------------------------------
    score = CUSUM(n_features=1, enable_penalty=True)
    start_time = time.perf_counter()
    threshold = calibrate_threshold_false_alarm(
        score,
        false_alarm_probability=0.05,
        n_paths=20_000,
        stream_len=100,
        pre_sampler=lambda rng: rng.standard_normal(),
        rng=0,
    )
    print(f"Calibration done in {time.perf_counter() - start_time:.1f}s. Threshold: {threshold[0]:.4f}")

    detector = GridDetector(score=score, threshold=threshold)

    # -- Simulate alarm times under the null --------------------------------
    # Takes approx. 20 minutes on a MacBook pro
    t0 = time.perf_counter()
    alarm_times = mc_alarm_times(
        detector,
        n_paths=100_000,
        stream_len=10_000,
        pre_sampler=lambda rng: rng.standard_normal(),
        rng=1,
        n_jobs=N_JOBS,  # must set 18 jobs for reproducibility
    )
    print(f"Simulation finished in {time.perf_counter()-t0:.1f}s")

    # -- False alarm rate over the stream length ----------------------------
    Ts = np.geomspace(10, 10_000, 5_000).astype(int)
    at_sorted = np.sort(alarm_times)
    false_alarm_rates = np.searchsorted(at_sorted, Ts) / 100_000

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(
        Ts[:-1],
        false_alarm_rates[:-1],
        color="steelblue",
        linewidth=2,
        label="Empirical false alarm rate",
    )
    ax.axhline(
        0.05,
        color="lightgreen",
        label=rf"Target false alarm rate, $\delta$={0.05}",
        zorder=1,
    )
    ax.axvline(
        100,
        color="gray",
        linestyle="--",
        label=f"Calibration stream length, $T$={100}",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Stream length $T$")
    ax.set_ylabel("False alarm rate")
    ax.set_title("False alarm rate over $T$")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, "5_1_far_simulation")


# ===========================================================================
# Section 5.2 -- Average run length simulation
# ===========================================================================
def section_5_2_arl_simulation():
    r"""Distributions of alarm times for different target average run lengths
    ($\mathrm{ARL}_0 = 10, 100, 1000$) when using the `CUSUM` score. For each
    target the threshold is calibrated with `calibrate_threshold_arl`, null
    streams are simulated with the calibrated detector, and the distribution of
    individual run lengths is compared against the theoretical $\mathrm{Exp}$
    approximation.
    """
    _banner("Section 5.2 -- Run-length distributions against the target ARL")
    from gridcp import GridDetector
    from gridcp.calibration import (
        calibrate_threshold_arl,
        mc_alarm_times,
        with_calibrated_threshold,
    )
    from gridcp.scores import CUSUM

    # -- Simulation parameters ----------------------------------------------
    # Warning: This simulation takes quite a long time (20 minutes+ on a
    # MacBook pro with M5 pro chip) with N_SIM = 1_000_000, which is what we
    # use in the paper.
    TARGET_ARLS = [10, 100, 1000]  # target ARL_0 values
    N_CAL = 20_000  # streams used to calibrate each threshold
    N_SIM = 100_000  # streams used for obtaining alarm times
    L_MAX_FACTOR = 10  # stream length for obtaining alarm times = L_MAX_FACTOR * target ARL
    RNG_CAL, RNG_TEST = 0, 1

    # Sampler function for sampling from the null, i.e. a standard normal distribution
    def sampler(rng):
        return rng.standard_normal()

    # -- Calibrate and simulate for each target ARL -------------------------
    def run_arl_simulation(target_arl):
        """Calibrate, simulate null streams, and save run lengths for one target ARL."""
        score = CUSUM(enable_penalty=False)
        detector = GridDetector(score=score)

        # Calibrate threshold
        t0 = time.perf_counter()
        threshold = calibrate_threshold_arl(
            score,
            target_arl=target_arl,
            n_paths=N_CAL,
            pre_sampler=lambda rng: rng.standard_normal(),
            rng=RNG_CAL,
            n_jobs=N_JOBS,
        )
        detector_cal = with_calibrated_threshold(detector, threshold)

        # Simulate alarm times
        stream_len = L_MAX_FACTOR * target_arl
        alarm_times = mc_alarm_times(
            detector_cal,
            N_SIM,
            stream_len,
            pre_sampler=sampler,
            rng=RNG_TEST,
            n_jobs=N_JOBS,
        )
        print(f"Simulation for target ARL = {target_arl} finished in {time.perf_counter()-t0:.1f}s")

        return alarm_times

    start_time = time.perf_counter()
    sim_results = {}
    for target_arl in TARGET_ARLS:
        sim_results[target_arl] = run_arl_simulation(target_arl)
    print(f"Total simulation time: {time.perf_counter() - start_time:.1f}s")

    # -- Plot distribution of run lengths per target ARL --------------------
    n_bins = 50

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, target in zip(axes, TARGET_ARLS):
        run_lengths = sim_results[target] + 1
        empirical_arl = float(np.mean(run_lengths))

        _, edges, _ = ax.hist(
            run_lengths,
            bins=n_bins,
            density=True,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(
            target,
            color="lightgreen",
            linestyle="--",
            linewidth=1.5,
            zorder=3,
            label=rf"Target $\mathrm{{ARL}}_0 = {target}$",
        )
        ax.axvline(
            empirical_arl,
            color="k",
            linestyle="-",
            linewidth=1.5,
            zorder=2,
            label=f"Empirical ARL = {empirical_arl:.1f}",
        )

        x = np.linspace(0.0, edges[-1], 400)
        exp_pdf = np.exp(-x / target) / target
        ax.plot(x, exp_pdf, color="darkorange", linewidth=1.5, label=rf"Exp(${target}$)")

        ax.set_xlabel("Run length")
        ax.set_title(rf"Target $\mathrm{{ARL}}_0 = {target}$")
        ax.set_xlim(0, 5 * target)
        ax.legend(fontsize="small")

    axes[0].set_ylabel("Density")

    plt.tight_layout()
    _savefig(fig, "5_2_arl_simulation")


# ===========================================================================
# Section 5.4 -- High-dimensional Gaussian simulation study
# ===========================================================================
def section_5_4_highd_mean_simulation():
    r"""Demonstrates `CUSUM` (aggregation="max-sum") on $p = 1000$-dimensional
    Gaussian data. Both FaProb-calibrated ($\alpha = 5\%$, $N = 1000$) and
    ARL-calibrated ($\mathrm{ARL}_0 = 1000$) detectors are compared across
    three sparsity levels ($s = 1, 10, 1000$) as signal strength
    $\|\mu\|_2$ varies.

    For exact reproducibility please do not change the variable n_jobs.
    """
    _banner("Section 5.4 -- High-dimensional Gaussian study: detection delay and runtime")
    import gridcp as gc
    from gridcp.scores import CUSUM

    # The notebook called plt.style.use(...) here, which mutates matplotlib's
    # global rcParams and would restyle any section run after this one. The
    # style is instead applied with plt.style.context around each plot below,
    # so it stays scoped to this section's figures.

    # -- Configuration ------------------------------------------------------
    P = 1000  # dimension
    N = 1000  # stream length
    TAU = N // 2  # changepoint location (0-based, first post-change index)
    FAPROB = 0.05  # target false alarm probability
    ARL_TARGET = N  # target average run length

    SPARSITY = [1, 10, P]  # number of active coordinates
    PHI = np.linspace(0.0, 4.0, 17)  # signal strengths ||mu||_2

    N_PATHS_CALIBRATE = 1000
    N_PATHS_EVAL = 1000
    RNG_SEED = 1234

    # -- Samplers -----------------------------------------------------------
    def gaussian_sampler(rng, mean):
        return rng.normal(loc=mean, scale=1.0, size=len(mean))

    NULL_MEAN = np.zeros(P)

    # -- Calibration --------------------------------------------------------
    # FaProb-calibrated: time-dependent penalty (enable_penalty=True)
    score_faprob = CUSUM(n_features=P, aggregation="max-sum", enable_penalty=True)
    threshold_faprob = gc.calibrate_threshold_false_alarm(
        score_faprob,
        false_alarm_probability=FAPROB,
        n_paths=N_PATHS_CALIBRATE,
        stream_len=N,
        pre_sampler=gaussian_sampler,
        pre_kwargs={"mean": NULL_MEAN},
        rng=RNG_SEED,
        n_jobs=N_JOBS,
    )
    detector_faprob = gc.GridDetector(score=score_faprob, threshold=threshold_faprob)
    print(f"FaProb threshold: {threshold_faprob}")

    # ARL-calibrated: stationary score required (enable_penalty=False)
    score_arl = CUSUM(n_features=P, aggregation="max-sum", enable_penalty=False)
    threshold_arl = gc.calibrate_threshold_arl(
        score_arl,
        target_arl=ARL_TARGET,
        n_paths=N_PATHS_CALIBRATE,
        pre_sampler=gaussian_sampler,
        pre_kwargs={"mean": NULL_MEAN},
        rng=RNG_SEED,
        n_jobs=N_JOBS,
    )
    detector_arl = gc.GridDetector(score=score_arl, threshold=threshold_arl)
    print(f"ARL threshold: {threshold_arl}")

    # -- Helpers ------------------------------------------------------------
    def make_mu(phi, s):
        """Mean vector with ||mu||_2 = phi concentrated in the first s coordinates."""
        mu = np.zeros(P)
        mu[:s] = phi / np.sqrt(s)
        return mu

    def compute_mean_delay(alarm_times, tau, stream_len):
        """Mean delay. Paths alarming before tau are capped at stream_len - tau."""
        delays = alarm_times - tau
        return float(np.mean(delays[delays >= 0]))

    # -- Simulation loop ----------------------------------------------------
    detectors = {"FaProb": detector_faprob, "ARL": detector_arl}
    records = []

    for name, detector in detectors.items():
        for s in SPARSITY:
            for phi in PHI:
                mu = make_mu(phi, s)
                alarm_times = gc.mc_alarm_times(
                    detector,
                    n_paths=N_PATHS_EVAL,
                    stream_len=N,
                    pre_sampler=gaussian_sampler,
                    pre_kwargs={"mean": NULL_MEAN},
                    post_sampler=gaussian_sampler,
                    post_kwargs={"mean": mu},
                    changepoint=TAU,
                    rng=RNG_SEED + 1,  # different RNG seed for evaluation
                    n_jobs=N_JOBS,
                )
                records.append(
                    {
                        "phi": phi,
                        "sparsity": s,
                        "calibration": name,
                        "mean_delay": compute_mean_delay(alarm_times, TAU, N),
                    }
                )

    results = pd.DataFrame(records)

    # -- Detection delay plot -----------------------------------------------
    with plt.style.context(PLOT_STYLE):
        COLORS = dict(zip(SPARSITY, ["tab:blue", "tab:orange", "tab:green"]))
        LABELS = {s: rf"$s = {s}$" for s in SPARSITY}
        TITLES = {
            "FaProb": rf"FaProb-calibrated ($\alpha = {int(FAPROB * 100)}\%$, $T = {N}$)",
            "ARL": rf"ARL-calibrated ($\mathrm{{ARL}}0 = {ARL_TARGET}$)",
        }

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, name in zip(axes, ["FaProb", "ARL"]):
            sub = results[results["calibration"] == name]
            for s in SPARSITY:
                row = sub[sub["sparsity"] == s].sort_values("phi")
                ax.plot(row["phi"], row["mean_delay"], color=COLORS[s], label=LABELS[s])
            ax.axhline(N - TAU, color="gray", linestyle="--", linewidth=0.8, label="Max delay")
            ax.set_xlabel(r"Signal strength $\|\mu\|_2$")
            ax.set_title(TITLES[name])
            ax.legend()
            ax.set_xlim(0, PHI[-1])
            ax.set_ylim(0, N - TAU + 25)

        axes[0].set_ylabel("Average detection delay")
        fig.tight_layout()
        _savefig(fig, "5_4_highd_mean_delay")

    # -- Runtime benchmark (post-JIT) ---------------------------------------
    # Warmup: trigger Numba JIT compilation before timing
    _warmup_state = detector_faprob.init_state()
    _warmup_rng = np.random.default_rng(0)
    for _ in range(100):
        _warmup_state, _ = detector_faprob.update(
            _warmup_state, _warmup_rng.standard_normal(P)
        )

    N_RT_REPS = 10
    N_RT_VALUES = [100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    _rt_rng = np.random.default_rng(0)
    runtime_records = []

    for n in N_RT_VALUES:
        data = _rt_rng.standard_normal((n, P))
        times = []
        for _ in range(N_RT_REPS):
            state = detector_faprob.init_state()
            t0 = time.perf_counter()
            for x in data:
                state, _ = detector_faprob.update(state, x)
            times.append(time.perf_counter() - t0)
        runtime_records.append({"N": n, "time_s": np.median(times)})

    runtime = pd.DataFrame(runtime_records)

    # -- Runtime plot -------------------------------------------------------
    with plt.style.context(PLOT_STYLE):
        n_arr = np.array(N_RT_VALUES, dtype=float)
        t_ref = runtime.loc[runtime["N"] == 1000, "time_s"].values[0]
        ref_line = t_ref * (n_arr * np.log(n_arr)) / (1000 * np.log(1000))

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(runtime["N"], runtime["time_s"], "o-", color="tab:blue", label="Measured")
        ax.plot(n_arr, ref_line, "--", color="gray", label=r"$O(T \log T)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Stream length $T$")
        ax.set_ylabel("Runtime (s)")
        ax.legend()
        fig.tight_layout()
        _savefig(fig, "5_4_highd_mean_runtime")

    runtime_10k = runtime.loc[runtime["N"] == 10_000, "time_s"].values[0]
    print(f"Time to process the first 10 000 samples: {runtime_10k:.3f} seconds")


# ===========================================================================
# Runner
# ===========================================================================
# Each section of the paper is one function call below. Comment out the ones
# you do not want to run, or run a single section directly from a Python shell:
#
#     from replication_materials import section_3_2_well_log
#     section_3_2_well_log()
#
# Approximate runtimes on a modern laptop:
#     3_1   ~1 minute
#     3_2   seconds
#     4_5   ~2 minutes
#     5_1   ~20 minutes
#     5_2   ~20 minutes
#     5_4   ~20 minutes

if __name__ == "__main__":

    # -- Section 3.1: CUSUM runtime for 10k observations at p=1000 ----------
    a = time.perf_counter()
    section_3_1_meancusum_runtime()
    print(f"Section 3.1 runtime: {time.perf_counter() - a:.1f} seconds")

    # -- Section 3.2: Well-log alarm times and penalized score --------------
    a = time.perf_counter()
    section_3_2_well_log()
    print(f"Section 3.2 runtime: {time.perf_counter() - a:.1f} seconds")

    # -- Section 4.5: Poisson GLR and NPFOCuS on GRB 171004857 -------------
    a = time.perf_counter()
    section_4_5_grb_detection()
    print(f"Section 4.5 runtime: {time.perf_counter() - a:.1f} seconds")

    # -- Section 5.1: Empirical false alarm rate over stream length --------
    a = time.perf_counter()
    section_5_1_far_simulation()
    print(f"Section 5.1 runtime: {time.perf_counter() - a:.1f} seconds")

    # -- Section 5.2: Run-length distributions against the target ARL ------
    a = time.perf_counter()
    section_5_2_arl_simulation()
    print(f"Section 5.2 runtime: {time.perf_counter() - a:.1f} seconds")

    # -- Section 5.4: High-dimensional Gaussian delay and runtime ----------
    a = time.perf_counter()
    section_5_4_highd_mean_simulation()
    print(f"Section 5.4 runtime: {time.perf_counter() - a:.1f} seconds")
