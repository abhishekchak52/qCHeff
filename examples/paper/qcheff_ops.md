---
title: Qcheff Ops
marimo-version: 0.8.13
width: medium
---

```python {.marimo name="setup"}
import marimo as mo
import seaborn.objects as so
from cupyx.profiler import benchmark
from more_itertools import zip_broadcast
import matplotlib
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as spsparse
import cupyx.scipy
import seaborn as sns
import polars as pl
import more_itertools
import itertools
import sys

# Initialization code that runs before all other cells
```

```python {.marimo}
from qcheff.operators import qcheffOperator, OperatorMatrix, SparseOperator
from qcheff.iswt import NPAD, ExactIterativeSWT
from qcheff.operators import create, destroy, number
```

```python {.marimo}
def bench_givens_rots(nlist):
    bench_list = []

    def test_NPAD(testSWT: ExactIterativeSWT):
        # testSWT.givens_rotation_matrix(0, 1)
        testSWT.eliminate_couplings(
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                # (4, 5),
                # (5, 6),
                # (6, 7),
                # (7, 8),
                # (8, 9),
                # (9, 10),
            )
        )
        # testSWT.eliminate_coupling(1, 2)
        # return testSWT.H.op  # .conj()

    for test_n in mo.status.progress_bar(nlist):
        # Base matrix to be used for benchmarking
        base_mat = spsparse.csr_array(destroy(test_n) + create(test_n) + number(test_n))
        testSWT_scipy = NPAD(qcheffOperator(base_mat))
        testSWT_cupy = NPAD(qcheffOperator(cpsparse.csr_matrix(base_mat)))

        mat_size = testSWT_scipy.H.op.data.nbytes

        def benchmark_givens_rotations(bench_func: callable, bench_name: str):
            incl_params_dict = {"N": test_n, "mat_size": mat_size}
            bench_results = benchmark(
                test_NPAD, args=(bench_func,), n_repeat=5, n_warmup=1
            )

            return pl.DataFrame(
                more_itertools.zip_broadcast(
                    *list(incl_params_dict.values()),
                    bench_name,
                    bench_results.cpu_times,
                    bench_results.gpu_times.squeeze(),
                ),
                schema=[
                    *list(incl_params_dict.keys()),
                    "Method",
                    "cpu_time",
                    "gpu_time",
                ],
            )

        bench_list.append(
            benchmark_givens_rotations(testSWT_scipy, "Givens Rotation (CPU)")
        )
        bench_list.append(
            benchmark_givens_rotations(testSWT_cupy, "Givens Rotation (GPU)")
        )

    return pl.concat(bench_list)
```

```python {.marimo}
test_n_list = np.logspace(5, 7, 3, dtype=int)
test_n_list
```

```python {.marimo}
def calculate_speedup(bench_df):
    mean_times = bench_df.group_by("Method").agg(pl.mean("time")).rows_by_key("Method")
    cupy_time = mean_times["Givens Rotation (GPU)"][0]
    scipy_time = mean_times["Givens Rotation (CPU)"][0]

    speedup = (
        scipy_time / cupy_time if scipy_time >= cupy_time else -cupy_time / scipy_time
    )
    return speedup
```

```python {.marimo column="1"}
with sns.axes_style("ticks"), sns.plotting_context("notebook"):
    speedbar_fig, speedbar_ax = plt.subplots(1, 1, layout="constrained", figsize=(5, 7))

    speedbar = (
        so.Plot(bench_df, x="N", y="time", color="Method")
        .add(so.Bar(alpha=1, baseline=1e-6, edgewidth=0), so.Agg(), so.Dodge())
        .add(so.Range(color="k", linewidth=1), so.Est(), so.Dodge(), legend=False)
        .scale(
            x=so.Continuous(trans="log").tick(minor=0),
            y=so.Continuous(trans="log").label(like="{x:.1f}"),
            color=so.Nominal(["#0068b5", "#76b900"]),
        )
        .limit(y=(9e-2, 4))
        .label(x="Matrix Dimension", y="Time (s)")
        .on(speedbar_ax)
        .plot(pyplot=True)
    )
    speedbar_ax.grid(axis="y")
    speedbar_ax.legend(
        speedbar._figure.legends.pop(0).legend_handles,
        ["NPAD (CPU)", "NPAD (GPU)"],
        frameon=False,
        loc="upper left",
        title="Method",
    )
```

```python {.marimo}
mo.hstack([mo.as_html(npad_bench_plot), mo.as_html(speedbar_fig)])
```

```python {.marimo}
bench_df = bench_givens_rots(test_n_list).drop("cpu_time").rename({"gpu_time": "time"})
```

```python {.marimo}
with sns.plotting_context("paper"):
    npad_bench_plot = sns.catplot(
        bench_df,
        kind="bar",
        hue_order=[
            # "QuTiP",
            "Givens Rotation (CPU)",
            "Givens Rotation (GPU)",
        ],
        palette={
            # "QuTiP": "slategray",
            "Givens Rotation (CPU)": (0 / 255, 104 / 255, 181 / 255),
            "Givens Rotation (GPU)": (118 / 255, 185 / 255, 0 / 255),
        },
        x="N",
        y="time",
        hue="Method",
        errorbar="ci",
        err_kws={"linewidth": 1},
        capsize=0.5,
    )
    npad_bench_plot.ax.set_yscale("log")
    npad_bench_plot.set_axis_labels("Matrix dimension", "Time (s)")

    npad_bench_plot.set_xticklabels([rf"$10^{int(i)}$" for i in np.log10(test_n_list)])
    sns.move_legend(npad_bench_plot, loc="upper center")
```