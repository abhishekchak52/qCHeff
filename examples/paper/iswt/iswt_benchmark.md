---
title: Qcheff Ops
marimo-version: 0.8.13
width: medium
---

```python {.marimo}
test_n_list = np.logspace(2, 7, 15, dtype=int)
# test_n_list = np.logspace(5, 7, 3, dtype=int)
test_n_list
```

```python {.marimo}
bench_res = benchmark_ISWT(test_n_list)
bench_df = (
    pd.DataFrame(bench_res)
    .explode(column=["cpu_time", "gpu_time"], ignore_index=True)
    .convert_dtypes()
)
bench_df
```

```python {.marimo}
def benchmark_ISWT(
    nlist,
    n_repeat=10,
    n_warmup=2,
    max_duration=10,
    diag_threshold=1e5,
):
    bench_list = []

    def test_NPAD(testSWT: ExactIterativeSWT):
        testSWT.eliminate_couplings(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))

    with mo.status.spinner(title="Benchmarking ISWT") as _spinner:
        for n in nlist:
            # Base matrices to be used for benchmarking
            base_mat = spsparse.csr_array(destroy(n) + create(n) + number(n))
            base_mat_gpu = cpsparse.csr_matrix(base_mat)
            # Convert to objects qCHeff understands
            testSWT_scipy = NPAD(qcheffOperator(base_mat))
            testSWT_cupy = NPAD(qcheffOperator(cpsparse.csr_matrix(base_mat)))
            mat_size = testSWT_scipy.H.op.data.nbytes

            benchmark_funcs = {
                "NPAD_CPU": lambda: test_NPAD(testSWT_scipy),
                "NPAD_GPU": lambda: test_NPAD(testSWT_cupy),
            }
            diag_benchmark_funcs = {
                "Diagonalization_CPU": lambda: spsparse.linalg.eigsh(base_mat),
                "Diagonalization_GPU": lambda: cpsla.eigsh(base_mat_gpu),
            }

            if n < diag_threshold:
                benchmark_funcs.update(diag_benchmark_funcs)

            for bench_name, bench_func in benchmark_funcs.items():
                method, device = bench_name.split("_")
                _spinner.update(
                    title=f"Benchmarking {method} on {device}",
                    subtitle=f"N = {n.item() / 1e6:g} million",
                )
                # Run the benchmark
                bench_results = benchmark(
                    bench_func,
                    name=bench_name,
                    n_repeat=n_repeat,
                    n_warmup=n_warmup,
                    max_duration=max_duration,
                )

                # Do not report benchmarks that did not finish.
                if bench_results.cpu_times.size == n_repeat:
                    results_dict = {
                        "N": n,
                        "mat_size": mat_size,
                        "Method": method,
                        "Device": device,
                        "cpu_time": bench_results.cpu_times,
                        "gpu_time": bench_results.gpu_times.squeeze(),
                    }
                    bench_list.append(results_dict)

    return bench_list
```

```python {.marimo}
bench_res
```

```python {.marimo hide_code="true" name="setup"}
import itertools
import sys

import more_itertools
from more_itertools import zip_broadcast

import marimo as mo

import matplotlib
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as spsparse
import scipy.sparse.linalg as spsla

import cupy as cp
import cupyx.scipy
from cupyx.profiler import benchmark
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsla

import polars as pl
import pandas as pd
# Initialization code that runs before all other cells
```

```python {.marimo}
from qcheff.operators import qcheffOperator, OperatorMatrix, SparseOperator
from qcheff.iswt import NPAD, ExactIterativeSWT
from qcheff.operators import create, destroy, number
```

```python {.marimo column="1"}
with sns.axes_style("ticks"), sns.plotting_context("notebook"):
    speedbar_fig, speedbar_ax = plt.subplots(
        1, 1, layout="constrained", figsize=(7, 5)
    )

    speedbar = (
        so.Plot(
            bench_df,
            x="N",
            y="gpu_time",
            linestyle="Method",
            marker="Method",
            color="Device",
        )
        .add(so.Dots(alpha=0.5, pointsize=7))
        .add(so.Line(alpha=1, marker=False), so.Agg())
        # .add(so.Range(color="k", linewidth=1), so.Est(), legend=False)
        .scale(
            x=so.Continuous(trans="log").tick(minor=0),
            y=so.Continuous(trans="log").label(like="{x:.3g}"),
            color=so.Nominal({"CPU": "#0068b5", "GPU": "#76b900"}),
            marker=so.Nominal(["s", "v"]),
        )
        .limit(x=more_itertools.minmax(test_n_list))
        .label(x="Matrix Dimension", y="Time (s)")
        .on(speedbar_ax)
        .plot(pyplot=True)
    )
    speedbar_ax.grid(axis="y")
    legend = speedbar._figure.legends[0]
    legend.set_frame_on(False)
    legend.set_bbox_to_anchor((0.98, 0.35))
speedbar_fig
```

```python {.marimo}
speedbar_fig.savefig("iswt_benchmark.pdf")
```