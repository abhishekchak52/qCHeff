---
title: Qcheff Ops
marimo-version: 0.8.13
width: medium
---

```{.python.marimo}
import marimo as mo
```

```{.python.marimo}
from qcheff.operators import (
    qcheffOperator,
    OperatorMatrix,
    SparseOperator,
)
from qcheff.operators import create, destroy, number
```

```{.python.marimo}
from qcheff.iswt import NPAD, ExactIterativeSWT
```

```{.python.marimo}
benchmark_cpu_checkbox = mo.ui.checkbox(value=False, label="CPU Benchmarks")
benchmark_gpu_checkbox = mo.ui.checkbox(value=False, label="GPU Benchmarks")
mo.vstack([benchmark_cpu_checkbox, benchmark_gpu_checkbox])
```

```{.python.marimo}
bench_df = bench_givens_rots(test_n_list).drop("cpu_time").rename({"gpu_time" : "time"})
```

```{.python.marimo}
cpu_bench = benchmark(
    test_NPAD,
    args=(testSWT_scipy,),
    n_repeat=10,
    n_warmup=1,
)
```

```{.python.marimo}
gpu_bench = benchmark(
    test_NPAD,
    args=(testSWT_cupy,),
    n_repeat=10,
    n_warmup=1,
)
```

```{.python.marimo}
def test_NPAD(testSWT: ExactIterativeSWT):
    # testSWT.givens_rotation_matrix(0, 1)
    testSWT.eliminate_couplings(
        (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            # (8, 9),
            # (9, 10),
        )
    )
    # testSWT.eliminate_coupling(1, 2)
    # return testSWT.H.op  # .conj()
    # pass
```

```{.python.marimo}
test_n_list = np.logspace(5, 7, 3, dtype=int)
test_n_list
```

```{.python.marimo}

```

```{.python.marimo}


def bench_givens_rots(nlist):
    bench_list = []
    for test_n in mo.status.progress_bar(nlist):
        base_mat = spsparse.csr_array(
            destroy(test_n) + create(test_n) + number(test_n)
        )
        testSWT_scipy = NPAD(qcheffOperator(base_mat))
        bench_list.append(
            create_bench_df(
                benchmark(
                    test_NPAD,
                    args=(testSWT_scipy,),
                    n_repeat=10,
                    n_warmup=1,
                ),
                method_name="Givens Rotation (CPU)",
                incl_params_dict={"N": test_n, "mat_size":1e-9*(testSWT_scipy.H.op.data.nbytes)},
            )
        )
        testSWT_cupy = NPAD(qcheffOperator(cpsparse.csr_matrix(base_mat)))
        bench_list.append(
            create_bench_df(
                benchmark(
                    test_NPAD,
                    args=(testSWT_cupy,),
                    n_repeat=10,
                    n_warmup=1,
                ),
                method_name="Givens Rotation (GPU)",
                incl_params_dict={"N": test_n, "mat_size":1e-9*(testSWT_cupy.H.op.data.nbytes)},
            )
        )
    return pl.concat(bench_list)
```

```{.python.marimo}
n_op = 1000000
base_mat = destroy(n_op) + create(n_op) + number(n_op)
```

```{.python.marimo}

print(f"""Matrix Size: {1e-9*(testSWT_cupy.givens_rotation_matrix(0, 1).data.nbytes):.2f} GB""")
```

```{.python.marimo}
mo.stop(not benchmark_cpu_checkbox.value, "CPU Benchmarks are disabled.")
test_op_scipy = qcheffOperator(spsparse.csr_array(base_mat))
testSWT_scipy = NPAD(test_op_scipy)
print(f"""Matrix Size: {1e-9*(test_op_scipy.op.data.nbytes):.2f} GB""")

```

```{.python.marimo}
mo.stop(not benchmark_gpu_checkbox.value, "GPU Benchmarks are disabled.")

test_op_cupy = qcheffOperator(cpsparse.csr_matrix(base_mat))
testSWT_cupy = NPAD(test_op_cupy)
print(f"""Matrix Size: {1e-9*(test_op_cupy.op.data.nbytes):.2f} GB""")
```

```{.python.marimo}
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
        aspect=0.8,
    )
    npad_bench_plot.ax.set_yscale("log")
    npad_bench_plot.set_axis_labels("Matrix dimension", "Time (s)")
    
    
    npad_bench_plot.set_xticklabels(
        [rf"$10^{int(i)}$" for i in np.log10(test_n_list)]
    )
    sns.move_legend(npad_bench_plot, loc="upper center")
npad_bench_plot
```

```{.python.marimo disabled="true"}
calculate_speedup(bench_df)
```

```{.python.marimo}
def create_bench_df(bench_results, method_name: str, **kwargs):
    incl_params_dict = kwargs.get("incl_params_dict", {})

    return pl.DataFrame(
        more_itertools.zip_broadcast(
            *list(incl_params_dict.values()),
            method_name,
            bench_results.cpu_times,
            bench_results.gpu_times.squeeze(),
        ),
        schema=[*list(incl_params_dict.keys()), "Method", "cpu_time", "gpu_time"],
    )
```

```{.python.marimo}
from more_itertools import zip_broadcast
```

```{.python.marimo}
from cupyx.profiler import benchmark
```

```{.python.marimo}
import cupy as cp
```

```{.python.marimo}
import numpy as np
```

```{.python.marimo}
import cupyx.scipy.sparse as cpsparse
```

```{.python.marimo}
import scipy.sparse as spsparse
```

```{.python.marimo}
import cupyx.scipy
```

```{.python.marimo}
import seaborn as sns
```

```{.python.marimo}
import polars as pl
```

```{.python.marimo}
import plotly.express as px
import more_itertools
import itertools
```

```{.python.marimo disabled="true"}
def calculate_speedup(bench_df):
    mean_times = (
        bench_df
        .group_by("method")
        .agg(pl.mean("time"))
        .rows_by_key("method")
    )
    cupy_time = mean_times["NPAD GPU"][0]
    scipy_time = mean_times["NPAD CPU"][0]

    speedup = scipy_time/cupy_time if scipy_time >= cupy_time else -cupy_time/scipy_time
    return speedup

```

```{.python.marimo}
import sys
```