---
title: Magnus Bench
marimo-version: 0.8.13
width: medium
---

```python {.marimo}
import marimo as mo
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from cupyx.profiler import benchmark
import cupy as cp
import scipy.sparse as spsparse
import cupyx.scipy.sparse as cpspars
import seaborn as sns
import itertools
import more_itertools
import polars as pl
```

```python {.marimo}
from qcheff.models.spin_chain.utils import (
    setup_magnus_chain_example,
    state_transfer_infidelity,
)
```

```python {.marimo}
from magnus_benchmark_utils import bench_magnus, magnus_bench_report, measure_accuracy
```

```python {.marimo column="1" hide_code="true"}
magnus_param_form = mo.ui.batch(
    mo.md(
        r"""
        # Magnus Simulation Parameters

        Chain size $N$: {chain_size}

        Gate time $t$: {gate_time} / {num_tlist} points / {num_magnus_intervals} intervals.

        Qubit frequency $\omega_q$ : {qubit_freq} GHz

        NN Interaction strength $J$ : {J} GHz
        
        NNN Interaction strength $g$ : {g} GHz

        Device: {device} 

        GPU # : {gpu_id}

        
        Which benchmarks to run: {qutip_bench}   {scipy_bench}  {cupy_bench}  {sparse}
        
        """
    ),
    {
        "qubit_freq": mo.ui.slider(
            start=0.5,
            stop=10,
            step=0.1,
            value=1,
            show_value=True,
        ),
        "g": mo.ui.number(
            start=0.0,
            stop=10.0,
            step=0.001,
            value=0.005,
        ),
        "J": mo.ui.number(
            start=0.0,
            stop=10.0,
            step=0.01,
            value=0.05,
        ),
        "chain_size": mo.ui.number(
            start=3,
            stop=15,
            step=1,
            value=5,
        ),
        "gate_time": mo.ui.number(
            start=10,
            stop=1000,
            step=1,
            value=25,
        ),
        "num_tlist": mo.ui.number(
            start=10,
            stop=100000,
            step=10,
            value=200,
        ),
        "num_magnus_intervals": mo.ui.number(
            start=10,
            stop=100000,
            step=10,
            value=100,
        ),
        "device": mo.ui.radio(
            options=["cpu", "gpu"],
            value="gpu",
        ),
        "gpu_id": mo.ui.number(start=0, stop=7, step=1, value=0),
        "sparse": mo.ui.checkbox(value=False, label="Sparse"),
        "qutip_bench": mo.ui.checkbox(value=True, label="QuTiP"),
        "scipy_bench": mo.ui.checkbox(value=False, label="SciPy"),
        "cupy_bench": mo.ui.checkbox(value=True, label="CuPy"),
    },
).form()
magnus_param_form
```

```python {.marimo column="2" hide_code="true"}
mo.md(r"""# Accuracy benchmarking""")
```

```python {.marimo hide_code="true"}
fig, ax = plt.subplots(1, 1)
ax.loglog(
    num_tpts_list[:-1],
    qutip_errs[0][:-1],
    "o-",
    markersize=10,
    label="QuTiP Numerical Integration",
    lw=3,
    alpha=0.5,
    color="slategray",
)
ax.loglog(
    num_tpts_list[:-1],
    magnus_cpu_errs[0][:-1],
    "X-",
    markersize=10,
    label=r"${\rm qCH_{\rm eff}}$ Magnus",
    lw=3,
    alpha=0.5,
    color=(118 / 255, 185 / 255, 0 / 255),
)
# ax.loglog(
#     num_tpts_list[:-1],
#     magnus_gpu_errs[0][:-1],
#     marker="P",
#     markersize=10,
#     label="Magnus GPU",
#     lw=3,
# color=(0 / 255, 104 / 255, 181 / 255),
#     alpha=0.5,
# )
ax.axhline(y=qutip_errs[0][-2], ls="--", lw=3, color="slategray")
ax.annotate("QuTiP Minimum Integration Error", xy=(40, 3e-10), fontsize=12)
ax.legend(fontsize=10, frameon=False, loc="upper right")
ax.set(
    xlabel="Time Points (QuTiP)/ Magnus Intervals",
    ylabel=r"Final state error",
    xlim=more_itertools.minmax(num_tpts_list[:-1]),
    ylim=(1e-30, 1),
)
fig
```

```python {.marimo}
test_system, test_magnus = setup_magnus_chain_example(
    pulse_coeffs=test_coeffs,
    # num_tlist=max(num_tpts_list),
    **magnus_param_form.value | {"chain_size": 4},
)
```

```python {.marimo}
magnus_param_form.value
```

```python {.marimo}
*magnus_gpu_errs, magnus_gpu_final_state = measure_accuracy(
    pulse_coeffs=test_coeffs,
    **(
        {k: v for k, v in magnus_param_form.value.items() if k != "num_tlist"}
        | {
            "sample_num_list": num_tpts_list,
            "method_name": "magnus",
            "device": "gpu",
        }
    ),
)
magnus_gpu_errs
```

```python {.marimo}
*magnus_cpu_errs, magnus_cpu_final_state = measure_accuracy(
    pulse_coeffs=test_coeffs,
    **(
        {k: v for k, v in magnus_param_form.value.items() if k != "num_tlist"}
        | {
            "sample_num_list": num_tpts_list,
            "method_name": "magnus",
            "device": "cpu",
        }
    ),
)
magnus_cpu_errs
```

```python {.marimo}
*qutip_errs, qutip_final_state = measure_accuracy(
    pulse_coeffs=test_coeffs,
    **(
        {k: v for k, v in magnus_param_form.value.items() if k != "num_tlist"}
        | {
            "sample_num_list": num_tpts_list,
            "method_name": "qutip",
        }
    ),
)
qutip_errs
```

```python {.marimo}
test_coeffs = [
    0.9817927559296183,
    -1.0,
    -0.19283820271219573,
    -0.9097270782845742,
    -0.7677860132667245,
    -0.9997505100417258,
    1.0,
    1.0,
    -1.0,
    0.7714579619598767,
]
num_tpts_list = [
    # 2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    # 10000,
    # 20000,
    # 100000,
    # 200000,
]
```

```python {.marimo}
list(
    map(
        lambda x: qt.hilbert_dist(x[0], x[1]),
        itertools.combinations(
            (magnus_cpu_final_state, magnus_gpu_final_state, qutip_final_state), 2
        ),
    )
)
```

```python {.marimo column="3" hide_code="true"}
mo.md(r"""# Performance benchmarks""")
```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to continue")
chain_size_list = [7, 8, 9]
bench_df = bench_magnus(
    pulse_coeffs=np.array(
        [
            0.9817927559296183,
            -1.0,
            -0.19283820271219573,
            -0.9097270782845742,
            -0.7677860132667245,
            -0.9997505100417258,
            1.0,
            1.0,
            -1.0,
            0.7714579619598767,
        ]
    ),
    chain_size_list=chain_size_list,
    **(magnus_param_form.value | {"qutip_tpts": 1000, "magnus_tpts": 20}),
)
```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to continue")
bench_plot = magnus_bench_report(bench_df)
sns.move_legend(bench_plot, loc="upper center")
bench_plot.set_axis_labels("Matrix Dimension")
bench_plot.set_xticklabels((rf"$2^{i}$" for i in chain_size_list))
```