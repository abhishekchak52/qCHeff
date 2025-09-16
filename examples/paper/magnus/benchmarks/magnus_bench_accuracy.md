---
title: Magnus Bench
marimo-version: 0.8.13
width: medium
---

```python {.marimo name="setup"}
import marimo as mo
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from cupyx.profiler import benchmark
import cupy as cp
import scipy.sparse as spsparse
import cupyx.scipy.sparse as cpspars
import seaborn as sns
import seaborn.objects as so
import matplotlib
import itertools
import more_itertools
from more_itertools import last
import pandas as pd
import polars as pl
import hashlib
from collections import ChainMap
# Initialization code that runs before all other cells
```

```python {.marimo}
from qcheff.models.spin_chain.utils import (
    setup_magnus_chain_example,
    state_transfer_infidelity,
)
```

```python {.marimo}
rng = np.random.default_rng()
```

```python {.marimo}
from magnus_benchmark_utils import (
    bench_magnus,
    magnus_bench_report,
    measure_accuracy,
)
```

```python {.marimo}
_chain_size = 6
test_system, test_magnus = setup_magnus_chain_example(
    pulse_coeffs=test_coeffs,
    # num_tlist=max(num_tpts_list),
    **magnus_param_form.value | {"chain_size": _chain_size},
)
allzero_state = qt.basis(dimensions=[2] * _chain_size, n=[0] * _chain_size)
```

```python {.marimo}
test_res = qt.sesolve(
    H=test_system.get_qutip_tdham(local_tlist),
    psi0=allzero_state,
    tlist=local_tlist,
    options={"store_final_state": True, "store_states": True},
)
```

```python {.marimo}
_local_tnum = 250
local_tlist = np.insert(
    np.take(
        test_magnus.tlist.reshape((_local_tnum, -1)),
        indices=np.asarray([-1]),
        axis=1,
    ).squeeze(),
    obj=0,
    values=np.asarray([0]),
)
magnus_states = [
    qt.Qobj(cp.asnumpy(state), dims=[[2] * 6, 1])
    for state in test_magnus.evolve(
        init_state=allzero_state[:],
        num_intervals=_local_tnum,
    )
]
magnus_states[:3]
```

```python {.marimo}
qutip_pops = list(
    map(lambda x: qt.expect(allzero_state.proj(), x), test_res.states[1:])
)
magnus_pops = list(
    map(lambda x: qt.expect(allzero_state.proj(), x), magnus_states)
)
```

```python {.marimo}
def measure_accuracy_both(
    pulse_coeffs, sample_num_list: tuple[float, ...], **kwargs
):
    chain_size = kwargs.get("chain_size")

    allzero_state = qt.basis(
        dimensions=[2] * chain_size, n=[0] * chain_size
    ).unit()

    sim_options = ChainMap(dict(num_tlist=max(sample_num_list)), kwargs)

    test_system, test_magnus = setup_magnus_chain_example(
        pulse_coeffs=pulse_coeffs, **sim_options
    )

    def local_tlist(sample_num):
        return np.take(
            test_magnus.tlist.reshape((sample_num, -1)),
            indices=np.asarray([-1]),
            axis=1,
        ).squeeze()

    def qutip_final_state(sample_num):
        return qt.sesolve(
            H=test_system.get_qutip_tdham(local_tlist(sample_num)),
            psi0=allzero_state,
            tlist=local_tlist(sample_num),
            options={"store_final_state": True, "store_states": True},
        ).final_state

    def magnus_final_state(num_intervals):
        return qt.Qobj(
            cp.asnumpy(
                more_itertools.last(
                    test_magnus.evolve(
                        init_state=allzero_state[:],
                        num_intervals=num_intervals,
                    )
                )
            )
        )

    final_states = {
        "chain_size": chain_size,
        "pulse_frequency": kwargs.get("pulse_freq"),
        "pulse_coeffs": pulse_coeffs,
        "num_points": sample_num_list,
        "magnus": list(map(magnus_final_state, sample_num_list)),
        "qutip": list(map(qutip_final_state, sample_num_list)),
    }

    return final_states
```

```python {.marimo}
sim_options = ChainMap(dict(chain_size=4), magnus_param_form.value)

accuracy_data = measure_accuracy_both(
    pulse_coeffs=test_coeffs,
    sample_num_list=num_tpts_list,
    **sim_options,
)
# accuracy_data
```

```python {.marimo}
manypulse_overlap_data_long = manypulse_overlap_data.drop("fidelity").unpivot(
    index=["num_points", "method"],
    variable_name="metric",
    value_name="metric_value",
)
manypulse_overlap_data_long
```

```python {.marimo}
def calculate_overlaps(result, reference):
    overlap_funcs = [qt.hilbert_dist, qt.fidelity, qt.hellinger_dist]
    overlap_dict = {
        func.__name__: func(result, reference) for func in overlap_funcs
    }
    return overlap_dict
```

```python {.marimo}
def calculate_overlap_with_final_state(states):
    final_state = states[-1]
    return [
        calculate_overlaps(state, reference=final_state) for state in states
    ]
```

```python {.marimo}
overlap_data = (
    pl.DataFrame(
        {
            "num_points": accuracy_data["num_points"],
            "qutip": calculate_overlap_with_final_state(
                accuracy_data["qutip"]
            ),
            "magnus": calculate_overlap_with_final_state(
                accuracy_data["magnus"]
            ),
        }
    )
    .unpivot(index="num_points", variable_name="method", value_name="overlaps")
    .unnest("overlaps")
    .with_columns(infidelity=1 - pl.col("fidelity"))
)
overlap_data
```

```python {.marimo}
overlap_data_long = overlap_data.drop("fidelity").unpivot(
    index=["num_points", "method"],
    variable_name="metric",
    value_name="metric_value",
)
overlap_data_long
```

```python {.marimo column="1" hide_code="true"}
with sns.axes_style("ticks"), sns.plotting_context("notebook"):
    accuracy_fig, accuracy_ax = plt.subplots(
        1, 1, layout="constrained", figsize=(4.5, 3.5)
    )

    acc_plot = (
        so.Plot(
            manypulse_overlap_data.sort("num_points"),
            x="num_points",
            y="hilbert_dist",
            color="method",
        )
        .add(so.Dots(alpha=0.1), legend=False)
        .add(so.Line(), so.Agg("mean"))
        .add(so.Range(), so.Est(errorbar="se"))
        .scale(x="log", y="log", color=["darkslategray", "#76B900"])
        .limit(x=(10, 10000), y=(1e-20, 10))
        .label(
            x="Time Points (QuTiP)/ Magnus Intervals", y=r"Final state error"
        )
        .on(accuracy_ax)
        .plot(pyplot=True)
    )
    legend = acc_plot._figure.legends.pop(0)
    accuracy_ax.legend(
        legend.legend_handles,
        [r"QuTiP Numerical Integration", r"${\rm qCH_{\rm eff}}$ Magnus"],
        frameon=False,
    )
    accuracy_ax.grid(axis="y")
accuracy_fig
```

```python {.marimo disabled="true"}
accuracy_fig.savefig("magnus_err.pdf")
```

```python {.marimo}
drive_freq_list = 2 * np.pi * np.linspace(0, 10, 20)
```

```python {.marimo}
manypulse_accuracy_data = list(
    measure_accuracy_both(
        pulse_coeffs=test_coeffs,
        sample_num_list=num_tpts_list,
        **sim_options,
        pulse_freq=wd,
    )
    for wd in mo.status.progress_bar(drive_freq_list)
)
```

```python {.marimo}
manypulse_accuracy_data
```

```python {.marimo}
manypulse_overlap_data = (
    pl.concat(
        [
            pl.DataFrame(
                {
                    "chain_size": _accuracy_data["chain_size"],
                    "pulse_frequency": _accuracy_data["pulse_frequency"],
                    "num_points": _accuracy_data["num_points"],
                    "qutip": calculate_overlap_with_final_state(
                        _accuracy_data["qutip"]
                    ),
                    "magnus": calculate_overlap_with_final_state(
                        _accuracy_data["magnus"]
                    ),
                }
            )
            for _accuracy_data in manypulse_accuracy_data
        ]
    )
    .unpivot(
        index=["chain_size", "pulse_frequency", "num_points"],
        variable_name="method",
        value_name="overlaps",
    )
    .unnest("overlaps")
    .with_columns(infidelity=1 - pl.col("fidelity"))
)
manypulse_overlap_data
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
ax.annotate("QuTiP Minimum Integration Error", xy=(40, 1e-9), fontsize=12)
ax.legend(fontsize=10, frameon=False, loc="upper right")
ax.set(
    xlabel="Time Points (QuTiP)/ Magnus Intervals",
    ylabel=r"Final state error",
    xlim=more_itertools.minmax(num_tpts_list[:-1]),
    ylim=(1e-15, 10),
)
fig
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
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    #     100000,
]
```

```python {.marimo}
magnus_param_form.value
```

```python {.marimo disabled="true"}
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

```python {.marimo disabled="true"}
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

```python {.marimo disabled="true"}
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
list(
    map(
        lambda x: qt.hilbert_dist(x[0], x[1]),
        itertools.combinations(
            (
                magnus_cpu_final_state,
                magnus_gpu_final_state,
                qutip_final_state,
            ),
            2,
        ),
    )
)
```

```python {.marimo column="3" hide_code="true"}
mo.md(r"""# Performance benchmarks""")
```

```python {.marimo hide_code="true"}
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
            start=1,
            stop=1000,
            step=1,
            value=4,
        ),
        "num_tlist": mo.ui.number(
            start=10,
            stop=100000,
            step=10,
            value=500,
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

```python {.marimo disabled="true"}
mo.stop(not magnus_param_form.value, "Submit form to continue")
chain_size_list = [5, 6, 7, 8]
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

```python {.marimo}
bench_df
```

```python {.marimo}
with sns.axes_style("ticks"), sns.plotting_context("notebook"):
    magnus_bench_fig, magnus_bench_ax = plt.subplots(
        figsize=(6, 4), layout="constrained"
    )

    magplot = (
        so.Plot(
            bench_df, x="chain_size", y="time", color="Method", marker="Method"
        )
        .add(so.Dots(alpha=0.5, pointsize=7))
        .add(so.Line(alpha=1, marker=False), so.Agg())
        .scale(
            x=so.Continuous().tick(every=1).label(like="$2^{x:g}$"),
            y=so.Continuous(trans="log").label(like="{x:.3g}"),
            color=so.Nominal(
                {
                    "QuTiP": "slategray",
                    "Magnus (CPU)": "#127cc1",
                    "Magnus (GPU)": "#76b900",
                }
            ),
            marker=so.Nominal(["s", "v", "^"]),
        )
        .limit(x=more_itertools.minmax(chain_size_list))
        .label(x="Matrix Dimension", y="Time (s)")
        .on(magnus_bench_ax)
        .plot(pyplot=True)
    )
    magnus_bench_ax.grid(axis="y")
    _legend = magplot._figure.legends[0]
    _legend.set_frame_on(False)
    _legend.set_bbox_to_anchor((0.98, 0.35))
magnus_bench_fig
```