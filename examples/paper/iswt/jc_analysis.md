---
title: Jc Analysis
marimo-version: 0.8.13
width: medium
---

```{.python.marimo disabled="true"}
bench_df = (
    JCMottAnalysis(model=test_model, level_labels=polariton_levels)
    .benchmark(
        methods=[
            "NPAD (CPU)",
            "NPAD (GPU)",
            # "scQubits",
        ],
        num_couplings=500,
    )
    .drop("cpu_time")
    .rename({"gpu_time": "time"})
)
```

```{.python.marimo}
npad_bench_plot = sns.catplot(
    bench_df,
    kind="bar",
    hue_order=["scQubits", "NPAD (CPU)", "NPAD (GPU)"],
    palette={
        "scQubits": "slategray",
        "NPAD (CPU)": (0 / 255, 104 / 255, 181 / 255),
        "NPAD (GPU)": (118 / 255, 185 / 255, 0 / 255),
    },
    x="Method",
    y="time",
    hue="Method",
    errorbar="ci",
    err_kws={"linewidth": 1},
    capsize=0.5,
    aspect=0.8,
)
npad_bench_plot.ax.set_yscale("log")
npad_bench_plot.set_axis_labels("Subroutine", "Time (s)")

npad_bench_plot
```

```{.python.marimo}
from qcheff.operators import qcheffOperator
from qcheff.iswt import NPAD
```

```{.python.marimo}
from qcheff.models.jaynes_cummings.models import JCModel
from qcheff.models.jaynes_cummings.utils import JCMottAnalysis
```

```{.python.marimo}
jc_param_form
```

```{.python.marimo}
test_model = JCModel(**jc_param_form.value)
```

```{.python.marimo}
test_hs = test_model.jc_scqubits_hilbertspace()
```

```{.python.marimo}
detuning_list = np.linspace(-10, 10, 101)
detuning_dict = {"detuning": detuning_list}


# def update_jch_detuning(param_sweep, detuning):
#     param_sweep.hilbertspace["q"].E_osc = (
#         param_sweep.hilbertspace["r"].E_osc - detuning
#     )


# jch_paramsweep = scq.ParameterSweep(
#     hilbertspace=test_hs,
#     paramvals_by_name=detuning_dict,
#     update_hilbertspace=update_jch_detuning,
#     evals_count=11,
# )
```

```{.python.marimo}
jch_evals = (
    (
        pl.DataFrame(
            detuning_dict
            | {
                "".join(map(str, label)): jch_paramsweep.energy_by_bare_index(
                    label
                )
                for label in polariton_levels
            }
        )
        .unpivot(
            index="detuning",
            variable_name="level",
            value_name="energy",
        )
        .with_columns(
            pl.col("level")
            .str.split_exact("", 2)
            .struct.rename_fields(["q", "r"])
            .alias("states"),
        )
        .unnest("states")
    )
    .with_columns(
        (pl.col("q").cast(pl.Int8) + pl.col("r").cast(pl.Int8)).alias(
            "polariton_number"
        ),
    )
    .with_columns(
        (
            pl.col("energy")
            - pl.col("polariton_number") * (test_hs["r"].E_osc)
        ).alias("corrected_energy")
    )
)
```

```{.python.marimo disabled="true"}
jc_ax = sns.relplot(
    jch_evals,
    kind="line",
    x="detuning",
    y="corrected_energy",
    hue="polariton_number",
    palette=sns.color_palette("tab10"),
    style="q",
    aspect=0.8,
).ax
jc_ax.set(
    # yscale="symlog",
    xlabel=r"$\Delta/g$",
    ylabel=r"(Energy- Optical Energy)/$g$",
    xlim=more_itertools.minmax(detuning_list),
)
jc_ax
```

```{.python.marimo}
npad_error_df
```

```{.python.marimo}
mott_df = (
    all_evals_df.filter(
        ((pl.col("qubit state") == 1) & (pl.col("detuning") > 0))
        | ((pl.col("qubit state") == 0) & (pl.col("detuning") < 0))
    )
    .pivot(
        on="polariton_number", index=["detuning", "Method"], values="energy"
    )
    .with_columns(
        ((pl.col("1") - (test_hs["r"].E_osc)).alias("0")),
        *(
            (pl.col(str(j)) - pl.col(str(i)) - (test_hs["r"].E_osc)).alias(
                f"{i}"
            )
            for i, j in itertools.pairwise(
                range(1, max(itertools.chain(*polariton_levels)))
            )
        ),
    )
).unpivot(
    index=["detuning", "Method"],
    on=[
        f"{i}"
        for i, j in itertools.pairwise(
            range(1, max(itertools.chain(*polariton_levels)))
        )
    ],
    variable_name="Level",
    value_name="Critical Chemical Potential",
)
```

```{.python.marimo}
mott_df.filter(pl.col("Method").eq("NPAD (CPU)")).sort(by="Level")
```

```{.python.marimo}
npad_jc_fig, npad_jc_axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    layout="constrained",
    figsize=(5, 8),
)
with sns.plotting_context("paper"):
    mott_ax = sns.lineplot(
        mott_df.filter(pl.col("Method").eq("NPAD (CPU)")).sort(by="Level"),
        x="detuning",
        y="Critical Chemical Potential",
        hue="Level",
        palette=sns.color_palette("Blues_r", n_colors=8),
        legend=False,
        lw=2,
        ax=npad_jc_axes[0],
    )
    mott_ax.plot(
        detuning_list,
        test_model.critical_chemical_potential(
            detuning_list,
            np.array(range(1, max(itertools.chain(*polariton_levels)) - 1))[
                :, None
            ],
        ).T,
        markevery=5,
        marker="x",
        color="k",
        lw=0,
        markersize=5,
    )
    mott_ax.set(
        xlim=more_itertools.minmax(detuning_list),
        xlabel=r"$\Delta/g$",
    )
    # mott_plot.fig.subplots_adjust(wspace=0.1)
    legend_elements = [
        plt.Line2D([0], [0], color="k", lw=1, label="Numerical"),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="k",
            label="Theory",
            markerfacecolor="k",
            lw=0,
            markersize=5,
        ),
    ]
    mott_ax.legend(
        handles=legend_elements,
        frameon=False,
        loc="lower right",
    )
    mott_ax.set(ylabel=r"$(\mu-\omega)/g$")
    npad_error_plot = sns.lineplot(
        npad_error_df,
        x="detuning",
        y="Error",
        hue="Method",
        palette={
            "scQubits Error": "slategrey",
            "NPAD Error": (118 / 255, 185 / 255, 0 / 255),
        },
        lw=1,
        ax=npad_jc_axes[1],
    )
    npad_error_plot.set(
        xlim=more_itertools.minmax(detuning_list),
        yscale="log",
        ylim=(1e-15, 1e-10),
        xlabel=r"$\Delta/g$",
        ylabel="Relative Error",
    )
    sns.despine(npad_jc_fig)
# mo.mpl.interactive(npad_jc_fig)
npad_jc_fig
```

```{.python.marimo}
npad_error_df = (
    mott_df.pivot(
        "Method",
        index=["detuning", "Level"],
        values="Critical Chemical Potential",
    )
    .with_columns(
        (
            (
                pl.col("Level").cast(pl.Int16)
                + pl.col("detuning") * pl.col("detuning") / 4
            ).sqrt()
            - (
                pl.col("Level").cast(pl.Int16)
                + 1
                + pl.col("detuning") * pl.col("detuning") / 4
            ).sqrt()
        ).alias("theory")
    )
    .with_columns(
        (
            (
                pl.col("NPAD (CPU)")
                .sub(pl.col("theory"))
                .truediv(pl.col("scQubits"))
            )
            .abs()
            .rolling_mean(
                window_size=10,
                center=True,
            )
            .alias("NPAD Error")
        ),
        (
            (
                pl.col("scQubits")
                .sub(pl.col("theory"))
                .truediv(pl.col("scQubits"))
            )
            .abs()
            .rolling_mean(
                window_size=10,
                center=True,
            )
            .alias("scQubits Error")
        ),
    )
    .drop_nulls()
).unpivot(
    index=["detuning", "Level"],
    on=["scQubits Error", "NPAD Error"],
    variable_name="Method",
    value_name="Error",
)
npad_error_df
```

```{.python.marimo}
test_op = qcheffOperator(spsparse.csr_array(test_hs.hamiltonian()[:]))
test_op.couplings()
```

```{.python.marimo}
couplings = [(i, (i - 1) + test_model.resonator_levels) for i in range(1, 10)]
print(couplings)
```

```{.python.marimo}
test_NPAD = NPAD(test_op, copy=True)
test_NPAD.eliminate_couplings(couplings)
test_NPAD.H.couplings()
```

```{.python.marimo}

```

```{.python.marimo}
polariton_levels = list(
    more_itertools.flatten(
        ([(0, 0)], *(((1, i), (0, i + 1)) for i in range(10)))
    )
)
print(polariton_levels)
```

```{.python.marimo}
all_evals_df = analysis_df.with_columns(
    (
        pl.col("qubit state").cast(pl.Int8)
        + pl.col("resonator state").cast(pl.Int8)
    ).alias("polariton_number"),
).with_columns(
    (
        pl.col("energy") - pl.col("polariton_number") * (test_hs["r"].E_osc)
    ).alias("corrected_energy")
)
```

```{.python.marimo}
analysis_df = pl.concat(
    JCMottAnalysis(model=test_model, level_labels=polariton_levels).analyse(
        detuning_list=detuning_list, methods=["scqubits", "npad_cpu"]
    )
)
```

```{.python.marimo}
jc_param_form = mo.ui.batch(
    mo.md(
        r"""
        # Jaynes-Cummings Model Parameters

        Resonator frequency range $\omega/g$ : {resonator_freq} 

        Resonator detuning $\Delta/g =  (\epsilon-\omega)/g$ : {detuning}, {resonator_levels} levels.

        """
    ),
    {
        "resonator_freq": mo.ui.number(
            start=5,
            stop=10,
            step=0.1,
            value=5,
        ),
        "detuning": mo.ui.slider(
            start=-5,
            stop=5,
            step=0.1,
            value=1,
            show_value=True,
        ),
        "resonator_levels": mo.ui.number(
            start=5,
            stop=200000000,
            step=1,
            value=10,
        ),
    },
).form()
```

```{.python.marimo}
pl.DataFrame(
    [
        {
            "level": level,
            "dressed_idx": test_hs.dressed_index(level),
            "dressed_energy": test_hs.energy_by_bare_index(level),
        }
        for level in polariton_levels
    ],
)
```

```{.python.marimo}
import cupyx
from tqdm.notebook import tqdm
import itertools
import more_itertools
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import plotly.express as px
import cupy as cp
import marimo as mo
import scqubits as scq
from itertools import product
import matplotlib as mpl
import seaborn as sns
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as spsparse
from copy import copy
import qutip as qt
import cupyx.scipy.sparse.linalg as cpla
```