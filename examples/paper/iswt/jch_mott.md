---
title: Jch Mott
marimo-version: 0.8.3
width: medium
---

```{.python.marimo}
from qcheff.jaynes_cummings_hubbard.models import JCHModel
```

```{.python.marimo}
jch_param_form = mo.ui.batch(
    mo.md(
        r"""
        # Jaynes-Cummings-Hubbard Model Parameters

        Chain size $N$: {n}

        Resonator frequency range $\omega/g$ : {resonator_freqs} GHz

        Resonator detuning $\Delta/g =  (\epsilon-\omega)/g$ : {detunings} GHz, {nr} levels.

        Cavity Tunneling Rate $\kappa/g$ : {kappa} GHz

        Chemical Potential $\mu/g$ : {mu} GHz
        """
    ),
    {
        "resonator_freqs": mo.ui.slider(
            start=0,
            stop=100,
            step=0.1,
            value=1,
            show_value=True,
        ),
        "detunings": mo.ui.slider(
            start=-5,
            stop=5,
            step=0.01,
            value=0.5,
            show_value=True,
        ),
        "kappa": mo.ui.number(
            start=0.0,
            stop=10.0,
            step=0.01,
            value=0.00,
        ),
        "nr": mo.ui.number(
            start=2,
            stop=20,
            step=1,
            value=3,
        ),
        "n": mo.ui.number(
            start=1,
            stop=20,
            step=1,
            value=2,
        ),
        "mu": mo.ui.number(
            start=0.0,
            stop=10.0,
            step=0.01,
            value=0.00,
        ),
    },
).form()
jch_param_form
```

```{.python.marimo}
def models_gen(**kwargs):
    g = kwargs.get("g", 0.01)
    kappa = kwargs.get("kappa", 0.00)
    nr = kwargs.get("nr", 3)
    delta = kwargs.get("detunings")
    wr_range = kwargs.get("wr_range")
    n = kwargs.get("chain_size", 2)
    wr_list = np.repeat(np.float64(kwargs.get("resonator_freqs")), n)
    mu = kwargs.get("mu", 0)
    model_params_in_form = ["g", "kappa", "mu"]

    yield JCHModel(
        **{
            "resonator_freqs": wr_list,
            "detunings": np.repeat(np.float64(delta), n),
            "g": g,
            "kappa": kappa,
            "mu": mu,
            "nr" : nr,
        }
    )
```

```{.python.marimo}
JCHModel(**jch_param_form.value)
```

```{.python.marimo}
test_model = next(models_gen(**jch_param_form.value))
```

```{.python.marimo}
test_hs = test_model.jch_scqubits_hilbertspace()
```

```{.python.marimo}
detuning_list = np.linspace(0.1, 1, 101)
detuning_dict = {"detuning": detuning_list}


def update_jch_detuning(param_sweep, detuning):
    for i in range(
        len(param_sweep.hilbertspace.subsystem_list) // 2
    ):  # better iteration helper needed
        param_sweep.hilbertspace[f"q_{i}"].E_osc = (
            param_sweep.hilbertspace[f"r_{i}"].E_osc - detuning
        )


jch_paramsweep = scq.ParameterSweep(
    hilbertspace=test_hs,
    paramvals_by_name=detuning_dict,
    update_hilbertspace=update_jch_detuning,
    evals_count=11,
)
```

```{.python.marimo}
A_jc_evals = (
    (
        pl.DataFrame(
            detuning_dict
            | {
                "".join(map(str, label)): jch_paramsweep.energy_by_bare_index(
                    label
                )
                for label in subsys_A_labels
            }
        )
        .unpivot(
            index="detuning",
            variable_name="level",
            value_name="energy",
        )
        .with_columns(
            pl.col("level")
            .str.split_exact("", 4)
            .struct.rename_fields(["q0", "q1", "r0", "r1"])
            .alias("states"),
        )
        .unnest("states")
    )
    .with_columns(
        (pl.col("q0").cast(pl.Int8) + pl.col("r0").cast(pl.Int8)).alias(
            "polariton_number"
        ),
    )
    .with_columns(
        (
            pl.col("energy")
            - pl.col("polariton_number") * (test_hs["r_0"].E_osc)
        ).alias("corrected_energy")
    )
)
```

```{.python.marimo}
sns.lineplot(
    A_jc_evals.filter(
        (pl.col("q0") == "1") & (pl.col("r1") == "0") & (pl.col("q1") == "0")
    ),
    x="detuning",
    y="energy",
    hue="polariton_number",
)
```

```{.python.marimo}
mott_df = (
    A_jc_evals.filter(
        (pl.col("q0") == "1") & (pl.col("r1") == "0") & (pl.col("q1") == "0")
    )
    .pivot(on="polariton_number", index="detuning", values="energy")
    .with_columns(
        (
            (
                pl.col("1")
                - (test_hs["q_0"].E_osc + pl.col("detuning")) / test_model.g
            ).alias("mu_critical_10")
        ),
        (
            (pl.col("2") - pl.col("1") - (test_hs["r_0"].E_osc)) / test_model.g
        ).alias("mu_critical_21"),
        (
            (
                pl.col("3")
                - pl.col("2")
                - (test_hs["q_0"].E_osc + pl.col("detuning"))
            )
            / test_model.g
        ).alias("mu_critical_32"),
    )
).unpivot(
    index="detuning",
    on=[
        "mu_critical_10",
        "mu_critical_21",
        "mu_critical_32",
    ],
    variable_name="level_pair",
    value_name="mu_critical",
)
```

```{.python.marimo}
test_model.g
```

```{.python.marimo}
mott_ax = sns.lineplot(
    mott_df, x="detuning", y="mu_critical", hue="level_pair"
)
# mott_ax.set(yscale="log")
mott_ax
```

```{.python.marimo}

```

```{.python.marimo}
mott_df
```

```{.python.marimo}
pl.DataFrame(
    [
        {
            "level": level,
            "dressed_idx": test_hs.dressed_index(level),
            "dressed_energy": test_hs.energy_by_bare_index(level),
        }
        for level in subsys_A_labels
    ],
)
```

```{.python.marimo}

```

```{.python.marimo}
subsys_A_labels = [
    # ground
    (0, 0, 0, 0),
    # 1 polariton
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    # 2 polaritons
    (1, 0, 1, 0),
    (0, 0, 2, 0),
    # 3 polaritons
    (1, 0, 2, 0),
    (0, 0, 3, 0),
]
subsys_B_labels = [
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 1, 0, 0),
    (0, 1, 0, 1),
    (0, 1, 0, 2),
]
```

```{.python.marimo}
mo.mpl.interactive(
    sns.relplot(
        A_jc_evals,
        kind="line",
        x="detuning",
        y="corrected_energy",
        style="polariton_number",
        hue="q0",
    ).ax
)
```

```{.python.marimo}
plt.imshow(
    np.abs(test_hs.hamiltonian()[:]), cmap="binary", norm=mpl.colors.LogNorm()
)
```

```{.python.marimo}
import cupyx
from tqdm.notebook import tqdm
import itertools
import more_itertools
from time import sleep
from qcheff.operators.operator_base import qcheffOperator
from qcheff.iswt.iswt import NPAD
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