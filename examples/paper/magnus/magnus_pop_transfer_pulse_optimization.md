---
title: Magnus Comparison
marimo-version: 0.8.11
width: medium
---

```python {.marimo name="setup"}
# Initialization code that runs before all other cells
import copy
import itertools
import functools
import more_itertools

import marimo as mo
import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as spsparse
import polars as pl
import qutip as qt

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

from scipy.optimize import minimize
```

```python {.marimo}
from qcheff.operators import DenseOperator, SparseOperator
from qcheff.models.spin_chain.utils import embed_operator
from qcheff.utils.system import QuTiPSystem
from qcheff.utils.pulses import FourierPulse

from qcheff.models.spin_chain.utils import (
    setup_magnus_chain_example,
    state_transfer_infidelity,
)
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
        
        Sparse: {sparse}
        
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
        "sparse": mo.ui.checkbox(value=False),
    },
).form()
magnus_param_form
```

```python {.marimo}
manual_coeffs = res.x
# manual_coeffs = np.random.random(size=10)
```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to start")

test_system, test_magnus = setup_magnus_chain_example(
    pulse_coeffs=manual_coeffs,
    **(
        magnus_param_form.value
        | {
            "device": "gpu",
            "pulse_freq": 0,
        }
    ),
    debug=True,
)
```

```python {.marimo}
type(test_magnus)
```

```python {.marimo hide_code="true"}
labels = [
    r"$P_{|0^{\otimes n}\rangle}$",
    r"$P_{|1^{\otimes n}\rangle}$",
    r"$P_{\rm rest}$",
]
with (
    sns.plotting_context("notebook"),
    mpl.rc_context({"mathtext.fontset": "cm"}),
):
    fig, ax = plt.subplots(
        2, 1, figsize=(5, 5.5), layout="constrained", sharex=True
    )
    for sig, lc, ls in zip(
        test_system.control_sigs,
        ["xkcd:tomato red", "xkcd:pinkish orange"],
        ["-", "--"],
    ):
        ax[0].plot(
            test_magnus.tlist,
            sig(test_magnus.tlist),
            label=sig.name,
            ls=ls,
            color=lc,
            lw=3,
        )
        ax[0].set(ylim=(-1, 1))
        ax[0].set_ylabel("Amplitude")

    magnus_plots = [
        ax[1].plot(mag_tlist, pop, label=label, lw=2, color=lc, ls=ls)[0]
        for pop, label, lc, ls in zip(
            pops,
            labels,
            ["xkcd:darkish blue", "xkcd:ultramarine blue", "xkcd:water blue"],
            ["-", "-.", ":"],
        )
    ]

    for pop, label in zip(qutip_pops, labels):
        ax[1].plot(
            test_magnus.tlist,
            pop,
            # label=label + " (QuTiP)",
            markevery=50,
            marker="+",
            lw=0,
            color="black",
        )

    ax[1].set(
        xlim=test_magnus.tlims,
        xlabel="Time (ns)",
        ylabel="Population",
        ylim=(0, 1),
    )
    ax[0].set(ylim=(-0.5, 0.5))
    ax[0].legend(
        fontsize=15,
        frameon=False,
        labels=[r"$u^{x}(t)$", r"$u^{y}(t)$"],
    )
    patch = plt.Line2D(
        [0],
        [0],
        marker="+",
        color="k",
        label="QuTiP",
        markerfacecolor="k",
        lw=0,
        markersize=10,
    )
    ax[1].legend(
        handles=[*magnus_plots, patch],
        fontsize=12,
        # loc="center left",
        frameon=False,
    )
    infidelity = 1 - qt.expect(P1, test_states[-1])
    fig.suptitle(f"Chain size: N={chain_size}, Error: {infidelity:.2e}")
fig
```

```python {.marimo}
fig.savefig("magnus_opt.pdf")
```

```python {.marimo}
chain_size = int(magnus_param_form.value["chain_size"])
```

```python {.marimo}
allzero_state = qt.basis(dimensions=[2] * chain_size, n=[0] * chain_size)
allone_state = qt.basis([2] * chain_size, n=[1] * chain_size)

P0 = allzero_state.proj()
P1 = allone_state.proj()
P_rest = qt.qeye([2] * chain_size) - P0 - P1
eops = [P0, P1, P_rest]
```

```python {.marimo}
test_psi0 = np.asarray((allzero_state).unit()[:])
magnus_states = test_magnus.evolve(
    init_state=test_psi0,
    num_intervals=magnus_param_form.value["num_magnus_intervals"],
)
test_states = [
    qt.Qobj(cp.asnumpy(state), dims=[[2] * chain_size, [1] * chain_size])
    for state in magnus_states
]
```

```python {.marimo}
tsmag = test_system.get_magnus_system(
    tlist=test_magnus.tlist,
    device="cpu",
    sparse=True,
)
```

```python {.marimo}
pops = [qt.expect(eop, test_states) for eop in eops]
qutip_pops = [qt.expect(eop, qutip_states) for eop in eops]
```

```python {.marimo}
mag_tlist = np.linspace(
    *test_magnus.tlims, magnus_param_form.value["num_magnus_intervals"]
)
```

```python {.marimo}
qutip_states = qt.sesolve(
    H=test_system.get_qutip_tdham(test_magnus.tlist),
    psi0=allzero_state,
    tlist=test_magnus.tlist,
    options={"store_states": True},
).states
```

```python {.marimo}
def func2opt(x, lgm: float = 1e3):
    """
    lgm: lagrange multiplier
    """
    return state_transfer_infidelity(
        pulse_coeffs=x,
        **(
            magnus_param_form.value
            | {
                "device": "gpu",
                # "sparse": True,
                "pulse_freq": 0,
            }
        ),
    )  # + lgm*np.linalg.norm(x, ord=1)
```

```python {.marimo}
rng = np.random.default_rng()
```

```python {.marimo}
with mo.status.spinner(title="Optimizing") as _spinner:
    for i in range(10):
        test_x = rng.random(size=10)

        def update_spinner(*, intermediate_result):
            _spinner.update(title=f"Optimizing attempt {i}/10:")
            _spinner.update(subtitle=f"Error = {intermediate_result.fun:.3e}")

        res = minimize(
            func2opt,
            test_x,
            method="L-BFgS-B",
            # method="COBYLA",
            # method="Nelder-Mead",
            bounds=[(-1, 1)] * len(test_x),
            callback=update_spinner,
            options={
                # "disp": True,
                # "gtol": 1e-12,
            },
        )
        if func2opt(res.x) < 1e-3:
            print("x: ", ", ".join(map(str, res.x)))
            print("Error: ", func2opt(res.x))
            break
```