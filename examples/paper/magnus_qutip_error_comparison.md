---
title: Magnus Time Evolution
marimo-version: 0.8.11
width: medium
---

```python {.marimo name="setup"}
# Initialization code that runs before all other cells
import copy
import functools
import itertools

import marimo as mo
import more_itertools

import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as spsparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import qutip as qt
from scipy.optimize import minimize
```

```python {.marimo}
from qcheff.models.spin_chain.utils import (
    setup_magnus_chain_example,
    state_transfer_infidelity,
)

from qcheff.utils.system import QuTiPSystem
from qcheff.models.spin_chain.utils import embed_operator
from qcheff.utils.pulses import FourierPulse
from qcheff.operators import SparseOperator, DenseOperator
import qcheff.operators.operators as qops
```

```python {.marimo hide_code="true"}
magnus_param_form = mo.ui.batch(
    mo.md(
        r"""
        # Magnus Simulation Parameters

        Chain size $N$: {chain_size}

        Gate time $t$: {gate_time} / {num_tlist} points / {num_magnus_intervals} intervals.

        Qubit frequency $\omega_q$ : {wq} GHz

        NN Interaction strength $J$ : {J} GHz
        
        NNN Interaction strength $g$ : {g} GHz

        """
    ),
    {
        "wq": mo.ui.slider(
            start=5,
            stop=10,
            step=0.1,
            value=(5),
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
            stop=10,
            step=1,
            value=3,
        ),
        "gate_time": mo.ui.number(
            start=10,
            stop=100,
            step=1,
            value=25,
        ),
        "num_tlist": mo.ui.number(
            start=100,
            stop=10000,
            step=100,
            value=1000,
        ),
        "num_magnus_intervals": mo.ui.number(
            start=100,
            stop=10000,
            step=100,
            value=500,
        ),
    },
).form()
magnus_param_form
```

```python {.marimo}
# mo.stop(coeff_sel.value == "Optimized", "Optimization is turned on.")
manual_coeffs = [
    0.9025997070283809,
    -0.9999950428875883,
    -0.7361496380026948,
    -0.7132350060400646,
    -0.9143142420450023,
    1.0,
    1.0,
    -1.0,
]
chosen_coeffs = manual_coeffs
```

```python {.marimo}
test_mat = SparseOperator(spsparse.csr_array(qops.create(20))).op
```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to start")

test_system, test_magnus = setup_magnus_chain_example(
    pulse_coeffs=chosen_coeffs,
    **magnus_param_form.value,
)


```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to start")

interval_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
err_list = [
    state_transfer_infidelity(
        pulse_coeffs=chosen_coeffs,
        **(
            magnus_param_form.value
            | {"num_magnus_intervals": interval, "disable_progress_bar": True}
        ),
    )
    for interval in interval_list
]
```

```python {.marimo}
fig2, ax2 = plt.subplots()
ax2.loglog(interval_list, err_list, "o-", alpha=0.6, lw=3)
ax2.set(
    xlabel="Number of Magnus intervals",
    ylabel="Infidelity",
    xlim=tuple(more_itertools.minmax(interval_list)),
)
fig2
```

```python {.marimo}
def func2opt(x):
    return state_transfer_infidelity(pulse_coeffs=x, **magnus_param_form.value)
```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to start")
chain_size = int(magnus_param_form.value["chain_size"])
```

```python {.marimo}
mo.stop(not magnus_param_form.value, "Submit form to start")
# mo.stop(coeff_sel.value == "Manual", "Manual coeffs chosen.")
rng = np.random.default_rng()
test_x = rng.random(size=8)
with mo.status.spinner(title="Optimizing"):
    res = minimize(
        func2opt,
        test_x,
        method="L-BFgS-B",
        bounds=[(-1, 1)] * len(test_x),
        # options={'disp': True},
    )
    print(res)
    print("x: ", ", ".join(map(str, res.x)))
    print("Error: ", func2opt(res.x))
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
pops = [qt.expect(eop, test_states) for eop in eops]
```

```python {.marimo}
labels = [r"$P_0$", r"$P_1$", r"$P_{\rm rest}$"]
fig, ax = plt.subplots(2, 1, figsize=(6, 7), layout="constrained", sharex=True)
test_system.plot_control_signals(tlist=test_magnus.tlist, axis=ax[0])
mag_tlist = np.linspace(
    *test_magnus.tlims, magnus_param_form.value["num_magnus_intervals"]
)
for pop, label in zip(pops, labels):
    ax[1].plot(mag_tlist, pop, label=label)

ax[1].set(
    xlim=test_magnus.tlims,
    xlabel="Time (ns)",
    ylabel="Population",
    ylim=(0, 1),
)
ax[1].legend(loc="best")
infidelity = 1 - qt.expect(P1, test_states[-1])
fig.suptitle(f"Chain size: {chain_size}, Error: { infidelity:.3e}")
fig
```