---
title: Spin Magnus Rwa
marimo-version: 0.8.11
width: medium
---

```{.python.marimo}
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
import plotly.express as px
```

```{.python.marimo}
from qcheff.utils.system import QuTiPSystem
from qcheff.utils.pulses import FourierPulse
```

```{.python.marimo}
qtres_rwa = qt.sesolve(
    H=create_single_spin_RWA_system(
        drive_coeffs=test_coeffs, gate_time=max(test_tlist)
    ).get_qutip_tdham(tlist=test_tlist),
    psi0=qt.basis(2, 0),
    tlist=test_tlist,
    e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
)
qtres_full = qt.sesolve(
    H=create_single_spin_full_system(
        drive_coeffs=test_coeffs, gate_time=max(test_tlist)
    ).get_qutip_tdham(tlist=test_tlist),
    psi0=qt.basis(2, 0),
    tlist=test_tlist,
    e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
)
plt.plot(test_tlist, np.asarray(qtres_rwa.expect).T, label=["x", "y", "z"])
plt.legend()
```

```{.python.marimo}
magnus_full_states = list(
    map(
        qt.Qobj,
        create_single_spin_full_system(
            drive_coeffs=test_coeffs, gate_time=max(test_tlist)
        )
        .get_magnus_system(tlist=test_tlist)
        .evolve(init_state=qt.basis(2, 0)[:], num_intervals=5000),
    )
)
magnus_RWA_states = list(
    map(
        qt.Qobj,
        create_single_spin_RWA_system(
            drive_coeffs=test_coeffs, gate_time=max(test_tlist)
        )
        .get_magnus_system(tlist=test_tlist)
        .evolve(init_state=qt.basis(2, 0)[:], num_intervals=5000),
    )
)
magnus_states = list(
    map(
        qt.Qobj,
        create_single_spin_full_system(
            drive_coeffs=test_coeffs, gate_time=max(test_tlist)
        )
        .get_magnus_system(tlist=test_tlist)
        .evolve(
            init_state=qt.basis(2, 0)[:],
            num_intervals=pulse_params_form.value["num_intervals"],
        ),
    )
)
magnus_full_expect = qt.expect(
    [qt.sigmax(), qt.sigmay(), qt.sigmaz()], magnus_full_states
)
magnus_RWA_expect = qt.expect(
    [qt.sigmax(), qt.sigmay(), qt.sigmaz()], magnus_RWA_states
)
magnus_expect = qt.expect(
    [qt.sigmax(), qt.sigmay(), qt.sigmaz()], magnus_states
)
```

```{.python.marimo}
bloch_fig = plt.figure(layout="constrained", figsize=(6, 6))
bloch_ax = bloch_fig.add_subplot(projection="3d")
b = qt.Bloch(fig=bloch_fig, axes=bloch_ax)
b.sphere_color = "white"
b.sphere_alpha = 0.01
b.frame_alpha = 0.2
b.zlpos = [1.3, -1.3]
b.xlpos = [1.3, -1.3]
b.render()
bloch_ax.set(xlim3d=(-1, 1), ylim3d=(-1, 1), zlim3d=(-1, 1))
rwa_trajectory = bloch_ax.plot(
    *magnus_RWA_expect,
    color="crimson",
    lw=5,
    label="RWA",
    alpha=0.8,
)[0]
full_trajectory = bloch_ax.plot(
    *magnus_full_expect,
    lw=5,
    color="#76B908",
    label="Full",
    alpha=0.8,
)[0]
magnus_trajectory = bloch_ax.plot(
    *magnus_expect,
    color="black",
    lw=0,
    label="Magnus",
    alpha=1,
    marker="+",
    markevery=2,
    markersize=10,
)[0]
bloch_fig.legend(
    handles=[
        full_trajectory,
        rwa_trajectory,
        magnus_trajectory,
    ],
    loc="upper center",
    ncols=3,
    fontsize=15,
    frameon=False,
)
bloch_fig
```

```{.python.marimo}
test_tlist = np.linspace(0, pulse_params_form.value["tg"], 5000)
```

```{.python.marimo}
test_coeffs = [1, 0, 0.1, 0]
```

```{.python.marimo}
# fig, ax = plt.subplots()
# create_single_spin_full_system(drive_coeffs=test_coeffs, gate_time=max(test_tlist)).plot_control_signals(tlist=test_tlist, axis=ax)
# fig
```

```{.python.marimo}
pulse_params_form = mo.ui.batch(
    mo.md("""
Amplitude : {amp}

Gate time : {tg}

Number of Magnus intervals: {num_intervals}
"""),
    {
        "amp": mo.ui.number(0.01, 10, 0.01, value=0.33),
        "tg": mo.ui.number(0.1, 100, 0.01, value=2.5),
        "num_intervals": mo.ui.number(5, 5000, 5, value=50),
    },
).form()
pulse_params_form
```

```{.python.marimo}
def create_single_spin_RWA_system(
    drive_coeffs: list[float], gate_time: float = 10.0
):
    drift_ham = qt.Qobj(np.zeros((2, 2)))  # qubit frequency = 1
    control_sigs = [
        FourierPulse(
            coeffs=drive_coeffs,
            gate_time=gate_time,
            frequency=0,
            amplitude=pulse_params_form.value["amp"],
            name="drive",
        )
    ]
    control_hams = [qt.sigmax() * np.pi]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    return system


def create_single_spin_full_system(
    drive_coeffs: list[float], gate_time: float = 10.0
):
    drift_ham = qt.Qobj(np.zeros((2, 2)))  # qubit frequency = 1
    control_sigs = [
        # sigx term
        FourierPulse(
            coeffs=drive_coeffs,
            gate_time=gate_time,
            frequency=0,
            amplitude=pulse_params_form.value["amp"],
            name="sigx drive",
        ),
        # sigx *cos term
        FourierPulse(
            coeffs=drive_coeffs,
            gate_time=gate_time,
            frequency=2,
            amplitude=pulse_params_form.value["amp"],
            name=" sigx*cos drive",
        ),
        # sigy*sin term
        FourierPulse(
            coeffs=drive_coeffs,
            gate_time=gate_time,
            frequency=2,
            phase=np.pi / 2,
            amplitude=pulse_params_form.value["amp"],
            name=" sigy*sin drive",
        ),
    ]
    control_hams = [
        qt.sigmax() * np.pi,
        qt.sigmax() * np.pi,
        -qt.sigmay() * np.pi,
    ]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    return system
```

```{.python.marimo}

```