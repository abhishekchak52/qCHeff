---
title: Spin Magnus Rwa
marimo-version: 0.8.11
width: medium
---

```python {.marimo hide_code="true"}
pulse_params_form = mo.ui.batch(
    mo.md("""

# Magnus vs RWA comparison (single spin)
    
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

```python {.marimo}
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
with sns.plotting_context("talk"), mpl.rc_context({"mathtext.fontset": "cm"}):
    _pop_fig, _pop_ax = plt.subplots(layout="constrained")
    _pop_ax.plot(
        test_tlist,
        np.asarray(qtres_rwa.expect).T,
        label=["x_RWA", "y_RWA", "z_RWA"],
    )
    _pop_ax.plot(
        test_tlist,
        np.asarray(qtres_full.expect).T,
        label=["x_full", "y_full", "z_full"],
    )
    _pop_ax.legend()
    _pop_ax.set(
        xlim=(0, pulse_params_form.value["tg"]),
        ylim=(-1.1, 1.1),
        xlabel="Time (ns)",
        ylabel=r"$\langle \sigma \rangle$",
    )

_pop_fig
```

```python {.marimo}
with sns.plotting_context("talk"), mpl.rc_context({"mathtext.fontset": "cm"}):
    bloch_fig, bloch_ax = plt.subplots(
        layout="constrained", figsize=(4, 4), subplot_kw={"projection": "3d"}
    )
    b = qt.Bloch(fig=bloch_fig, axes=bloch_ax)
    b.sphere_color = "white"
    b.sphere_alpha = 0.01
    b.frame_alpha = 0.2
    b.zlpos = [1.15, -1.3]
    b.xlpos = [1.45, -1.3]
    b.render()
    bloch_ax.set(xlim3d=(-0.7, 0.7), ylim3d=(-0.7, 0.7), zlim3d=(-0.65, 0.7))
    rwa_trajectory = bloch_ax.plot(
        *magnus_RWA_expect, color="crimson", lw=5, label="RWA", alpha=0.8
    )[0]
    full_trajectory = bloch_ax.plot(
        *magnus_full_expect, lw=5, color="#76B908", label="Full", alpha=0.8
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
        handles=[full_trajectory, rwa_trajectory, magnus_trajectory],
        loc="upper left",
        ncols=1,
        fontsize=12,
        frameon=False,
    )
bloch_fig
```

```python {.marimo}
bloch_fig.savefig("magnus_rwa.pdf")
```

```python {.marimo column="1" name="setup"}
import marimo as mo
import qutip as qt
import matplotlib as mpl
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
# Initialization code that runs before all other cells
```

```python {.marimo}
from qcheff.utils.system import QuTiPSystem
from qcheff.utils.pulses import FourierPulse
```

```python {.marimo}
test_tlist = np.linspace(0, pulse_params_form.value["tg"], 5000)
```

```python {.marimo}
test_coeffs = [1, 0, 0.1, 0]
```

```python {.marimo}
pauli_eops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
```

```python {.marimo}
def magnus_evolve_single_spin_system(creation_func, num_intervals):
    system = creation_func(drive_coeffs=test_coeffs, gate_time=max(test_tlist))
    magnus = system.get_magnus_system(tlist=test_tlist)
    psi0 = qt.basis(2, 0)[:]
    states = magnus.evolve(init_state=psi0, num_intervals=num_intervals)
    return list(map(qt.Qobj, states))
```

```python {.marimo}
magnus_full_states = magnus_evolve_single_spin_system(
    create_single_spin_full_system, num_intervals=5000
)
magnus_RWA_states = magnus_evolve_single_spin_system(
    create_single_spin_RWA_system, num_intervals=5000
)
magnus_states = magnus_evolve_single_spin_system(
    create_single_spin_full_system,
    num_intervals=pulse_params_form.value["num_intervals"],
)
magnus_full_expect = qt.expect(pauli_eops, magnus_full_states)
magnus_RWA_expect = qt.expect(pauli_eops, magnus_RWA_states)
magnus_expect = qt.expect(pauli_eops, magnus_states)
```

```python {.marimo}
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