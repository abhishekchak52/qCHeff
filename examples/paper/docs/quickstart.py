import marimo

__generated_with = "0.16.5"
app = marimo.App()

with app.setup(hide_code=True):
    import qutip as qt
    import numpy as np
    import cupy as cp
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    from qcheff import qcheff_config, temp_config

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Operators""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Operators in $\text{qCH}_\text{eff}$

    We provide easy access to some operators commonly-used in quantum physics simulations: `eye()`, `create()`, `destroy()`, `number()`, `identity()`, `basis()`, `projector()`, `charge()`, `position()`, `momentum()`, `sigmax()`, `sigmay()`, `sigmaz()`, `sigmap()`, `sigmam()`. Please refer to the [operators documentation](https://qcheff.readthedocs.io/en/latest/apidocs/qcheff/qcheff.operators.operators.html) for more details. For example, `eye()` creates an identity operator:
    """
    )
    return


@app.cell
def _():
    from qcheff.operators import eye

    eye_op_cpu = eye(3)
    eye_op_cpu
    return (eye,)


@app.cell(hide_code=True)
def _():
    mo.md(
        rf"""By default, operators are created as sparse matrices on the CPU. We can also create operators on the GPU and use dense operators on either device."""
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Backend options can be set globally using the `qcheff_config` object. We can list the default parameters for the default config by printing it out."""
    )
    return


@app.cell
def _():
    qcheff_config
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Backend options can be changed locally or for a single object using the `temp_config()` context manager. Below we create both a sparse and dense identity operator on the GPU."""
    )
    return


@app.cell
def _(eye):
    with temp_config(backend="gpu"):
        eye_op_gpu = eye(3)

    eye_op_gpu
    return


@app.cell
def _(eye):
    with temp_config(backend="gpu", sparse=False):
        eye_op_gpu_dense = eye(3)

    eye_op_gpu_dense
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Creating operators from Numpy/CuPy arrays or `Qobj`. 

    To convert Numpy/CuPy arrays into a form that works with $\text{qCH}_\text{eff}$, we can directly create either a `SparseOperator` or `DenseOperator`, which will automatically use the correct backend.
    """
    )
    return


@app.cell
def _():
    from qcheff.operators import SparseOperator, DenseOperator
    return (DenseOperator,)


@app.cell
def _(DenseOperator):
    cp_eye = cp.eye(4)
    DenseOperator(cp_eye)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""QuTiP's `Qobj` can also be converted to either format by passing the underlying data array to `SparseOperator` or `DenseOperator`."""
    )
    return


@app.cell
def _(DenseOperator):
    DenseOperator(qt.qeye(3)[:])
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""These operators can then be used as inputs to methods specified in $\text{qCH}_\text{eff}$."""
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Iterative Schrieffer-Wolff Transformation (ISWT)

    We provide a numerical implementation of [NPAD](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.030313). If the structure and symmetries of the Hilbert space are known to be block-diagonalizable, NPAD can efficiently calculate a specific subset of eigenvalues without resorting to full numerical diagonalization.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    Let us consider an example of two coupled Kerr oscillators. We will create a test system using `scqubits`, with the Hamiltonian $H=H_0 + V$, where

    $$ H_0 = \sum_{i=1, 2} E_i a_i^\dag a_i - K_i a_i^\dag a_i^\dag a_ia_i$$

    $$ V = g (a_1 a^\dag_2 + a^\dag_1 a_2)$$
    """
    )
    return


@app.cell
def _():
    import scqubits as scq


    q1 = scq.KerrOscillator(E_osc=10.0, K=5.3, truncated_dim=3)
    q2 = scq.KerrOscillator(E_osc=12.3, K=3.7, truncated_dim=3)

    hs = scq.HilbertSpace([q1, q2])
    hs.add_interaction(
        g=0.5,
        op1=q1.creation_operator,
        op2=q2.annihilation_operator,
        add_hc=True,
    )
    return (hs,)


@app.cell
def _(hs):
    ham = hs.hamiltonian()
    ham
    return (ham,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""Let's now use NPAD to eliminate the smallest coupling. We can do this after initializing the `NPAD` object and then using the `eliminate_coupling()` method. This method will perform a Givens rotation in the specified two-level subspace. We can see below that the coupling has been eliminated and the energies of the coupled level (on the diagonal) are changed.""")
    return


@app.cell
def _(DenseOperator, ham):
    from qcheff.iswt import NPAD

    # Instantiate the NPAD object
    zz_npad = NPAD(DenseOperator(ham[:]))

    # Eliminate one coupling
    zz_npad.eliminate_coupling(1, 3)
    qt.Qobj(zz_npad.H.op.toarray())
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Time evolution using the Magnus expansion

    We provide a direct first-order Magnus series-based time evolution in $\text{qCH}_\text{eff}$. We can use this method to simulate time-evolution for quantum systems
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Below we show how to use our Magnus-based time evolution for a single resonantly-driven spin. This system is described by the Hamiltonian

    $$H_{\rm lab} = \frac{\omega}{2}\sigma_{z}+\frac{\Omega(t)}{2}\cos(\omega t)\sigma_{x},$$

    The first term is called the drift Hamiltonian and the second term is the control/drive Hamiltonian. $\Omega(t)$ is the control pulse, modulatated at the qubit frequency $\omega$. For simplicity, we assume a flat-top pulse.
    """
    )
    return


@app.cell
def _(DenseOperator):
    from qcheff.operators import sigmaz, sigmax, basis

    # Qubit frequency
    omega = 2 * np.pi * 1
    drift_ham = DenseOperator(0.5 * omega * sigmaz())
    control_hams = [DenseOperator(sigmax())]

    # Initial State
    psi0 = DenseOperator(np.array([0, 1]))

    # Time list for creating the control signal
    tlist = np.linspace(0, 2 * np.pi * 2, 1000)
    control_sigs = 0.85 * np.cos(omega * tlist)
    return control_hams, control_sigs, drift_ham, psi0, tlist


@app.cell
def _(control_hams, control_sigs, drift_ham, tlist):
    from qcheff.magnus import magnus

    # Magnus object
    spin_magnus = magnus(tlist, drift_ham, control_sigs, control_hams)
    return (spin_magnus,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""This returns a `MagnusTimeEvolDense` object. We can call the `.evolve()` method specifying how many Magnus intervals we want to use for the integration."""
    )
    return


@app.cell
def _(psi0, spin_magnus):
    # time-evolved states (returned as generator)
    states = spin_magnus.evolve(init_state=psi0, num_intervals=100)

    # Sigma Z expectation values.
    popz = [qt.expect(qt.sigmaz(), qt.Qobj(state)) for state in states]
    return (popz,)


@app.cell
def _(popz, tlist):
    with (
        sns.plotting_context("talk"),
        sns.axes_style("ticks"),
        matplotlib.rc_context({"mathtext.fontset": "cm"}),
    ):
        fig, ax = plt.subplots()

        ax.plot(tlist[::10], popz, "x-")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
        ax.set(
            ylim=(-1, 1),
            ylabel=r"$\langle \sigma_z\rangle$",
            xlabel="Time (ns)",
            xlim=tlist[[0, -1]],
        )

    fig
    return


if __name__ == "__main__":
    app.run()
