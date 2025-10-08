# Iterative Schreiffer-Wolff Transformation (ISWT)

:::{admonition} Expert Technique
The techniques discussed below require careful consideration of the symmetry of the quantum system under consideration. 
:::

This submodule contains the NPAD technique, which is an exact SWT at the level of couplings. We provide a general `ExactIterativeSWT` class interface. This interface provides the following methods:

- `givens_rotation_matrix()`: Create the Given's rotation matrix.
- `unitary_transformation()`: {math}`UHU^\dagger` for NPAD. May be generalized to time-dependent case.
- `eliminate_coupling()`: Eliminate a single coupling.
- `eliminate_couplings()`: Eliminate multiple couplings simultaneously.
- `largest_couplings()`: Obtain the largest couplings.

There are two subclasses implementing NPAD: `NPADScipySparse` and `NPADCupySparse`, owing to differences between the SciPy and CuPy sparse matrix/array API. They each implement a `givens_rotation_matrix()` method using the appropriate sparse matrix module, since these matrices are largely sparse ({math}`N+2` nonzero elements for {math}`N\times N` matrices). For the user, an `NPAD(H)` function is provided for convenience. This will return an object of the appropriate subclass based on the input array `H`.