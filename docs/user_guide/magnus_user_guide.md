# Magnus Series-based Time Evolution


The `MagnusTimeEvol` class provides a standard interface for all Magnus time-evolution operations. The interface includes two main functions:

- `update_control_sigs()`: Change the control pulse applied to the system.
- `evolve()`: Evolve the system for a given number of Magnus intervals.

There are two subclasses that implement this interface: `MagnusTimeEvolDense` and `MagnusTimeEvolSparse`. Notably, `MagnusTimeEvol` computes the propagators for all Magnus intervals simultaneously. Sparse arrays are limited to be 2-d in SciPy/CuPy, so we cannot use the same technique without converting to dense matrices. Furthermore, exponentiating a matrix uses a lot of memory, so the dense implementation is limited in the size of systems it can handle. Instead, we loop over the propagators for each Magnus interval. This lets the user examine larger systems at the expense of slow iteration in Python loop. At the time of writing, CuPy does not have an implementation of matrix exponentiation `expm()` that works with 3-d arrays, so we have implemented such a routine based on the Taylor expansion, inspired by [https://github.com/lezcano/expm](https://github.com/lezcano/expm) for use on the GPU.

For the user, a function `magnus(tlist, drift_ham, control_sigs, control_hams)` is provided for convenience. This will return an object of the appropriate subclass based on the input array.



