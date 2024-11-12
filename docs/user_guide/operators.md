# Operators Reference

In the interest of minimal code duplication, {math}`{\rm qCH_{\rm eff}}` defines a convenience wrapper class around NumPy/SciPy(sparse)/CuPy arrays. This provides a uniform interface to access some commonly used operations on these matrices. For [iterative SWT](iswt) methods, it is necessary to separate the diagonal and off-diagonal values in a matrix since they correspond to energies and couplings respectively, so `qcheff.operator` provides a uniform interface to these operations that dispatch to the appropriate underlying array module. This wrapper is for convenience of the user only. The NPAD subroutines work directly with the underlying array.

We define an abstract base class 

```{autodoc2-object} qcheff.operators.operator_base.OperatorMatrix
    :members:
    :undoc-members:
    :show-inheritance:
```



## Basic Operators

We have defined some basic operators in `qcheff.operators.operators`. The set of operators is a subset of all the operators in QuTiP.


```{autodoc2-object} qcheff.operators.operators
    :members:
    :undoc-members:
    :show-inheritance:
```