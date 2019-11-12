import numpy as np
from optfunction import OptFunction


def backtracking_line_search(optfun, x, direction, alpha=0.35, beta=0.95):
    """
    Given a descent direction `direction` for the function `optfun` at `x` in
    the domain of `optfun`, and given parameters `alpha` and `beta`, return a
    step size `t` satisfying the backtracking line search inequality.
    """
    raise NotImplemented("Backtracking Line Search needs implementation!")


def gradient_descent(optfun, x0, iterations=100, tolerance=1e-5):
    """
    Given an OptFunction `optfun` and a starting point `x0`, and given
    parameters `iterations` and `tolerance`, perform the gradient descent
    algorithm using backtracking line search to minimize `optfun`.

    This implementation should return a pair of two arrays:
    
      * `values`: an array of points of the same shape as x0 which corresponds
                  to the iterations of the gradient descent algorithm
      * `errors`: an array of "errors" of the gradient descent algorithm. In
                  this context, the "error" is the magnitude difference between
                  two consecutive points in the iteration (i.e., ||x_k -
                  x_{k-1}||).
    """
    return (np.array([[1,2],[3,4]]), np.array([1,2]))
    # raise NotImplemented("Gradient Descent Algorithm needs implementation!")

