import numpy as np
from optfunction import OptFunction


def backtracking_line_search(optfun, x, direction, alpha=0.35, beta=0.95):
    """
    Given a descent direction `direction` for the function `optfun` at `x` in
    the domain of `optfun`, and given parameters `alpha` and `beta`, return a
    step size `t` satisfying the backtracking line search inequality.
    """
    t = 1.
    current_val, current_grad = optfun(x, order=1)
    while optfun(x + t * direction, order=0) > current_val + alpha * t * np.dot(current_grad.T, direction):
        t *= beta
    return t

    


def gradient_descent(optfun, x0, iterations=100, tolerance=1e-5):
    """
    Given an OptFunction `optfun` and a starting point `x0`, and given
    parameters `iterations` and `tolerance`, perform the gradient descent
    algorithm using backtracking line search to minimize `optfun`.

    This implementation should return a pair of two arrays:
    
      * `points`: an array of points of the same shape as x0 which corresponds
                  to the iterations of the gradient descent algorithm
      * `ferrors`: an array of "function errors" of the gradient descent algorithm. In
                  this context, the "error" is the magnitude difference between
                  two consecutive points in the iteration (i.e., |f(x_k) - f(x_{k-1})|).
      * `xerrors`: an array of "point errors" of the gradient descent algorithm. In
                  this context, the "error" is the magnitude difference between
                  two consecutive points in the iteration (i.e., ||x_k - x_{k-1}||).
    """
    points = np.empty((iterations + 1, *x0.shape))
    ferrors = np.empty(iterations)
    xerrors = np.empty(iterations)

    points[0] = x0
    for step in range(iterations):
        value, direction = optfun(points[step], order=1)
        stepsize = backtracking_line_search(optfun, points[step], direction)
        points[step+1] = points[step] + stepsize * direction
        xerrors[step] = np.linalg.norm(points[step+1] - points[step])
        ferrors[step] = np.abs(value - optfun(points[step+1], order=0))

        if np.min(xerrors[step], ferrors[step]) < tolerance:
            break

    return points[:step+1], ferrors[:step], xerrors[:step]



