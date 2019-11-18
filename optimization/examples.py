import numpy as np
from optfunction import OptFunction

def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])

def ex_f1(x):
    """
    Example function 1: the norm of a 2D vector.
    """
    return x.T.dot(x)

def ex_gradf1(x):
    """
    Gradient function 1
    """
    return 2 * x

ex1 = OptFunction(ex_f1, ex_gradf1)

def conditioned_quadratic(cond=50):
    """
    Produces a quadratic form which has a specified condition number. The
    quadratic form has some random variation and a random rotation.
    """
    A = np.array([[cond, 0], [0, 1]])
    theta = np.random.rand() * 2 * np.pi
    forward_rot = rotation_matrix(theta)
    backward_rot = rotation_matrix(-theta)
    A = backward_rot.dot(A.dot(forward_rot))

    def f(x):
        return x.T.dot(A.dot(x))

    def gradf(x):
        return (A + A.T).dot(x)

    def hessf(x, y):
        return A + A.T

    return OptFunction(f, gradf, hessf)

def ex_f2(x):
    return np.log(x[0]) * np.log(x[0]) + x[1] * x[1]

def ex_gradf2(x):
    dx = 2 * np.log(x[0]) / x[0]
    dy = 2 * x[0]
    return np.array([dx, dy])

def ex_hessf2(x):
    dxdx = (2. - 2 * np.log(x[0])) / x[0] / x[0]
    dydy = 2.
    return np.array([[dxdx, 0.], [0., dydy]])

ex2 = OptFunction(ex_f2, ex_gradf2, ex_hessf2)
