import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent
from examples import ex1, ex2, conditioned_quadratic
from plotutils import PlaneIterationPlotter

plt1 = PlaneIterationPlotter(ex1)
gd1 = gradient_descent(ex1, np.random.randn(1, 1))


plt2 = PlaneIterationPlotter(ex2)
gd2 = gradient_descent(ex2, np.random.randn(1, 1))


pltcond = [
    PlaneIterationPlotter(conditioned_quadratic(cond)) for cond in [10**k for k in range(4)]
]
gdcond = [
    gradient_descent(
        plt.f, np.random.randn(1, 1)
    ) for plt in pltcond

