import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent
from examples import ex1, ex2, conditioned_quadratic
from plotutils import PlaneIterationPlotter
import seaborn as sns
sns.set_context("talk")

fun = conditioned_quadratic(cond=10)

starting_point = np.ones((2,1)) / np.sqrt(2)

condition_numbers = [1, 2, 4, 8, 16, 32, 64, 128]

for number in condition_numbers:
    fun = conditioned_quadratic(cond=number)

    plotter = PlaneIterationPlotter(fun, xlim=(-1,1), ylim=(-1,1))
    descents, _, _ = gradient_descent(fun, starting_point, tolerance=1e-10)
    
    plotter.plot_contours(colors=plt.cm.Greys(np.linspace(0,5)))
    plotter.plot_points(descents, 'ro')
    plotter.ax.set_title(f'Gradient Descent on Quadratic Form with $\kappa={number}$')
    plotter.save(f'figures/gd-condition-number-{number}.png', bbox_inches='tight')

