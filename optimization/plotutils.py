import numpy as np
import matplotlib.pyplot as plt

class PlaneIterationPlotter:

    def __init__(self, f, **kwargs):
        self.f = f
        self.kwargs = kwargs
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10,10))

        self._configure_kwargs()

    def _configure_kwargs(self):
        self.kwargs['xlim'] = (-1.,1.) if self.kwargs.get('xlim') is None else self.kwargs['xlim']
        self.kwargs['ylim'] = (-1.,1.) if self.kwargs.get('ylim') is None else self.kwargs['ylim']
        self.kwargs['steps'] = 50

        self.ax.set_xlim(self.kwargs['xlim'])
        self.ax.set_ylim(self.kwargs['ylim'])
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.set_aspect('equal')


    def plot_contours(self, *args, **kwargs):
        x, y = np.linspace(*self.kwargs['xlim']), np.linspace(*self.kwargs['ylim'])
        X, Y = np.meshgrid(x, y)

        Z = np.array([
            [self.f(np.array([X[i,j], Y[i,j]])) for j in range(self.kwargs['steps'])]
            for i in range(self.kwargs['steps'])
        ]).reshape(X.shape)

        self.ax.contour(X, Y, Z, *args, **kwargs)

    def plot_points(self, x, *args, **kwargs):
        self.ax.plot(x[:,0], x[:,1], *args, **kwargs)

    def show(self, *args, **kwargs):
        self.fig.show(*args, **kwargs)

    def save(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)
