class OptFunction:
    """
    Wrapper class for functions associated with optimization algorithms. The
    idea is to include the function, the gradient function, and/or the Hessian
    function inside of the same object. Then when the function is called, you
    can specify what "order" of call you want:

    Order: Return
    -------------
    0: function value
    1: function value, gradient value
    2: function value, gradient value, Hessian value

    This is useful for optimization algorithms when all of this information may
    be needed repeatedly.
    """

    def __init__(self, f, gradf=None, hessf=None):
        self.f = f
        self.gradf = gradf
        self.hessf = hessf
        self.order = int(self.gradf is None) + int(self.hessf is None)

    def __call__(self, x, order=0):
        print(order, self.order)
        if order > self.order:
            raise ValueError(f"Function does not support calls of order {order}")
        fx = self.f(x)
        if order == 0:
            return (fx,)
        else:
            gradfx = self.gradfx(x)
            if order == 1:
                return (fx, gradfx)
            else:
                hessfx = self.hessfx(x)
                return (fx, gradfx, hessfx)
        
