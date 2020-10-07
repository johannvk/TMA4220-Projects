import numpy as np 
import scipy.integrate as sciint

from fem_2d_solver import Poisson2DSolver

from gaussian_quad import quadrature2D

def generate_triangle_jacobian(self, p1, p2, p3):
    """
    Function to generate the Jacobian J = ∂(x,y)/∂(r, s)\n
    for transforming from the reference triangle to global coordinates.\n
    element: The target element (triangle) of the transformation from the reference element.
    """
    J = np.column_stack([p2-p1, p3-p1])
    return J

def test_triangle_integral():
    
    # Source Function:
    def f(p):
        """
        Source function f(r, theta) = −8π*cos(2πr²)+ 16π²r²sin(2πr²)
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        term_1 = -8.0*np.pi*np.cos(2*np.pi*r_squared)
        term_2 = 16*np.pi**2*r_squared*np.sin(2*np.pi*r_squared)
        return term_1 + term_2
    
    x0, y0 = 0.1, 0.5
    a = 0.02
    b = 0.06

    # Scipy-integration:
    x_start, x_end = x0, x0 + a
    y_start = lambda x: y0
    y_end = lambda x: -(b/a)*(x - x0) + (y0 + b)

    integrand = lambda y, x: f([x, y])

    sci_int = sciint.dblquad(integrand, x_start, x_end, y_start, y_end, )

    # Vertex integration:
    p1 = np.array([x0, y0])
    p2 = np.array([x0 + a, y0])
    p3 = np.array([x0, y0 + b])

    quad_int = quadrature2D(f, p1, p2, p3, Nq=4)
    print("Tull!")
    # Make a test solver:
    # N = 100
    # test_solver = Poisson2DSolver(N=)

test_triangle_integral()

    