import numpy as np 
import scipy.integrate as sciint

from ..project_1.fem_2d_solver import Poisson2DSolver

from gaussian_quad import quadrature1D, quadrature2D, \
     gaussquad1d_points_weights, gaussquad2d_points_weights


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


def test_quadrature1D(show_weights_points=False, Nq=4):
    # Testing functionality of 1D-Gaussian Quadrature:

    if show_weights_points: # Display Gaussian points and weights.
        for key, value in gaussquad1d_points_weights.items():
            print(f"Gauss Points: {key}\nPoints: {value[0]}\nWeights: {value[1]}\n")

    func = lambda x: np.exp(x)
    a, b = 1.0, 2.0
    I_quad = quadrature1D(func, a, b, Nq)

    I_exact = np.exp(b) - np.exp(a)

    print(f"\nExact answer: {I_exact}\
            \nGauss Quad. answer: {I_quad}\
            \nRelative Error: {abs((I_exact - I_quad)/I_exact):.6e}")


def test_vector1DGauss():
    a = np.array([2.0, 2.0])
    b = np.array([6.0, 4.0])

    f = lambda p: 3*p[0]**2 + 4*p[1] + 5

    result = quadrature1D(f, a, b, Nq=4)
    print("Result:", result)


def test_quadrature2D(show_weights_points=False):
    if show_weights_points: # Display Gaussian points and weights.
        for key, value in gaussquad2d_points_weights.items():
            print(f"Gauss Points: {key}\nPoints: {value[0]}\nWeights: {value[1]}\n")    
    
    def problem_2_test():
        print("\nTesting Problem 2. Integral of log(x+y):")
        func = lambda p: np.log(p[0] + p[1])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([3.0, 1.0])
        p3 = np.array([3.0, 2.0])

        I_exact = 1.165417026740377
        I_quad = quadrature2D(func, p1, p2, p3, Nq=4)

        # Ugly expression:
        # I_exact1 = 1.165417026740377
        # I_quad1 = quadrature2D(func, p1, p2, p3, Nq=4)

        print(f"\nExact answer: {I_exact}\
            \nGauss Quad. answer: {I_quad}\
            \nRelative Error: {abs((I_exact - I_quad)/I_exact):.6e}")

    def polynomial_triangle_test():
        print("\nTesting Polynomial integration: f(x,y) = 2x²-y:")
        a = 4.0 
        b = 12.0

        func = lambda p: 2*p[0]**2 - p[1]
        p1 = np.array([0.0, 0.0])
        p2 = np.array([a, 0.0])
        p3 = np.array([0.0, b])
        
        I_exact = (a*b/6.0)*(a**2 - b)
        I_quad = quadrature2D(func, p1, p2, p3, Nq=3)

        # I_quad = sciint.dblquad(lambda y, x: func([x, y]), 0.0, a, y_0_x, y_1_x)[0]

        print(f"\nExact answer: {I_exact}\
            \nGauss Quad. answer: {I_quad}\
            \nRelative Error: {abs((I_exact - I_quad)/I_exact):.6e}")

    problem_2_test()    
    polynomial_triangle_test()




if __name__ == "__main__":
    test_quadrature1D(show_weights_points=False, Nq=4)
    test_quadrature2D()
    # test_triangle_integral()
    test_vector1DGauss()
 
    