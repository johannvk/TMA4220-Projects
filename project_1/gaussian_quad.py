import numpy as np
import scipy.integrate as sciint

gaussquad1d_points_weights = {
        1: ((0.0), (2.0)),
        2: ((-np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)), 
            (1.0, 1.0)),
        3: ((-np.sqrt(3/5), 0.0, np.sqrt(3/5)), 
            (5.0/9.0, 8.0/9.0, 5.0/9.0)),
        4: ((-np.sqrt((3+2*np.sqrt(6.0/5.0))/7.0), -np.sqrt((3-2*np.sqrt(6.0/5.0))/7.0), 
              np.sqrt((3-2*np.sqrt(6.0/5.0))/7.0),  np.sqrt((3+2*np.sqrt(6.0/5.0))/7.0)), 
            ((18 - np.sqrt(30))/36.0, (18 + np.sqrt(30))/36.0, 
             (18 + np.sqrt(30))/36.0, (18 - np.sqrt(30))/36.0))
}

gaussquad2d_points_weights = {
    1: ([np.ones(3)/3.0], np.array([1.0])),
    3: ((np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5])), 
        (1/3.0, 1/3.0, 1/3.0)),
    4: ((np.array([1/3.0, 1/3.0, 1/3.0]), np.array([3/5.0, 1/5.0, 1/5.0]), 
         np.array([1/5.0, 3/5.0, 1/5.0]), np.array([1/5.0, 1/5.0, 3/5.0])), 
        (-9/16.0, 25/48.0, 25/48.0, 25/48.0))
}


def quadrature1D(g, a, b, Nq=3, *args):
    """
    g: Integrand.
    a: Start of integration interval.
    b: End of integration interval.
    Nq: Number of integration points.
        Integrates polynomials upto order 
        2*Nq - 1 exactly.
    *args: Extra parameters to pass to g(x, *args).
    """
    # Coordinate transform from eta in [-1, 1] to x in [a, b]:    
    def inv_transform(eta):
        return 0.5*((b-a)*eta + (b + a))

    integrand = lambda eta: g(inv_transform(eta), *args)

    # Retrieve appropriate points and weights:
    points, weights = gaussquad1d_points_weights[Nq]

    # Perform summation and scale up with Jacobian from transforming the integral:    
    return ((b - a)/2.0)*sum(w*integrand(eta) for eta, w in zip(points, weights))


def triangle_area(p1, p2, p3):
    t1 = p1[0]*(p2[1] - p3[1])
    t2 = p2[0]*(p3[1] - p1[1])
    t3 = p3[0]*(p1[1] - p2[1])
    return 0.5*abs(t1 + t2 + t3)

def area_coord_inv_transform(etas, p1, p2, p3):
    return etas[0]*p1 + etas[1]*p2 + etas[2]*p3


def quadrature2D(g, p1, p2, p3, Nq=3, *args):
    """
    g a function (x, y) -> R, accepting vector-argument p=[x, y]. 
    g([x, y]) will be evaluated.
    """
    # Only supported choice of quadrature points:
    # assert(Nq in (1, 3, 4))

    integrand = lambda etas: g(area_coord_inv_transform(etas, p1, p2, p3), *args)
    points, weights = gaussquad2d_points_weights[Nq]

    I = sum(w*integrand(etas) for etas, w in zip(points, weights))
    return triangle_area(p1, p2, p3)*I


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


def test_quadrature1D(show_weights_points=False):
    # Testing functionality of 1D-Gaussian Quadrature:

    if show_weights_points: # Display Gaussian points and weights.
        for key, value in gaussquad1d_points_weights.items():
            print(f"Gauss Points: {key}\nPoints: {value[0]}\nWeights: {value[1]}\n")

    func = lambda x: np.exp(x)
    a, b = 1.0, 2.0
    N_q = 4
    I_quad = quadrature1D(func, a, b, N_q)

    I_exact = np.exp(b) - np.exp(a)

    print(f"\nExact answer: {I_exact}\
            \nGauss Quad. answer: {I_quad}\
            \nRelative Error: {abs((I_exact - I_quad)/I_exact):.6e}")


if __name__ == "__main__":
    # test_quadrature1D()
    test_quadrature2D()

 