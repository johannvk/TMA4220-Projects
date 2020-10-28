import numpy as np
import scipy.linalg as la
import scipy.integrate as sciint

gaussquad1d_points_weights = {
        1: (np.array([0.0]), np.array([2.0])),
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
    Can be in R², but then g must accept g([x, y]).
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
    return (la.norm(b - a)/2.0)*sum(w*integrand(eta) for eta, w in zip(points, weights))


def triangle_area(p1, p2, p3):
    t1 = p1[0]*(p2[1] - p3[1])
    t2 = p2[0]*(p3[1] - p1[1])
    t3 = p3[0]*(p1[1] - p2[1])
    return 0.5*abs(t1 + t2 + t3)


def area_coord_inv_transform(eta, p1, p2, p3):
    return eta[0]*p1 + eta[1]*p2 + eta[2]*p3


def quadrature2D(g, p1, p2, p3, Nq=3, *args):
    """
    g a function (x, y) -> R, accepting vector-argument p=[x, y]. 
    g([x, y]) will be evaluated.
    """
    # Only supported choice of quadrature points:
    # assert(Nq in (1, 3, 4))

    integrand = lambda eta: g(area_coord_inv_transform(eta, p1, p2, p3), *args)
    points, weights = gaussquad2d_points_weights[Nq]

    I = sum(w*integrand(eta) for eta, w in zip(points, weights))
    return triangle_area(p1, p2, p3)*I
