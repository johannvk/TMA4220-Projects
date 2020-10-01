import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# Our code:
from fem_2d_solver import Poisson2DSolver, matprint


def basic_tests():
    # Filler arguments:
    a = Poisson2DSolver(200, 0.0, 0.0, 0.0, 0.0)

    # Test transforms:
    element_i = 12
    eta_test = [0.62, 0.135]
    test_xy = a.reference_to_global_transformation(eta=eta_test, element=element_i)
    eta_result = a.global_to_reference_transformation(p=test_xy, element=element_i)
    # After applying the transform and its inverse for the same element, 
    # "eta_test" should be the same as "eta_result"
    assert(np.allclose(eta_test, eta_result))

    # node_to_hit = a.nodes[a.triang[element_i][1]]
    # a.display_mesh(nodes=4)
    # a.display_mesh()

    # Test surface plot:
    # u_h = np.zeros(a.num_nodes)
    # u_h[10] = 1.0
    # u_h[11] = -1.0
    u_h = np.arange(a.num_nodes)/(a.num_nodes)
    a.display_solution(u_h)


def test_A_i_j():
    num_nodes = 10
    a = Poisson2DSolver(num_nodes, 0.0, 0.0, 0.0, 0.0)
    print("nodes:\n", a.nodes)
    # a.display_mesh(nodes=len(a.nodes)-1)
    
    # test_elem = 0
    # J = a.generate_jacobian(test_elem)
    # J_inv = np.linalg.inv(J)
    # elem_area = np.linalg.det(J)*0.5
 
    # i, j = 1, 2
    # A_i_j = a.A_i_j(i, j, J_inv, elem_area)
    # print(f"A_{i}_{j} = {A_i_j:.6f}")

    a.generate_A_h()
    print("\nA_h:")
    A_h = a.A_h.toarray()
    matprint(A_h)
    col_sums = np.sum(A_h, axis=-1)
    print("col_sums:\n", col_sums)
    x_test = np.linalg.solve(A_h, np.random.randn(num_nodes))
    A_h_eigvals = la.eigvals(A_h) # One zero-eigenvalue. Singular.
    a.display_mesh()


def test_F_h():

    def f(p):
        """
        Source function f(r, theta) = −8π*cos(2πr²)+ 16π²r²sin(2πr²)
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        term_1 = -8.0*np.pi*np.cos(2*np.pi*r_squared)
        term_2 = 16*np.pi**2*r_squared*np.sin(2*np.pi*r_squared)
        return term_1 + term_2

    def u(p):
        """
        Analytical solution to the Poisson-problem, with homogeneous dirichlet BCs
        and the source term f(p).
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        return np.sin(2*np.pi*r_squared)

    N = 2000
    solver = Poisson2DSolver(N, f, 0.0, 0.0, 0.0)
    solver.generate_F_h()
    print(solver.F_h)
    solver.display_mesh()


def main():
    basic_tests()
    test_A_i_j()
    test_F_h()


if __name__ == "__main__":
    main()