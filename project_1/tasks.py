import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# Our code:
from fem_2d_solver import Poisson2DSolver


def task_2_e():
    # Function conforming that the full Stiffness Matrix is singular.
    FEM_dummy_solver = Poisson2DSolver(100, True, True, True, True)
    FEM_dummy_solver.generate_A_h()
    A_h = FEM_dummy_solver.A_h.toarray()
    eigvals = la.eigvals(A_h)
    if any(np.abs(eigvals) < 1.0e-15):
        print("The matrix A_h, constructed and without imposing any boundary conditions, is Singular.")


def display_analytical_solution(N=1000):
    
    def u(p):
        """
        Analytical solution to the Poisson-problem, with homogeneous dirichlet BCs
        and the source term f(p).
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        return np.sin(2*np.pi*r_squared)

    solver = Poisson2DSolver(N, 0.0, 0.0, 0.0, 0.0)
    u_exact = np.array([u(p) for p in solver.nodes])
    solver.display_solution(u_exact)


def test_FEM_solution(N=2000):
    
    def f(p):
        """
        Source function f(r, theta) = −8π*cos(2πr²)+ 16π²r²sin(2πr²)
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        term_1 = -8.0*np.pi*np.cos(2*np.pi*r_squared)
        term_2 = 16*np.pi**2*r_squared*np.sin(2*np.pi*r_squared)
        return term_1 + term_2
    
    def g_D(p):
        return 0.0
    
    def class_BC(p):
        """
        Classify all edge nodes as Dirichlet
        """
        return 1
    
    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC, eps=1.0e-14)
    FEM_solver.solve_big_number_dirichlet()  # Might expand 'solve()' method to perform the above function calls.
    FEM_solver.display_solution()


if __name__ == "__main__":
    test_FEM_solution()