import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# Our code:
from fem_2d_solver import Poisson2DSolver, matprint


def task_2_e(N=200):
    # Function conforming that the full Stiffness Matrix is singular.
    FEM_dummy_solver = Poisson2DSolver(N, True, True, True, True)
    FEM_dummy_solver.generate_A_h()
    A_h = FEM_dummy_solver.A_h.toarray()
    eigvals = la.eigvals(A_h)
    if any(np.abs(eigvals) < 1.0e-15):
        print("The matrix A_h, constructed without imposing any boundary conditions, is Singular.")


def display_analytical_solution(N=1000, u=None):
    if u is None:    
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


def test_big_number_FEM_solution(N=2000):
    
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
    FEM_solver.solve_big_number_dirichlet()
    print("Showing Big Number Dirichlet solution:")
    FEM_solver.display_solution()


def test_direct_FEM_solution(N=1000):

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
    FEM_solver.solve_direct_dirichlet()  # Might expand 'solve()' method to perform the above function calls.
    min_uh = min(FEM_solver.u_h)
    max_uh = max(FEM_solver.u_h)

    print("Showing Direct Dirichlet solution:")

    print(f"Minimum value: {min_uh:.5e}\nMax value: {max_uh:.5e}")

    # FEM_solver.display_mesh()
    FEM_solver.display_solution()


def compare_analytical_numerical(N=20):

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

    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC)

    FEM_solver.display_mesh()

    FEM_solver.solve_direct_dirichlet()
    FEM_solver.display_solution()

    def u(p):
        """
        Analytical solution to the Poisson-problem, with homogeneous dirichlet BCs
        and the source term f(p).
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        return np.sin(2*np.pi*r_squared)

    exact_solver = Poisson2DSolver(N, 0.0, 0.0, 0.0, 0.0)
    u_exact = np.array([u(p) for p in exact_solver.nodes])
    exact_solver.display_solution(u_exact)

def test_simpler_solution(N=1000):

    def u_simple(p):
        """
        Analytical solution to the Poisson-problem, with homogeneous dirichlet BCs
        and the source term f(p).
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        return np.sin(np.pi*r_squared)

    def f_simple(p):
        """
        Source function f(r, theta) = −8π*cos(2πr²)+ 16π²r²sin(2πr²)
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        term_1 = -4.0*np.pi*np.cos(np.pi*r_squared)
        term_2 = 4*np.pi**2*r_squared*np.sin(np.pi*r_squared)
        return term_1 + term_2
    
    def g_D(p):
        return 0.0
    
    def class_BC(p):
        """
        Classify all edge nodes as Dirichlet
        """
        return 1
    
    FEM_solver = Poisson2DSolver(N=N, f=f_simple, g_D=g_D, g_N=None, class_BC=class_BC)
    FEM_solver.solve_direct_dirichlet()  # Might expand 'solve()' method to perform the above function calls.
    min_uh = min(FEM_solver.u_h)
    max_uh = max(FEM_solver.u_h)

    print("Showing Simple Direct Dirichlet solution:")

    print(f"Minimum value: {min_uh:.5e}\nMax value: {max_uh:.5e}")

    # FEM_solver.display_mesh()
    FEM_solver.display_solution()

    print("Showing analytical solution:")
    display_analytical_solution(N=N, u=u_simple)


if __name__ == "__main__":
    # task_2_e()
    # test_direct_FEM_solution(N=5000)
    # test_big_number_FEM_solution(N=200)
    # test_big_number_FEM_solution()
    # compare_analytical_numerical()
    test_simpler_solution(N=1000)
    pass