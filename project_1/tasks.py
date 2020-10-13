import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# Our code:
from fem_2d_solver import Poisson2DSolver, matprint, BCtype
from gaussian_quad import quadrature2D


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


def task_2_e(N=1000):
    # Function conforming that the full Stiffness Matrix is singular.
    FEM_dummy_solver = Poisson2DSolver(N, True, True, True, True)
    FEM_dummy_solver.generate_A_h()
    A_h = FEM_dummy_solver.A_h.toarray()
    eigvals = la.eigvals(A_h)
    if any(np.abs(eigvals) < 1.0e-14):
        print("\nTask 2 e):\n\tThe matrix A_h, constructed without imposing any boundary conditions, is Singular.")


def big_number_dirichlet_FEM_solution(N=1000):
    
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

    print("Solving Big Number Dirichlet solution:")
    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC, eps=1.0e-2)
    FEM_solver.solve_big_number_dirichlet()
    print("Showing Big Number Dirichlet solution:")
    FEM_solver.display_solution()


def direct_dirichlet_FEM_solution(N=1000):

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

    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC, eps=1.0e-8)
    FEM_solver.solve_direct_dirichlet()
    min_uh = min(FEM_solver.u_h)
    max_uh = max(FEM_solver.u_h)

    print("Showing Direct Dirichlet solution:")

    print(f"Minimum value: {min_uh:.5e}\nMax value: {max_uh:.5e}")

    # FEM_solver.display_mesh()
    FEM_solver.display_solution()


def task_3(N=1000):
    print("\nTask 3: Neumann & Dirichlet Boundary Conditions")
 
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
    
    def g_N(p):
        r = np.sqrt(p[0]**2 + p[1]**2)
        return 4*np.pi*r*np.cos(2*np.pi*r**2)

    def class_BC(p):
        if p[1] <= 0.0:
            return BCtype.Dir
        else:
            return BCtype.Neu
    
    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=g_N, class_BC=class_BC)
    FEM_solver.solve()
    FEM_solver.display_solution()


def display_error_neumann_BC(N=500, u=None):

    if u is None:
        def u(p):
            """
            Analytical solution to the Poisson-problem, with homogeneous dirichlet BCs
            and the source term f(p).
            p: np.array([x, y])
            """
            r_squared = p[0]**2 + p[1]**2
            return np.sin(2*np.pi*r_squared)

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
    
    def g_N(p):
        r = np.sqrt(p[0]**2 + p[1]**2)
        return 4*np.pi*r*np.cos(2*np.pi*r**2)

    def class_BC(p):
        if p[1] <= 0.0:
            return BCtype.Dir
        else:
            return BCtype.Neu

    print("\nCalculating numerical solution:")
    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=g_N, class_BC=class_BC)

    print("Displaying Numerical solution:")
    FEM_solver.solve()
    FEM_solver.display_solution()
    
    u_exact = np.array([u(p) for p in FEM_solver.nodes])
    error_coeffs = u_exact - FEM_solver.u_h
    FEM_solver.display_solution(error_coeffs)


def test_error():

    def u_ex(p):
        return np.sin(2*np.pi * (p[0]**2 + p[1]**2))

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
        return BCtype.Dir

    Es = []
    Ns = [20, 40, 80, 160, 320, 640, 1280]
    for N in Ns:

        FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC, eps=1.0e-14)
        FEM_solver.solve_direct_dirichlet()

        e = FEM_solver.error_est(u_ex)

        Es.append(e)

    Es = np.array(Es, dtype=float)
    Ns = np.array(Ns, dtype=float)
    plt.loglog(Ns, Es, 'k-', label=r"$||u - u_h||_{L_2(\Omega)}$")
    plt.xlabel("Degrees of freedom")
    #plt.ylabel(r"$||u - u_h||_{L_2(\Omega)}$")
    plt.legend()

    def beta(x, y):
        '''
            Estimator for the coefficient of beta in linear regression model
                y = alpha + beta * x
        '''
        n = x.shape[0]
        beta = np.sum( (x - np.mean(x)) * (y - np.mean(y))) / np.sum( (x - np.mean(x))**2 )
        return beta

    beta = beta(np.log(Ns), np.log(Es))
    print(f'beta = {beta}')

    plt.show()


def test_error_neumann():

    def u_ex(p):
        return np.sin(2*np.pi * (p[0]**2 + p[1]**2))

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

    def g_N(p):
        r = np.sqrt(p[0]**2 + p[1]**2)
        return 4*np.pi*r*np.cos(2*np.pi*r**2)
    
    def class_BC(p):
        if p[1] <= 0.0:
            return BCtype.Dir
        else:
            return BCtype.Neu

    Es = []
    Ns = [20, 40, 80, 160, 320, 640, 1280]
    for N in Ns:

        FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=g_N, class_BC=class_BC, eps=1.0e-14)
        FEM_solver.solve_direct_dirichlet()

        e = FEM_solver.error_est(u_ex)

        Es.append(e)

    norm_u = np.pi / 2

    Es = np.array(Es, dtype=float)
    Es_rel = Es / norm_u
    Ns = np.array(Ns, dtype=float)
    plt.loglog(Ns, Es_rel, 'k-', label=r"$||u - u_h||_{L_2(\Omega)}$")
    plt.xlabel("Degrees of freedom")
    plt.title("Relative error, Neumann")
    plt.legend()

    def beta(x, y):
        '''
            Estimator for the coefficient of beta in linear regression model
                y = alpha + beta * x
        '''
        n = x.shape[0]
        beta = np.sum( (x - np.mean(x)) * (y - np.mean(y))) / np.sum( (x - np.mean(x))**2 )
        return beta

    beta = beta(np.log(Ns), np.log(Es_rel))
    print(f'beta = {beta}')

    plt.show()


if __name__ == "__main__":
    task_2_e()
    task_3(N=1000)

    # big_number_dirichlet_FEM_solution()
    # direct_dirichlet_FEM_solution()
    # display_error_neumann_BC()
    # display_analytical_solution(N=1000)
    