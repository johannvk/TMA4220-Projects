import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# Our code:
from .fem_2d_solver import Poisson2DSolver 
from ..tools import quadrature1D, quadrature2D, \
                          BCtype, matprint
from .tasks import display_analytical_solution


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
    # x_test = np.linalg.solve(A_h, np.random.randn(num_nodes))
    # A_h_eigvals = la.eigvals(A_h) # One zero-eigenvalue. Singular.
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


def integrate_source_func_over_triangle(f, k, i_loc, FEM_solver):
    # Testing integration of the source function over a triangle. 
    # In the Global Coordinates.
    glob_loc_coord = FEM_solver.global_to_reference_transformation

    def integrand(p):
        return f(p)*FEM_solver.basis_functions[i_loc](glob_loc_coord(p, k))

    p1, p2, p3 = FEM_solver.nodes[FEM_solver.triang[k]]

    I = quadrature2D(integrand, p1, p2, p3)
    return I


def big_polynomial_solution(N=1000):

    def f(p):
        return 4.0
    
    def u(p):
        return 1.0 - (p[0]**2 + p[1]**2)

    def class_BC(p):
        """
        Classify all edge nodes as Dirichlet
        """
        return 1
    
    # Jukser, setter presise Dirichlet-betingelser i hver kant-node:
    FEM_poly = Poisson2DSolver(N=N, f=f, g_D=u, g_N=None, class_BC=class_BC)

    FEM_poly.solve_direct_dirichlet()

    display_analytical_solution(N=N, u=u)

    FEM_poly.display_solution()


def small_polynomial_solution(N=4):
    print("\nTESTING POLYNOMIAL EASY SOLUTION, N=4 nodes!")
    # N = 4
    
    def f(p):
        return 4.0
    
    def u(p):
        return 1.0 - (p[0]**2 + p[1]**2)

    def class_BC(p):
        """
        Classify all edge nodes as Dirichlet
        """
        return 1
    
    FEM_poly = Poisson2DSolver(N=N, f=f, g_D=u, g_N=None, class_BC=class_BC)
    FEM_poly.display_mesh(nodes=None, elements=FEM_poly.edge_triangle_indexes)

    # Løser verdien i ett punkt:
    FEM_poly.solve_direct_dirichlet()
    FEM_poly.display_solution()

    # Prøvde å endre lokal-indekseringen av noder i et element:
    # FEM_poly.triang[1] = np.array([0, 3, 2], dtype='int32')
    # Gav rare utslag, som at A[2, 2] = 9.99e-16
        
    FEM_poly.generate_A_h()
    print("A_h:")
    matprint(FEM_poly.A_h.toarray())

    # Insert very specific source vec:
    # FEM_poly.F_h = np.array([FEM_poly.A_h[0, 0], 0.0, 0.0, 0.0]).reshape(4, 1)
    # FEM_poly.apply_direct_dirichlet()
    # FEM_poly.u_h = np.array([u(p) for p in FEM_poly.nodes])
    # FEM_poly.u_h[0] = sp.linalg.spsolve(FEM_poly.A_h, FEM_poly.F_h)
    # FEM_poly.display_solution()

    FEM_poly.generate_F_h()
    print(f"\nF_h:")
    print(FEM_poly.F_h)

    def F_1():
        """
        Find the second element of the source vector in the 4-node 
        super simple case.
        """
        print("Finding the second element of the source vector, F_h[1]:")
        I_0_2 = integrate_source_func_over_triangle(f=f, k=0, i_loc=1, FEM_solver=FEM_poly)
        I_2_3 = integrate_source_func_over_triangle(f=f, k=2, i_loc=2, FEM_solver=FEM_poly)
        F_h_1 = I_0_2 + I_2_3
    
        print(f"F_h[1] integrated in global coordinates: {F_h_1:.6e}")
        return F_h_1
    
    def F_0():
        """
        Find the first element of the source vector in the 4-node 
        super simple case.
        """
        print("Find the first element of the source vector, F_h[0]:")
        I_0_0 = integrate_source_func_over_triangle(f=f, k=0, i_loc=0, FEM_solver=FEM_poly)
        I_1_1 = integrate_source_func_over_triangle(f=f, k=1, i_loc=1, FEM_solver=FEM_poly)
        I_2_0 = integrate_source_func_over_triangle(f=f, k=2, i_loc=0, FEM_solver=FEM_poly)
        F_h_0 = I_0_0 + I_1_1 + I_2_0

        print(f"F_h[0] integrated in global coordinates: {F_h_0:.6e}")
        return F_h_0

    F_1()
    F_0()

    def A_1_1():
        print("\nFind A_1_1 by 'hand'.")

        J_0 = FEM_poly.generate_jacobian(element=0)
        J_0_inv = la.inv(J_0)
        I_0 = 0.5*la.det(J_0)*la.norm(J_0_inv[:, 0])**2

        J_2 = FEM_poly.generate_jacobian(element=2)
        J_2_inv = la.inv(J_2)
        I_2 = 0.5*la.det(J_2)*la.norm(J_2_inv[:, 1])**2

        print(f"A_1_1: {(I_0 + I_2):.6e}")
        return I_0 + I_2

    def A_3_3():
        print("\nFind A_3_3 by 'hand'.")

        J_1 = FEM_poly.generate_jacobian(element=1)
        J_1_inv = la.inv(J_1)
        inv_col_sum = J_1_inv[:, 0] + J_1_inv[:, 1]
        I_1 = 0.5*la.det(J_1)*la.norm(inv_col_sum)**2

        J_2 = FEM_poly.generate_jacobian(element=2)
        J_2_inv = la.inv(J_2)
        I_2 = 0.5*la.det(J_2)*la.norm(J_2_inv[:, 0])**2

        print(f"A_3_3: {(I_1 + I_2):.6e}")
        return I_1 + I_2

    A_1_1()
    A_3_3()

    print("\nFull matrix A_h:")
    matprint(FEM_poly.A_h.toarray())


def test_CG_FEM_solution(N=1000, TOL=1e-5):

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

    FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC, eps=1.0e-14)
    from time import time
    start = time()
    FEM_solver.solve_direct_dirichlet_CG(TOL=TOL)  # Might expand 'solve()' method to perform the above function calls.
    end = time()
    min_uh = min(FEM_solver.u_h)
    max_uh = max(FEM_solver.u_h)

    print("Showing Direct Dirichlet CG solution:")

    print(f"Minimum value: {min_uh:.5e}\nMax value: {max_uh:.5e}")

    print(f"CG solver: {(end-start):.2f} s")

    # FEM_solver.display_mesh()
    FEM_solver.display_solution()


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
        FEM_solver.solve()

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


def test_main():
    basic_tests()
    test_A_i_j()
    test_F_h()
    test_error()
    test_error_neumann()
    return

if __name__ == "__main__":
    test_main()