import numpy as np
import matplotlib.pyplot as plt

from .elasticity_solver import Elasticity2DSolver
from root.tools import BCtype, delete_from_csr


def test_elasticity_solver(N=5, area="plate"):
    
    def u(p):
        temp = (p[0]**2 - 1)*(p[1]**2 - 1)
        return np.array([temp, temp])

    def u2(p):
        return np.array([2*p[0] - 0.1*p[0]*p[1], p[1]**2 + 0.2*p[0]])

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": lambda p: BCtype.Dir, "E": 100.0, "nu": 0.22, "rho": 10.0, "area": area}
    a = Elasticity2DSolver.from_dict(model_dict)
    a.solve_vibration_modes(num=10)

    # a.display_vibration_mode(k=0)
    a.display_vibration_mode(k=4)
    
    # Skip first three "zero"-vibration frequencies:
    plt.scatter(range(len(a.vibration_frequencies))[3:], np.sqrt(a.vibration_frequencies[3:]))
    plt.title("Vibration frequencies ω")
    plt.xlabel("Index")
    plt.ylabel("ω")
    plt.show()

    print("Testing!")

    # a.display_vector_field(u2, title="TESTING")
    # a.display_mesh(nodes=a.edge_nodes)

    pass


def test_animation(N=20, area="plate", mode=5):
    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 10.0, "nu": 0.22, "rho": 1.0, "area": area}    
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=mode+1)
    solver.animate_vibration_mode(mode, alpha=1, l=5, savename=None)
    print("What?!")
    return


def test_animate_mesh_stress(N=6, area="plate"):
    from itertools import product as iter_product

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=N)

    k = N//2

    solver.animate_vibration_mode_stress(k=k, alpha=0.05, l=3)

    return


def test_markov(N=10, area="plate"):

    from time import time

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=N)

    k = N//2

    start = time()
    solver.animate_vibration_mode_stress(k=k, alpha=0.05, l=1, show=False, savename="mtest2", playtime=2*np.pi, fps=30)
    end = time()
    print(f'{(end-start)} s')

    return


def test_full_solver(N=10, area="plate"):
    E, nu, rho = 100, 0.22, 10

    def u(p):
        temp = (p[0]**2 - 1)*(p[1]**2 - 1)
        return np.array([temp, temp])

    def f(p):
        x, y = p
        f_1 = -2*y**2 - (1 - nu)*x**2 - 2*(1 + nu)*x*y + 3 - nu
        f_2 = -2*x**2 - (1 - nu)*y**2 - 2*(1 + nu)*x*y + 3 - nu
        return (E/(1 - nu**2))*np.array([f_1, f_2])

    def g_D(p):
        return np.array([0.0, 0.0])
    
    def class_BC(p):
        if p[0] == -1.0:
            return BCtype.Dir
        else:
            return -1

    def rho_func(p):
        return 100*abs(p[0] + p[1])*(rho) + 10.0

    model_dict = {"N": N, "f": f, "g_D": g_D, "g_N": lambda _: False,
                  "class_BC": class_BC, "E": E, "nu": nu, "rho": rho_func, "area": area}
    
    solver = Elasticity2DSolver.from_dict(model_dict)

    # Display analytical solution:
    # solver.display_vector_field(u=u, title="Analytical solution")

    # Lock the left-most edge:
    solver.solve_direct_dirichlet()

    """
    # Internal Model-Hacking:
    solver.generate_M_h()
    solver.M_h = delete_from_csr(solver.M_h, row_indices=solver.dirichlet_BC_basis_functions, 
                                             col_indices=solver.dirichlet_BC_basis_functions)
    """

    solver.solve_vibration_modes(num=20)
    solver.animate_vibration_mode_stress(k=4, alpha=0.5, l=1, show=True, savename=None, fps=30)

    # solver.display_vector_field(u=solver.u_h, title="FEM solution")
    # solver.display_mesh_stress(displacement=solver.u_h, show=True)

def test_mosaic(N=10, k=5, area="plate", figsize=(10,10), dims=(3,3), alpha=1,
                savename=None, show=None, dpi=None):

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 10.0, "nu": 0.22, "rho": 1.0, "area": area}    
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=k+1)

    solver.vibration_stress_mosaic(k=k, alpha=alpha, dims=dims, figsize=figsize,
                                    show=show, savename=savename, dpi=dpi)

    return

def test_show_frequencies(N=10, num=20, area="plate", savename=None, show=None):

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 10.0, "nu": 0.22*1.5, "rho": 1.0, "area": area}    
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=num)

    solver.show_frequencies()
    return

