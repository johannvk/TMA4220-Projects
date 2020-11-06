import numpy as np
import matplotlib.pyplot as plt

from .elasticity_solver import Elasticity2DSolver

def test_elasticity_solver(N=5, area="plate"):
    
    def u(p):
        temp = (p[0]**2 - 1)*(p[1]**2 - 1)
        return np.array([temp, temp])

    def u2(p):
        return np.array([2*p[0] - 0.1*p[0]*p[1], p[1]**2 + 0.2*p[0]])

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
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
    solver.solve_vibration_modes(num=10)
    solver.animate_vibration_mode(mode, alpha=1, l=5, savename=None)
    print("What?!")
    return


def steel_animation(N=10, area="plate", mode=5, max_modes=20):
    # Stainless Steel 18-8: 2x2 meter slab. 1 cm thick.
    # E:   193 [GPa] = 193e9 [Pa]
    # rho: 7.93 [g/cm^3] = 7.93e-3 [kg/cm^3]
    #      Per area: 7.93e-3 [kg/cm^2] = 7.93e1 [kg/m^2] 
    # nu:  0.305
    steel_plate = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": None, "E": 193.0e9, "nu": 0.305, "rho": 79.3e1, "area": area}
    steel_solver = Elasticity2DSolver.from_dict(steel_plate)
    steel_solver.solve_vibration_modes(num=max_modes)
    steel_solver.animate_vibration_mode_stress(mode, alpha=1.0, l=5, savename=None)


def aluminium_animation(N=10, area="plate", mode=5, max_modes=20):
    # Aluminum, 6061-T6: 2x2 meter slab. 1cm thick.
    # E:   69 [GPa] = 69e9 [Pa]
    # rho: 2.70 [g/cm^3] = 2.70e-3 [kg/cm^3]
    #      Per area: 2.70e-3 [kg/cm^2] = 2.70e1 [kg/m^2] 
    # nu:  0.35
    alu_plate = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                "class_BC": None, "E": 69.0e9, "nu": 0.35, "rho": 2.7e1, "area": area}
    alu_solver = Elasticity2DSolver.from_dict(alu_plate)
    alu_solver.solve_vibration_modes(num=max_modes)
    alu_solver.animate_vibration_mode_stress(mode, alpha=1.0, l=5, savename=None)


def timber_animation(N=10, area="plate", mode=5, max_modes=20):
    # Timber, western larch: 2x2 meter slab. 1cm thick.
    # E:   12.9 [GPa] = 12.9e9 [Pa]
    # rho: 5.2 [kg/m^3] 
    # nu:  0.276
    timber_plate = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                "class_BC": None, "E": 12.9e9, "nu": 0.276, "rho": 5.2e0, "area": area}
    timber_solver = Elasticity2DSolver.from_dict(timber_plate)
    timber_solver.solve_vibration_modes(num=max_modes)
    timber_solver.animate_vibration_mode_stress(mode, alpha=1.0, l=5, savename=None)


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
