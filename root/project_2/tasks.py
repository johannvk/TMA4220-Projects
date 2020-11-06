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

def test_animation(N=20, area="plate"):

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=10)

    solver.animate_vibration_mode(9, alpha=0.1, l=2, savename=None)

    return

def test_display_mesh_stress(N=6, area="plate"):
    from itertools import product as iter_product

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=10)

    k = 4

    vibration_eigenvec = solver.vibration_eigenvectors[:, k]
        
    displacement_vec = np.zeros(solver.nodes.shape)
    
    for n, d in iter_product(range(solver.num_nodes), (0, 1)):
        displacement_vec[n, d] = vibration_eigenvec[2*n + d]


    fig, ax = plt.subplots()

    #plot = solver.display_mesh_stress(displacement=displacement_vec * 0.05, show=False, ax=ax)

    solver.animate_vibration_mode_stress(k=4, alpha=0.2, l=3)

    return

