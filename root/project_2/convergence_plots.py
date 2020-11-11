import numpy as np

from .elasticity_solver import Elasticity2DSolver
from ..tools import BCtype


def L2_convergence(N=10, area="plate"):
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
        return BCtype.Dir

    model_dict = {"N": N, "f": f, "g_D": g_D, "g_N": lambda _: False,
                  "class_BC": class_BC, "E": E, "nu": nu, "rho": rho, "area": area}
    
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_direct_dirichlet()
    solver.display_vector_field()
    L_2_error = solver.L2_norm_error(u_ex=u)

    # Display analytical solution:
    # solver.display_vector_field(u=u, title="Analytical solution")

    # Lock the left-most edge:
    solver.solve_direct_dirichlet()
    # Calculate the L2-norm convergence-plot for analytical solutions to 
    # the linear elasticity equation.
    pass