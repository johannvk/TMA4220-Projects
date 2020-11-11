import numpy as np
import matplotlib.pyplot as plt

from .elasticity_solver import Elasticity2DSolver
from ..tools import BCtype

def beta(x, y):
    '''
        Estimator for the coefficient of beta in linear regression model
            y = alpha + beta * x
    '''
    n = x.shape[0]
    beta = np.sum( (x - np.mean(x)) * (y - np.mean(y))) / np.sum( (x - np.mean(x))**2 )
    return beta


def L2_convergence(area="plate", show=True):
    # Calculate the L2-norm convergence-plot for analytical solutions to 
    # the linear elasticity equation.

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

    Ns = [5, 10, 15, 20, 30]
    errors = np.zeros(len(Ns))
    hs = np.zeros(len(Ns))

    test_k = 1

    for i, n in enumerate(Ns):
        print(f"Dirichlet: N = {n}")
        model_dict = {"N": n, "f": f, "g_D": g_D, "g_N": lambda _: False,
                    "class_BC": class_BC, "E": E, "nu": nu, "rho": rho, "area": area}
        
        solver = Elasticity2DSolver.from_dict(model_dict)
        solver.solve_direct_dirichlet()
        solver.display_L2_error(u_ex=u)
        # solver.display_single_element_error(k=test_k, u=lambda p: u(p) - solver.fem_solution(k=test_k)(p))
        # solver.display_vector_field(title="FEM solution")
        # solver.display_vector_field(u=u, title="Exact solution")
        # solver.display_mesh_stress(displacement=np.array([u(p) for p in solver.nodes]))

        # Show erronous stresses:
        # error_displacement = solver.u_h - np.array([u(p) for p in solver.nodes])
        # solver.display_mesh_stress(displacement=error_displacement)
        # solver.display_vector_field(u=error_displacement, title="Error vector field.")

        errors[i] = solver.L2_norm_error(u_ex=u)
        hs[i] = solver.find_h()

    # Source: https://www.wolframalpha.com/input/?i=integrate+2*%28%28x%5E2-1%29*%28y%5E2-1%29%29%5E2+from+x+%3D+-+1+to+x%3D1%2C+from+y+%3D+-1+to+y%3D1
    norm_u = 512.0/225.0

    errors_rel = errors / norm_u

    betaD = beta(np.log(hs), np.log(errors_rel))
    print(f'Dirichlet: beta = {betaD}')
    
    plt.rcParams.update({'font.size': 14})

    plt.figure()
    plt.loglog(hs, errors_rel, 'k-', label=r"$||u - u_h||_{L_2(\Omega)} / ||u||_{L_2(\Omega)}$")
    plt.text(hs[2], 0.8*errors_rel[2], fr"Slope $\approx$ {betaD:.2f}")
    plt.xlabel("$h$")
    plt.title("Relative error, Dirichlet BC's")
    plt.legend()

    if show:
        plt.show()
    
