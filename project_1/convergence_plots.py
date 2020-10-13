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


def beta(x, y):
    '''
        Estimator for the coefficient of beta in linear regression model
            y = alpha + beta * x
    '''
    n = x.shape[0]
    beta = np.sum( (x - np.mean(x)) * (y - np.mean(y))) / np.sum( (x - np.mean(x))**2 )
    return beta

def dirichlet_convergence(show=True):

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
    Ns = [20, 40, 80, 160, 320, 640, 1280, 2560]
    for N in Ns:
        print(f"Dirichlet: N = {N}")

        FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=None, class_BC=class_BC, eps=1.0e-14)
        FEM_solver.solve_direct_dirichlet()

        e = FEM_solver.error_est(u_ex)

        Es.append(e)

    norm_u = np.pi / 2

    Es = np.array(Es, dtype=float)
    Es_rel = Es / norm_u
    Ns = np.array(Ns, dtype=float)

    betaD = beta(np.log(Ns), np.log(Es_rel))
    print(f'Dirichlet: beta = {betaD}')
    
    plt.rcParams.update({'font.size': 14})

    plt.figure()
    plt.loglog(Ns, Es_rel, 'k-', label=r"$||u - u_h||_{L_2(\Omega)} / ||u||_{L_2(\Omega)}$")
    plt.text(Ns[2], Es_rel[2], f"Slope = {betaD:.2f}")
    plt.xlabel("Degrees of freedom")
    plt.title("Relative error, Dirichlet")
    plt.legend()

    if show:
        plt.show()


def neumann_convergence(show=True):

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
    Ns = [20, 40, 80, 160, 320, 640, 1280, 2560]
    for N in Ns:
        print(f"Neumann: N = {N}")

        FEM_solver = Poisson2DSolver(N=N, f=f, g_D=g_D, g_N=g_N, class_BC=class_BC, eps=1.0e-14)
        FEM_solver.solve_direct_dirichlet()

        e = FEM_solver.error_est(u_ex)

        Es.append(e)

    norm_u = np.pi / 2

    Es = np.array(Es, dtype=float)
    Es_rel = Es / norm_u
    Ns = np.array(Ns, dtype=float)

    betaN = beta(np.log(Ns), np.log(Es_rel))
    print(f'Neumann: beta = {betaN}')

    plt.rcParams.update({'font.size': 14})

    plt.figure()
    plt.loglog(Ns, Es_rel, 'k-', label=r"$||u - u_h||_{L_2(\Omega)} / ||u||_{L_2(\Omega)}$")
    plt.text(Ns[2], Es_rel[2], f"Slope = {betaN:.2f}")
    plt.xlabel("Degrees of freedom")
    plt.title("Relative error, Neumann")
    plt.legend()


    if show:
        plt.show()



if __name__ == '__main__':

    dirichlet_convergence(show=False)
    neumann_convergence(show=False)
    plt.show()
    
    pass
