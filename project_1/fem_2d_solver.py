import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Mesh generation code from lecturers:
from triangulation.getdisc import GetDisc

# Our Gaussian-quadrature-code:
from gaussian_quad import quadrature2D


class Poisson2DSolver():
    """
    2D-FEM solver of the Poisson equation:\n
        -∆u(x, y) = f(x, y), (x, y) ∈ ℜ².
    \nUsing Linear basis function Polynomials.    
    """
    
    def __init__(self, N, f, g_D, g_N, dir_BC, area="disc"):
        """
        Initializer for 2D-FEM solver of the Poisson equation:\n
        -∆u = f, (x, y) ∈ Ω \n

        N: Number of nodes for the mesh.
        f: Source function.
        g_D: Function g([x, y]) -> R, specifying Dirichlet boundary conditions.
        g_N: Function g([x, y]) -> R, specifying Neumann boundary conditions.
        Dir_BC: Function BC_type([x, y]) -> Bool, returning True if point [x, y] \n
                should be a Dirichlet BC. Assumed Neumann if not. 
        area: What geometry to solve the poisson equation on. Default: disc.
        """
        if area == "disc":
            self.nodes, self.triang, self.edges = GetDisc(N)
            self.edge_nodes = self.edges[:, 0]
        else:
            raise NotImplementedError("Ups, only support the 'disc' geometry.")
        
        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        self.dir_BC = dir_BC
        self.area = area  # A bit superfluous. 

    def display_mesh(self, node: int =None, elements=None):
        if elements is None and node is None:
            element_triang = self.triang
        elif node is not None:
            triangle_indices = [i for i, triangle in enumerate(self.triang) if node in triangle] 
            element_triang = self.triang[triangle_indices]
        else:
            element_triang = self.triang[elements]

        plt.triplot(self.nodes[:, 0], self.nodes[:, 1], triangles=element_triang)
        plt.show()



a = Poisson2DSolver(15, 0.0, 0.0, 0.0, 0.0)
a.display_mesh(node=1)
