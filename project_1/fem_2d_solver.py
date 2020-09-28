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
        # Geometry setup:
        if area == "disc":
            self.nodes, self.triang, self.edges = GetDisc(N)
        else:
            raise NotImplementedError("Ups, only support the 'disc' geometry.")
        self.edge_nodes = self.edges[:, 0]
        self.num_unknowns = N - len(self.edge_nodes)

        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        self.dir_BC = dir_BC
        self.area = area  # A bit superfluous. 

    def display_mesh(self, nodes=None, elements=None):
        if elements is None and nodes is None:
            element_triang = self.triang

        elif nodes is not None:
            # Find all triangles with nodes-elements as vertices.
            if type(nodes) is int:
                triangle_indices = [i for i, triangle in enumerate(self.triang) if nodes in triangle] 
            else:
                # Stupidly unreadable One-line solution:
                triangle_indices = list(filter(lambda i: any((node in self.triang[i] for node in nodes)), 
                                               np.arange(len(self.triang))))
                
                # Old solution:
                # triangle_indices = []
                # for node in nodes:
                #     triangle_indices += [i for i, triangle in enumerate(self.triang) if node in triangle]

            element_triang = self.triang[triangle_indices]

        else:
            element_triang = self.triang[elements]

        plt.triplot(self.nodes[:, 0], self.nodes[:, 1], triangles=element_triang)
        plt.show()


a = Poisson2DSolver(15, 0.0, 0.0, 0.0, 0.0)
a.display_mesh(nodes=np.arange(start=0, stop=3))
a.display_mesh()