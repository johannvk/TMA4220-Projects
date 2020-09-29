import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


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
        self.num_nodes = N
        self.num_unknowns = N - len(self.edge_nodes)

        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        self.dir_BC = dir_BC
        self.area = area  # A bit superfluous. 

        # Linear Lagrange polynomial basis functions on the reference triangle:
        self.basis_functions = [lambda eta: 1 - eta[0] - eta[1], lambda eta: eta[0], lambda eta: eta[1]]

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

    def generate_jacobian(self, element: int):
        """
        Function to generate the Jacobian J = ∂(x,y)/∂(r, s)\n
        for transforming from the reference triangle to global coordinates.\n
        element: The target element (triangle) of the transformation from the reference element.
        """
        p1, p2, p3 = self.nodes[self.triang[element]]
        J = np.column_stack([p2-p1, p3-p1])
        return J

    def reference_to_global_transformation(self, eta, element, J=None):
        """
        Function transforming reference coordinates eta = [r, s] to 
        global coordinates [x, y] for a given element.
        eta: np.array([r, s])
        element: int in range(Num_elements)
        J: If the transformation is called repeatedly on the same element,\n
           one can provide the already calculated Jacobian.
           TODO: Do we want to return the calculated Jacobian as well? Think No.
        """
        translation = self.nodes[self.triang[element][0]]
        if J is None:
            J = self.generate_jacobian(element)
        return J @ np.array(eta) + translation
    
    def global_to_reference_transformation(self, p, element, J_inv=None):
        """
        Function transforming global coordinates p = [x, y] to 
        reference coordinates eta = [r, s] for a given element.
        p: np.array([x, y])
        element: int in range(Num_elements)
        J_inv: If the transformation is called repeatedly on the same element,\n
               one can provide the already calculated inverse Jacobian.
               TODO: Do we want to return the calculated inverse-Jacobian as well? Think No.
        """
        translation = self.nodes[self.triang[element][0]]
        if J_inv is None:
            J_inv = np.linalg.inv(self.generate_jacobian(element))
        return J_inv @ (p - translation)

    def evaluate(self, p):
        """
        Some smart generator function returning sum of basis functions at the point [x, y],
        located in some element K.
        p: np.array([x, y])
        """
        pass

    def display_solution(self, u_h):
        """
        Need a way of evluating the sum of basis functions in a smart way. 
        """
        assert(len(u_h) == self.num_nodes)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.suptitle("Solution Plot")
        # Kind of what we want: Values at nodes decided by u_h-array.
        # Then the value descendes to zero on all nodes around it.
        ax.plot_trisurf(self.nodes[:, 0], self.nodes[:, 1], u_h, triangles=self.triang, 
                        cmap=plt.cm.viridis, antialiased=True)
        ax.set_zlim(-1.2, 1.2)
        plt.show()
        

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


def main():
    basic_tests()
    pass


if __name__ == "__main__":
    main()