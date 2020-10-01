import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D


# Mesh generation code from lecturers:
from triangulation.getdisc import GetDisc

# Our Gaussian-quadrature-code:
from gaussian_quad import quadrature2D, triangle_area

class BCtype:
    Neu = 0
    Dir = 1


class Poisson2DSolver():
    """
    2D-FEM solver of the Poisson equation:\n
        -∆u(x, y) = f(x, y), (x, y) ∈ ℜ².
    \nUsing Linear basis function Polynomials.    
    """
    
    def __init__(self, N, f, g_D, g_N, class_BC, quad_points=4, area="disc", eps=1.0e-14):
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
        self.quad_points = quad_points
        self.area = area  # A bit superfluous. 

        # Problem source- and BC-functions:
        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        self.class_BC = class_BC
        self.eps = eps  # Big-Number Epsilon.

        # Boolean indicating whether Boundary Conditions have been applied:
        self.applied_BC = False

        # Initialize the full Stiffness matrix to None before construction:
        self.A_h = None

        # Initialize the Source vector to None before construction:
        self.F_h = None

        # Initialize the Basis function coefficients "u_h" to None:
        self.u_h = None

        # Linear Lagrange polynomial basis functions on the reference triangle:
        self.basis_functions = (lambda eta: 1 - eta[0] - eta[1], lambda eta: eta[0], lambda eta: eta[1])

        # Gradients of basis functions in reference coordinates:
        self.reference_gradients = (np.array([-1.0, -1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

        # Reference triangle:
        self.reference_triangle_nodes = (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

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
        J: If the transformation is called repeatedly on the same element, 
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
        J_inv: If the transformation is called repeatedly on the same element,
               one can provide the already calculated inverse Jacobian.
               TODO: Do we want to return the calculated inverse-Jacobian as well? Think No.
        """
        translation = self.nodes[self.triang[element][0]]
        if J_inv is None:
            J_inv = la.inv(self.generate_jacobian(element))
        return J_inv @ (p - translation)

    def A_i_j(self, i, j, J_inv, elem_area):
        """
        Function calculating the (Aₕ)ᵢ,ⱼ-th element of the "Stiffness"-matrix.
        i: Local index of basis function. [0, 1, 2]
        j: Local index of basis function. [0, 1, 2]
        element: Which element to integrate over. Scale integrand using the Jacobian matrix.
        J_inv: Inverse Jacobian: ∂(r,s)/∂(x, y)
        elem_area: Area of the element: |J|/2
        """
        
        grad_i = J_inv @ self.reference_gradients[i]
        grad_j = J_inv @ self.reference_gradients[j]

        return elem_area * np.inner(grad_i, grad_j)

    def generate_A_h(self):
        """
        Generate the Stiffness Matrix A_h, based on linear Langrange basis functions on triangles.
        """
        self.A_h = sp.dok_matrix((self.num_nodes, self.num_nodes))

        # Loop through elements (triangles):        
        for k, element in enumerate(self.triang):
            
            J = self.generate_jacobian(k)
            J_inv = la.inv(J)
            element_area = 0.5*la.det(J)

            # Loop through nodes. Exploit symmetry of the (A_h)_sub-matrix symmetry.
            # Only compute the upper-triangular part i <= j. Symmetric around i=j.
            for i, node_i in enumerate(element):
                A_i_i = self.A_i_j(i, i, J_inv, element_area)
                self.A_h[node_i, node_i] += A_i_i

                # print(f"At nodes (i, j): ({node_i}, {node_i})")

                for j in range(i+1, 3):
                    node_j = element[j]
                    A_i_j = self.A_i_j(i, j, J_inv, element_area)

                    self.A_h[node_i, node_j] += A_i_j
                    self.A_h[node_j, node_i] += A_i_j

                    # print(f"At nodes (i, j): ({node_i}, {node_j})")

    def generate_F_h(self):
        """
        Generate the source vector. Sum over elements and add contributions from each basis function.
        """
        # Making the full Source vector:
        self.F_h = np.zeros(self.num_nodes)

        # Reference triangle nodes:
        eta1, eta2, eta3 = self.reference_triangle_nodes

        for k, element in enumerate(self.triang):
            J_k = self.generate_jacobian(k)
            det_J_k = la.det(J_k)

            # Loop through nodes in element:
            for i, node in enumerate(element):

                # Integrand:
                integrand = lambda eta: self.f(self.reference_to_global_transformation(eta, k, J_k))*self.basis_functions[i](eta)

                # Add contribution from element to node-row. Integrate overe reference triangle.
                self.F_h[node] += det_J_k*quadrature2D(integrand, eta1, eta2, eta3, self.quad_points)

    def apply_big_number_dirichlet(self, eps=None):
        """
        Apply pure Dirichlet boundary conditions to A_h and F_h 
        using the "Big Number"-approach.
        """
        if self.A_h is None or self.F_h is None:
            print("Cannot apply boundary conditions before A_h and F_h are constructed!")
            return
        if eps is None:
            eps = self.eps

        eps_inv = 1.0/eps        
        for node in self.edge_nodes:
            p = self.nodes[node]
            class_BC = self.class_BC(p)

            if class_BC == BCtype.Dir:
                """
                If node is a Dirichlet node
                """
                g_D = self.g_D(p)
                self.A_h[node, node] = eps_inv
                self.F_h[node] = g_D*eps_inv
        
        self.applied_BC = True

    def solve_big_number_dirichlet(self):
        self.generate_A_h()
        self.generate_F_h()
        self.apply_big_number_dirichlet()
        self.A_h.tocsr()
        self.u_h = sp.linalg.spsolve(self.A_h, self.F_h)

    def evaluate(self, p):
        """
        Some smart generator function returning sum of basis functions at the point [x, y],
        located in some element K.
        p: np.array([x, y])
        """
        pass

    def display_solution(self, u_h=None):
        """
        Need a way of evaluating the sum of basis functions in a smart way. 
        """
        if u_h is None:
            u_h = self.u_h
        else:
            assert(len(u_h) == self.num_nodes)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.suptitle("Solution Plot")
        
        # Kind of what we want: Values at nodes decided by u_h-array.
        # Then the value descendes to zero on all nodes around it.
        ax.plot_trisurf(self.nodes[:, 0], self.nodes[:, 1], u_h, triangles=self.triang, 
                        cmap=plt.cm.viridis, antialiased=True)
        ax.set_zlim(-1.2, 1.2)
        plt.show()
            

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
