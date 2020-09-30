import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# Mesh generation code from lecturers:
from triangulation.getdisc import GetDisc

# Our Gaussian-quadrature-code:
from gaussian_quad import quadrature2D, triangle_area


class Poisson2DSolver():
    """
    2D-FEM solver of the Poisson equation:\n
        -∆u(x, y) = f(x, y), (x, y) ∈ ℜ².
    \nUsing Linear basis function Polynomials.    
    """
    
    def __init__(self, N, f, g_D, g_N, dir_BC, quad_points=4, area="disc"):
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

        # Problem source- and BC-functions:
        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        self.dir_BC = dir_BC
        self.area = area  # A bit superfluous. 

        # Initialize the full Stiffness matrix to None before construction:
        self.A_h = None

        # Initialize the Source vector to None before construction:
        self.F_h = None

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


def test_A_i_j():
    num_nodes = 10
    a = Poisson2DSolver(num_nodes, 0.0, 0.0, 0.0, 0.0)
    print("nodes:\n", a.nodes)
    # a.display_mesh(nodes=len(a.nodes)-1)
    
    # test_elem = 0
    # J = a.generate_jacobian(test_elem)
    # J_inv = np.linalg.inv(J)
    # elem_area = np.linalg.det(J)*0.5
 
    # i, j = 1, 2
    # A_i_j = a.A_i_j(i, j, J_inv, elem_area)
    # print(f"A_{i}_{j} = {A_i_j:.6f}")

    a.generate_A_h()
    print("\nA_h:")
    A_h = a.A_h.toarray()
    matprint(A_h)
    col_sums = np.sum(A_h, axis=-1)
    print("col_sums:\n", col_sums)
    x_test = np.linalg.solve(A_h, np.random.randn(num_nodes))
    A_h_eigvals = la.eigvals(A_h) # One zero-eigenvalue. Singular.
    a.display_mesh()


def test_F_h():

    def f(p):
        """
        Source function f(r, theta) = −8π*cos(2πr²)+ 16π²r²sin(2πr²)
        p: np.array([x, y])
        """
        r_squared = p[0]**2 + p[1]**2
        term_1 = -8.0*np.pi*np.cos(2*np.pi*r_squared)
        term_2 = 16*np.pi**2*r_squared*np.sin(2*np.pi*r_squared)
        return term_1 + term_2
    N = 100
    Solver = Poisson2DSolver(N, f, 0.0, 0.0, 0.0)
    Solver.generate_F_h()
    print(Solver.F_h)
    Solver.display_mesh()



def task_2_e():
    # Function conforming that the full Stiffness Matrix is singular.
    FEM_dummy_solver = Poisson2DSolver(100, True, True, True, True)
    FEM_dummy_solver.generate_A_h()
    A_h = FEM_dummy_solver.A_h.toarray()
    eigvals = la.eigvals(A_h)
    if any(np.abs(eigvals) < 1.0e-15):
        print("The matrix A_h, constructed and without imposing any boundary conditions, is Singular.")
    

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def main():
    # basic_tests()
    # test_A_i_j()
    
    test_F_h()
    # task_2_e()
    pass


if __name__ == "__main__":
    main()