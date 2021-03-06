import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Plotting imports:
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# 2D FEM base class:
from ..triangular_2d_fem import Triangular2DFEM

# Our Gaussian-quadrature-code and other tools:
from ..tools import matprint, delete_from_csr, \
                    quadrature1D, quadrature2D, \
                    triangle_area, BCtype


class Poisson2DSolver(Triangular2DFEM):
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
        class_BC: Function BC_type([x, y]) -> Bool, returning True if point [x, y] \n
                  should be a Dirichlet BC. Assumed Neumann if not. 
        area: What geometry to solve the poisson equation on. Default: disc.
        """
        super(Poisson2DSolver, self).__init__(N=N, area=area)
        
        # Find which triangles are on the edge of our geometry:
        self.edge_triangle_indexes = list(filter(lambda i: any((node in self.triang[i] for node in self.edge_nodes)), 
                                                 np.arange(len(self.triang))))
        self.edge_triangles = list(self.triang[self.edge_triangle_indexes])
        
        self.num_nodes = N
        self.quad_points = quad_points
        self.area = area  # A bit superfluous. 

        # Problem source- and BC-functions:
        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        self.class_BC = class_BC
        self.eps = eps  # Big-Number Epsilon.

        # Store which nodes are BCtype.Dir and their values:
        self.dirichlet_BC_mask = np.zeros(self.num_nodes, dtype=bool)
        self.dirichlet_BC_nodes = []
        self.dirichlet_BC_values = []

        # Boolean indicating whether Boundary Conditions have been applied:
        self.applied_BC = False

        # Initialize the full Stiffness matrix to None before construction:
        self.A_h = None

        # Initialize the Source vector to None before construction:
        self.F_h = None

        # Initialize the Basis function coefficients "u_h" to None:
        self.u_h = None

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

    def A_i_j(self, i, j, J_inv_T, elem_area):
        """
        Function calculating the (Aₕ)ᵢ,ⱼ-th element of the "Stiffness"-matrix.
        i: Local index of basis function. [0, 1, 2]
        j: Local index of basis function. [0, 1, 2]
        J_inv_T: Inverse Jacobian transposed: [∂(r,s)/∂(x, y)].T
        elem_area: Area of the element: |J|/2
        """
        
        grad_i = J_inv_T @ self.reference_gradients[i]
        grad_j = J_inv_T @ self.reference_gradients[j]

        return elem_area * np.inner(grad_i, grad_j)

    def generate_A_h(self):
        """
        Generate the Stiffness Matrix A_h, based on linear Langrange basis functions on triangles.
        """
        self.A_h = sp.dok_matrix((self.num_nodes, self.num_nodes))

        # Loop through elements (triangles):        
        for k, element in enumerate(self.triang):
            
            J = self.generate_jacobian(k)
            J_inv_T = la.inv(J).T  # KRITISK FUCKINGS '.T'!
            element_area = 0.5*la.det(J)

            # Loop through nodes. Exploit symmetry of the (A_h)_sub-matrix symmetry.
            # Only compute the upper-triangular part i <= j. Symmetric about i=j.
            for i, node_i in enumerate(element):
                A_i_i = self.A_i_j(i, i, J_inv_T, element_area)

                self.A_h[node_i, node_i] += A_i_i
                
                for j in range(i+1, 3):
                    node_j = element[j]
                    A_i_j = self.A_i_j(i, j, J_inv_T, element_area)

                    self.A_h[node_i, node_j] += A_i_j
                    self.A_h[node_j, node_i] += A_i_j
        
        # Convert A_h to csr-format for ease of calculations later:
        self.A_h = self.A_h.tocsr() 
        
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
        
        # Reshape F_h to a column vector:
        self.F_h = self.F_h.reshape(self.num_nodes, 1)

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

    def apply_node_dirichlet_bc(self, node, p=None):
        if p is None:
            p = self.nodes[node]
        dir_bc_value = self.g_D(p)
        
        self.dirichlet_BC_mask[node] = True
        self.dirichlet_BC_nodes.append(node)
        self.dirichlet_BC_values.append(dir_bc_value)

        # Subtract from the source vector the column corresponding to 'node', 
        # times its Dirichlet BC-value: 
        self.F_h -= self.A_h[:, node]*dir_bc_value

    def apply_direct_dirichlet(self):
        # Dirichlet Boundary 
        for node in self.edge_nodes:
            p = self.nodes[node]

            if self.class_BC(p) == BCtype.Dir:
                self.apply_node_dirichlet_bc(node, p)
            
            elif self.class_BC(p) == BCtype.Neu:
                # Find which triangles the Edge node belongs to:
                raise ValueError("Cannot apply Neumann BC Using Direct Dirichlet function!")
        
        # Remove redundant degrees of freedom from A_h and F_h:
        F_mask = np.ones(len(self.F_h), dtype=bool)
        F_mask[self.dirichlet_BC_nodes] = False
        self.F_h = self.F_h[F_mask]

        self.A_h = delete_from_csr(self.A_h, row_indices=self.dirichlet_BC_nodes, 
                                             col_indices=self.dirichlet_BC_nodes)
        self.applied_BC = True

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions element-wise. Take care to avoid
        applying Dirichlet boundary conditions for a node more than once.
        """
        for k, element in zip(self.edge_triangle_indexes, self.edge_triangles):
            
            element_edge_nodes = [node for node in element if node in self.edge_nodes]
            edge_bc_types = [self.class_BC(self.nodes[node_i]) for node_i in element]

            # Should never have more than two edge nodes in an element:
            assert(len(element_edge_nodes) in (1, 2))            
            
            F_inv = lambda p: self.global_to_reference_transformation(p, element=k)

            for i, (node, bc_type) in enumerate(zip(element, edge_bc_types)):
                # Only add boundary conditions for edge nodes:
                if node not in element_edge_nodes:
                    continue
                
                # Ensure that the Dirichlet node's bc has not been applied yet
                if bc_type == BCtype.Dir and not self.dirichlet_BC_mask[node]:
                    self.apply_node_dirichlet_bc(node)

                elif bc_type == BCtype.Neu:
                    
                    # No contribution if there is only one edge node in the element:
                    if len(element_edge_nodes) == 1:
                        continue
                    
                    a = self.nodes[element_edge_nodes[0]]
                    b = self.nodes[element_edge_nodes[1]]

                    def integrand(p):
                        phi = self.basis_functions[i](F_inv(p))
                        g_n = self.g_N(p)
                        return g_n*phi
                    
                    neumann_contribution = quadrature1D(integrand, a, b, Nq=self.quad_points)

                    # Add contribution from integrating over the element outer edge:
                    self.F_h[node] += neumann_contribution
                    
        # Remove redundant degrees of freedom from A_h and F_h:
        F_mask = np.ones(len(self.F_h), dtype=bool)
        F_mask[self.dirichlet_BC_nodes] = False
        self.F_h = self.F_h[F_mask]

        self.A_h = delete_from_csr(self.A_h, row_indices=self.dirichlet_BC_nodes, 
                                             col_indices=self.dirichlet_BC_nodes)
        self.applied_BC = True

    def solve_big_number_dirichlet(self):
        self.generate_A_h()
        self.generate_F_h()
        self.apply_big_number_dirichlet()
        self.u_h = sp.linalg.spsolve(self.A_h, self.F_h)

    def solve_direct_dirichlet(self):
        self.generate_A_h()
        self.generate_F_h()
        self.apply_direct_dirichlet()
        
        reduced_u_h = sp.linalg.spsolve(self.A_h, self.F_h)
        self.u_h = np.zeros(self.num_nodes)
        self.u_h[~self.dirichlet_BC_mask] = reduced_u_h
        self.u_h[self.dirichlet_BC_mask] = self.dirichlet_BC_values

    def solve(self):
        self.generate_A_h()
        self.generate_F_h()
        self.apply_boundary_conditions()

        reduced_u_h = sp.linalg.spsolve(self.A_h, self.F_h)
        self.u_h = np.zeros(self.num_nodes)
        self.u_h[~self.dirichlet_BC_mask] = reduced_u_h
        self.u_h[self.dirichlet_BC_mask] = self.dirichlet_BC_values

    def solve_direct_dirichlet_CG(self, TOL=1e-5):
        self.generate_A_h()
        self.generate_F_h()
        self.apply_direct_dirichlet()
        
        reduced_u_h, exit_code = sp.linalg.cg(self.A_h, self.F_h, tol=TOL)
        assert exit_code == 0 
        self.u_h = np.zeros(self.num_nodes)
        self.u_h[~self.dirichlet_BC_mask] = reduced_u_h
        self.u_h[self.dirichlet_BC_mask] = self.dirichlet_BC_values

    def error_est(self, u_ex, quad_points=None):
        assert isinstance(self.u_h, np.ndarray)

        if quad_points == None:
            quad_points = self.quad_points

        E = 0

        """ For each element in triangulation. """ 
        for k, element in enumerate(self.triang):

            F_inv = lambda p: self.global_to_reference_transformation(p, k, J_inv=None)
            p1, p2, p3 = element
            x1 = self.nodes[p1]
            x2 = self.nodes[p2]
            x3 = self.nodes[p3]

            u1 = self.u_h[p1]
            u2 = self.u_h[p2]
            u3 = self.u_h[p3]

            
            phi1, phi2, phi3 = self.basis_functions
            """ err = ( u_h - u_ex )**2 """
            err = lambda x: ( u1*phi1(F_inv(x)) + u2*phi2(F_inv(x)) + u3*phi3(F_inv(x)) - u_ex(x) )**2
            
            """ Gauss quadrature approximation to contribution to square error from element k """
            E += quadrature2D(err, x1, x2, x3, Nq=quad_points)

        return np.sqrt(E)

    def find_h(self):

        h = 0

        for k, element in enumerate(self.triang):

            p1, p2, p3 = element
            x1 = self.nodes[p1]
            x2 = self.nodes[p2]
            x3 = self.nodes[p3]
            hk = max(np.linalg.norm(x2 - x1), np.linalg.norm(x3 - x1), np.linalg.norm(x3 - x2))

            if hk > h:
                h = hk

        return h

    def evaluate(self, p):
        """
        Some smart generator function returning sum of basis functions at the point [x, y],
        located in some element K.
        p: np.array([x, y])
        """

        raise NotImplementedError

    def display_solution(self, u_h=None, title=None):
        if u_h is None:
            u_h = self.u_h
        else:
            assert(len(u_h) == self.num_nodes)
        
        plt.rcParams.update({'font.size': 18})

        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        if title is None:
            fig.suptitle("Solution Plot")
        else:
            fig.suptitle(title)
        
        min_u_h, max_u_h = np.min(u_h), np.max(u_h)
        scale = abs(max_u_h - min_u_h)

        cmap = plt.cm.jet        
        norm = mpl.colors.Normalize(vmin=min_u_h - 0.1*scale, vmax=max_u_h + 0.1*scale)

        tri_plot = ax.plot_trisurf(self.nodes[:, 0], self.nodes[:, 1], u_h, triangles=self.triang, 
                                   cmap=cmap, antialiased=True)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        
        ax.set_zlim(min_u_h - 0.1*scale, max_u_h + 0.1*scale)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        plt.show()
            
