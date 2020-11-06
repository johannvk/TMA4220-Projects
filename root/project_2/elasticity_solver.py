import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from itertools import product as iter_product

from ..triangulation import getPlate, GetDisc

from ..tools import matprint, delete_from_csr, \
                    quadrature1D, quadrature2D, BCtype


class Elasticity2DSolver():
    """
    2D-FEM solver of the static equilibrium equation:\n
        ∇ᵀσ(u) = -f, (x, y) ∈ Ω = [-1, 1]² \n
    \nUsing Linear basis function Polynomials.    
    """
    def __init__(self, N, f, g_D, g_N, class_BC, E, nu, rho,
                 quad_points=4, area="plate", eps=1.0e-10):
        """
        Initializer for 2D-FEM solver of the static equilibrium equation:\n
            ∇ᵀσ(u) = -f, (x, y) ∈ Ω = [-1, 1]² \n

        N:      sqrt(Number of nodes for the mesh). \n
        f:      Source function. ℜ² ⇒ ℜ².          \n
        E:      Youngs modulus of your material.    \n
        nu:     Poisson ratio of your material.     \n
        rho:    Density per area of your materal.   \n
        g_D:    Function g([x, y]) -> R, specifying Dirichlet boundary conditions. \n
        g_N:    Function g([x, y]) -> R, specifying Neumann boundary conditions.   \n
        class_BC: Function BC_type([x, y]) -> Bool, returning True if point [x, y] 
                  should be a Dirichlet BC. Assumed Neumann if not.                \n
        area: What geometry to solve the elasticity equation on. Default: "plate".
        """
        # Geometry setup:
        if area == "plate":
            # We get incorrect edges out of 'getPlate()'. 
            # Fixed to only get edge nodes for now.
            self.nodes, self.triang, self.edge_nodes = getPlate(N)
        elif area == "disc":
            self.nodes, self.triang, edges = GetDisc(N)
            self.edge_nodes = edges[:, 0]
        else:
            raise NotImplementedError("Ups, only support the 'plate' and 'disc' geometry.")
        
        self.edge_triangle_indexes = list(filter(lambda i: any((node in self.triang[i] for node in self.edge_nodes)), 
                                                 np.arange(len(self.triang))))
        self.edge_triangles = list(self.triang[self.edge_triangle_indexes])

        self.num_nodes = len(self.nodes)

        # Two basis functions for each node: 
        # (ϕ_i_0 = [ϕ_i, 0], ϕ_i_1 = [0, ϕ_i]), i = 1, .., num_nodes.
        self.num_basis_functions = 2*self.num_nodes  

        self.quad_points = quad_points
        self.area = area  # A bit superfluous. 

        # Problem source- and BC-functions:
        self.f = f
        self.g_D = g_D
        self.g_N = g_N
        
        # Material properties:
        self.E = E
        self.nu = nu
        self.rho = rho

        self.class_BC = class_BC
        self.eps = eps  # Big-Number Epsilon.
        
        # Store the transformation matrix C: σ_vec = C @ ε_vec
        self.C = (E/(1 - nu**2)) * np.array([[1.0,  nu,          0.0],
                                           [ nu, 1.0,          0.0],
                                           [0.0, 0.0, (1 - nu)/2.0]])

        # Store which nodes are BCtype.Dir and their values:
        self.dirichlet_BC_mask = np.zeros(self.num_nodes, dtype=bool)
        self.dirichlet_BC_nodes = []
        self.dirichlet_BC_values = []

        # Boolean indicating whether Boundary Conditions have been applied:
        self.applied_BC = False

        # Initialize the full Stiffness matrix to None before construction:
        self.A_h = None

        # Initialize the full Mass matrix to None before construction:
        self.M_h = None

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
    
    @classmethod
    def from_dict(cls, model_dict):
        return cls(**model_dict)

    def display_mesh(self, nodes=None, elements=None, displacement=None, show=True, ax=None):
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

            element_triang = self.triang[triangle_indices]

        else:
            element_triang = self.triang[elements]
        
        nodes_x = np.copy(self.nodes[:, 0])
        nodes_y = np.copy(self.nodes[:, 1])

        if displacement is not None:
            # Apply displacement to each node:
            assert(self.nodes.shape == displacement.shape)
            nodes_x += displacement[:, 0]
            nodes_y += displacement[:, 1]

        if ax is not None:
            plot = ax.triplot(nodes_x, nodes_y, triangles=element_triang, color='black')                        
        else:
            plot = plt.triplot(nodes_x, nodes_y, triangles=element_triang)

        if show:

            x_min, x_max = np.min(nodes_x), np.max(nodes_x)
            y_min, y_max = np.min(nodes_y), np.max(nodes_y)
            scale_x = abs(x_max - x_min)
            scale_y = abs(y_max - y_min)

            margin = 0.05
            plt.xlim(x_min - margin*scale_x, x_max + margin*scale_x)
            plt.ylim(y_min - margin*scale_y, y_max + margin*scale_y)

            plt.show()

        else:
            return plot

    def display_mesh_stress(self, nodes=None, elements=None, displacement=None, face_colors=None, show=True, ax=None):

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

            element_triang = self.triang[triangle_indices]

        else:
            element_triang = self.triang[elements]
        
        nodes_x = np.copy(self.nodes[:, 0])
        nodes_y = np.copy(self.nodes[:, 1])

        if displacement is not None:
            # Apply displacement to each node:
            assert(self.nodes.shape == displacement.shape)
            nodes_x += displacement[:, 0]
            nodes_y += displacement[:, 1]

        if face_colors is None:
            xmid = nodes_x[element_triang].mean(axis=1)
            ymid = nodes_y[element_triang].mean(axis=1)
            zcolors = xmid**2 + ymid**2
            #raise NotImplementedError

        if ax is not None:
            #plot = ax.triplot(nodes_x, nodes_y, triangles=element_triang, color='black')
            plot = ax.tripcolor(nodes_x, nodes_y, triangles=element_triang, facecolors=zcolors, edgecolors='k')
        else:
            #plot = plt.triplot(nodes_x, nodes_y, triangles=element_triang)
            plot = plt.tripcolor(nodes_x, nodes_y, triangles=element_triang, facecolors=zcolors, edgecolors='k')

        if show:

            x_min, x_max = np.min(nodes_x), np.max(nodes_x)
            y_min, y_max = np.min(nodes_y), np.max(nodes_y)
            scale_x = abs(x_max - x_min)
            scale_y = abs(y_max - y_min)

            margin = 0.05
            plt.xlim(x_min - margin*scale_x, x_max + margin*scale_x)
            plt.ylim(y_min - margin*scale_y, y_max + margin*scale_y)

            plt.show()

        else:
            return plot

    def display_vector_field(self, u, title=None):
        """
        Display a vector field over the domain Ω.\n
            u([x, y]) → [u_x(x, y), u_y(x, y)]
        """
        assert(callable(u))
        
        plt.rcParams.update({'font.size': 18})

        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(111)

        if title is None:
            fig.suptitle("Vector Field Plot")
        else:
            fig.suptitle(title)
        # Arrow locations:
        X, Y = self.nodes[:, 0], self.nodes[:, 1]

        # Vector X- and Y-components:
        vectors = np.array([u(p) for p in self.nodes])
        U, V = vectors[:, 0], vectors[:, 1]
        
        Q = ax.quiver(X, Y, U, V, scale=12, angles='xy', scale_units='xy')
        ax.quiverkey(Q, X=0.3, Y=0.97, U=1, coordinates='figure',
                     label='Quiver key, length = 1', labelpos='W')
        
        ax.triplot(X, Y, self.triang, alpha=0.3, zorder=-10)        
        ax.set_xlabel("X")
        ax.set_ylabel("Y", rotation=0)

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95)
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
     
    def basis_func_eps_vec(self, i, d, J_inv_T=None, k=None):
        """
        Calculate the epsilon-vector for a basis-function:\n
            ε(ϕ) = [∂ϕ₁/∂x, ∂ϕ₂/∂y, ∂ϕ₁/∂y + ∂ϕ₂/∂x],
            ϕ = [ϕ₁, ϕ₂] = [(1-d)*ϕ_i, d*ϕ_i].\n
        Based on transforming derivatives from the reference element.\n
        i: Local basis function index. i = 0, 1, 2.\n
        d: x- or y-component basis vector. {x: 0, y: 1}\n
        J_inv_T: Inverse Jacobian: [∂(r,s)/∂(x, y)].T\n
        k: Element number.
        """
        # Slow extra checking. Might be removed. 
        if J_inv_T is not None:
            pass
        elif J_inv_T is None and type(k) is int:
            J_inv_T = la.inv(self.generate_jacobian(k)).T
        else:
            raise ValueError("Either the inverse-Jacobian transpose 'J_int_T' or element number 'k' must be given.")

        gradient = J_inv_T @ self.reference_gradients[i]

        eps_xx = (1 - d)*gradient[0]
        eps_yy = d*gradient[1]
        eps_xy = d*gradient[0] + (1-d)*gradient[1]
        
        return np.array([eps_xx, eps_yy, eps_xy])
     
    def A_i_j(self, i_loc, j_loc, d_i, d_j, A_k, J_inv_T):
        """
        Function calculating a (Aₕ)ᵢ,ⱼ-th contribution to the "Stiffness"-matrix.
        i_loc: Local index of basis function. [0, 1, 2]
        j_loc: Local index of basis function. [0, 1, 2]
        d_i, d_j: Components of the vector function ϕ_i_d = [(1-d)*ϕ_i, d*ϕ_i]
        A_k: Area of the element: |Jₖ|/2
        J_inv_T: Inverse Jacobian: [∂(r,s)/∂(x, y)].T
        """
        eps_i = self.basis_func_eps_vec(i_loc, d_i, J_inv_T)        
        eps_j = self.basis_func_eps_vec(j_loc, d_j, J_inv_T)        
        # Could do: eps_i.T @ self.C @ eps_j, but '.T' does nothing for 1-D array.
        return A_k * (eps_i @ self.C @ eps_j)

    def generate_A_h(self):
        # Hopefully updated correctly!
        """
        Generate the Stiffness Matrix A_h, based on linear Langrange basis functions on triangles.
        """
        self.A_h = sp.dok_matrix((self.num_basis_functions, self.num_basis_functions))
        d_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))

        # Loop through elements (triangles):        
        for k, element in enumerate(self.triang):
            # Six basis functions per element. 3 nodes, 2 functions per node.
            # (3x2)^2 = 36 interactions per element k.
            # Only 6*(6+1)/2 = 21 unique interactions due to symmetry, but
            # for simplicity in code we perform 24 calculations.

            J = self.generate_jacobian(k)
            J_inv_T = la.inv(J).T  # KRITISK FUCKINGS '.T'!
            element_area = 0.5*la.det(J)
            
            # Exploit symmetry of the (A_h)_sub-matrix about i=j.
            # Only compute the upper-triangular part i <= j.
            for i_loc, node_i in enumerate(element):
                
                # Do every vector-component combination. 
                for (d_i, d_j) in d_pairs:
                    A_i_di_j_dj = self.A_i_j(i_loc=i_loc, j_loc=i_loc, d_i=d_i, d_j=d_j, 
                                             A_k=element_area, J_inv_T=J_inv_T)
                    self.A_h[2*node_i + d_i, 2*node_i + d_j] += A_i_di_j_dj

                for j_loc in range(i_loc+1, 3):
                    node_j = element[j_loc]
                    
                    for (d_i, d_j) in d_pairs:
                        A_i_di_j_dj = self.A_i_j(i_loc=i_loc, j_loc=j_loc, d_i=d_i, d_j=d_j, 
                                                 A_k=element_area, J_inv_T=J_inv_T)
                        self.A_h[2*node_i + d_i, 2*node_j + d_j] += A_i_di_j_dj
                        self.A_h[2*node_j + d_j, 2*node_i + d_i] += A_i_di_j_dj
        
        # Convert A_h to csr-format for ease of calculations later:
        self.A_h = self.A_h.tocsr() 
    
    def M_i_j(self, i_loc, j_loc, d_i, d_j, det_J_k):
        """
        Function calculating a (Mₕ)ᵢ,ⱼ-th contribution to the "Stiffness"-matrix. \n
        i_loc: Local index of basis function. [0, 1, 2] \n
        j_loc: Local index of basis function. [0, 1, 2] \n
        d_i, d_j: Components of the vector function ϕ_i_d = [(1-d)*ϕ_i, d*ϕ_i]  \n
        det_J_k: Determinant of the jacobian for element k.
        """
        # Only 36 interactions. 21 unique. Can calculate the 21 elements beforehand, and 
        # insert the values det_J_k*M_ref for each element.

        # d_i, d_j = [0, 1]:
        # Dot product (ϕ_i)^T ⋅ ϕ_j = (1 - d_i)*(1 - d_j)ϕ_i*ϕ_j + (d_i)*(d_j)ϕ_i*ϕ_j.
        if d_i != d_j:
            return 0.0
        phi_i, phi_j = self.basis_functions[i_loc], self.basis_functions[j_loc]

        # As we integrate over the reference triangle, 
        # we can use the basis function definitions directly:
        integrand = lambda p: phi_i(p)*phi_j(p)
        p1, p2, p3 = self.reference_triangle_nodes
        I = quadrature2D(integrand, p1, p2, p3, Nq=self.quad_points)
        return self.rho*I*det_J_k

    def generate_M_ref(self):
        # 36 interactions.
        M_ref = np.zeros((6, 6), dtype=float)
        local_indices = (0, 1, 2)
        d_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
        
        for i, j, (d_i, d_j) in iter_product(local_indices, local_indices, d_pairs):
            row, col = 2*i + d_i, 2*j + d_j
            # Calculate the mass matrix on the reference element: det_J_k = 1.0
            M_ref[row, col] = self.M_i_j(i, j, d_i, d_j, det_J_k=1.0)

        return M_ref

    def generate_M_h(self):
        
        d_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
        M_ref = self.generate_M_ref()

        self.M_h = sp.dok_matrix((self.num_basis_functions, self.num_basis_functions))

        # Generate the mass matrix.
        for k, element in enumerate(self.triang):
            
            # TODO: Avoid generating all Jacobians twice for A_h and M_h.
            J = self.generate_jacobian(k)
            det_J_k = la.det(J)
            
            # Exploit symmetry of the (A_h)_sub-matrix about i=j.
            # Only compute the upper-triangular part i <= j.
    
            for i_loc, node_i in enumerate(element):
                # All values scaled by det_J_k compared to M_ref: 
                
                for (d_i, d_j) in d_pairs:
                    i_i_value = det_J_k*M_ref[2*i_loc + d_i, 2*i_loc + d_j]
                    self.M_h[2*node_i + d_i, 2*node_i + d_j] += i_i_value

                for j_loc in range(i_loc+1, 3):
                    node_j = element[j_loc]
                    
                    for (d_i, d_j) in d_pairs:
                        i_j_value = det_J_k*M_ref[2*i_loc + d_i, 2*j_loc + d_j]

                        self.M_h[2*node_i + d_i, 2*node_j + d_j] += i_j_value
                        self.M_h[2*node_j + d_j, 2*node_i + d_i] += i_j_value

        # Convert A_h to csr-format for ease of calculations later:
        self.M_h = self.M_h.tocsr()

    def solve_vibration_modes(self, num=20):
        """
        Find the 'num' lowest generalized eigenvalues, along with eigenvectors. \n
            Aₕu = ω²Mₕu 
        """
        self.generate_A_h()
        self.generate_M_h()

        eigvals_small, eigvecs_small = spla.eigsh(A=self.A_h, M=self.M_h, 
                                                  k=num, which="LM", sigma=0.0)
        self.num_eigenpairs = num
        self.vibration_frequencies = eigvals_small
        self.vibration_eigenvectors = eigvecs_small        

    def display_vibration_mode(self, k):
        if k > self.num_eigenpairs - 1:
            raise ValueError(f"Too high an eigen-number. Have only solved for {self.num_eigenpairs} eigenpairs.")
        
        # Eigenvectors stored column-wise:
        vibration_eigenvec = self.vibration_eigenvectors[:, k]
        
        displacement_vec = np.zeros(self.nodes.shape)
        
        for n, d in iter_product(range(self.num_nodes), (0, 1)):
            displacement_vec[n, d] = vibration_eigenvec[2*n + d]
        
        return self.display_mesh(displacement=displacement_vec)

    def animate_vibration_mode(self, k, alpha=1, l=1, show=None, savename=None, playtime=5, fps=60, repeat_delay=0):
        if k > self.num_eigenpairs - 1:
            raise ValueError(f"Too high an eigen-number. Have only solved for {self.num_eigenpairs} eigenpairs.")

        from matplotlib.animation import ArtistAnimation

        vibration_eigenvec = self.vibration_eigenvectors[:, k]
        
        displacement_vec = np.zeros(self.nodes.shape)
        
        for n, d in iter_product(range(self.num_nodes), (0, 1)):
            displacement_vec[n, d] = vibration_eigenvec[2*n + d]

        N_frames = playtime * fps
        ts = np.linspace(0, 2*np.pi, N_frames)
        disp_vecs = [alpha * np.sin(l*t) * displacement_vec for t in ts]

        fig, ax = plt.subplots()

        artists = [self.display_mesh(displacement=disp_vecs[i], show=False, ax=ax) for i in range(N_frames)]

        ani = ArtistAnimation(fig, artists, interval=1000//fps, repeat_delay=repeat_delay, repeat=True, blit=True)

        if savename is not None:
            ani.save(f"{savename}.mp4")

        if show is None:
            if savename is None:
                show = True
            else:
                show = False
        
        if show:
            plt.show()

    def animate_vibration_mode_stress(self, k, alpha=1, l=1, show=None, savename=None, playtime=5, fps=60, repeat_delay=0):






        return


    def generate_F_h(self):
        # TODO: UPDATE
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
        # TODO: UPDATE
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
        # TODO: UPDATE
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
        # TODO: UPDATE
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
        # TODO: UPDATE
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
        # TODO: UPDATE
        self.generate_A_h()
        self.generate_F_h()
        self.apply_big_number_dirichlet()
        self.u_h = sp.linalg.spsolve(self.A_h, self.F_h)

    def solve_direct_dirichlet(self):
        # TODO: UPDATE
        self.generate_A_h()
        self.generate_F_h()
        self.apply_direct_dirichlet()
        
        reduced_u_h = sp.linalg.spsolve(self.A_h, self.F_h)
        self.u_h = np.zeros(self.num_nodes)
        self.u_h[~self.dirichlet_BC_mask] = reduced_u_h
        self.u_h[self.dirichlet_BC_mask] = self.dirichlet_BC_values

    def solve(self):
        # TODO: UPDATE
        self.generate_A_h()
        self.generate_F_h()
        self.apply_boundary_conditions()

        reduced_u_h = sp.linalg.spsolve(self.A_h, self.F_h)
        self.u_h = np.zeros(self.num_nodes)
        self.u_h[~self.dirichlet_BC_mask] = reduced_u_h
        self.u_h[self.dirichlet_BC_mask] = self.dirichlet_BC_values

    def solve_direct_dirichlet_CG(self, TOL=1e-5):
        # TODO: UPDATE
        self.generate_A_h()
        self.generate_F_h()
        self.apply_direct_dirichlet()
        
        reduced_u_h, exit_code = sp.linalg.cg(self.A_h, self.F_h, tol=TOL)
        assert exit_code == 0 
        self.u_h = np.zeros(self.num_nodes)
        self.u_h[~self.dirichlet_BC_mask] = reduced_u_h
        self.u_h[self.dirichlet_BC_mask] = self.dirichlet_BC_values

    def error_est(self, u_ex, quad_points=None):
        # TODO: UPDATE
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

    def evaluate(self, p):
        """
        Some smart generator function returning sum of basis functions at the point [x, y],
        located in some element K.
        p: np.array([x, y])
        """

        raise NotImplementedError

