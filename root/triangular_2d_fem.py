import numpy as np
import scipy.linalg as la

from .triangulation import getPlate, GetDisc


class Triangular2DFEM():

    def __init__(self, N, area="plate"):
        """
        N: Description of the size of the mesh. N = 'number of grid points' for 'area' == 'disc', 
           or N^2 = 'number of grid points for 'area' = 'plate'.
        area:     What geometry to solve the elasticity equation on. Default: "plate".

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

        # Linear Lagrange polynomial basis functions on the reference triangle:
        self.basis_functions = (lambda eta: 1 - eta[0] - eta[1], lambda eta: eta[0], lambda eta: eta[1])

        # Gradients of basis functions in reference coordinates:
        self.reference_gradients = (np.array([-1.0, -1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

        # Reference triangle:
        self.reference_triangle_nodes = (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

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
    
    def global_to_reference_transformation(self, p, element: int, J_inv=None):
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
