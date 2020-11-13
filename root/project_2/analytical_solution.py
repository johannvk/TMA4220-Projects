import numpy as np
import sympy as sy

# Coodinates and function components:
x, y, u1, u2 = sy.symbols("x, y, u1, u2")

# Material properties:
E, nu = sy.symbols("E, nu")

# Analytical solution:
u1 = (x**2 - 1)*(y**2 - 1)
u2 = (x**2 - 1)*(y**2 - 1)

# Strain components:
eps_xx, eps_yy, eps_xy = sy.diff(u1, x), sy.diff(u2, y), \
                         sy.diff(u1, y) + sy.diff(u2, x)
eps_vec = sy.Matrix([eps_xx, eps_yy, eps_xy])

# Stress-Strain Transformation:
C = (E/(1 - nu**2))*sy.Matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]])

# Stress components:
sigma_xx, sigma_yy, sigma_xy = C @ eps_vec

# Nabla times Stress matrix:
grad_sigma = sy.Matrix([sy.diff(sigma_xx, x) + sy.diff(sigma_xy, y), 
                        sy.diff(sigma_xy, x) + sy.diff(sigma_yy, y)])

# Analytical Source term:
f_1 = (E/(1 - nu**2))*(-2*y**2 -x**2 + nu*x**2 -2*nu*x*y - 2*x*y + 3 - nu)
f_2 = (E/(1 - nu**2))*(-2*x**2 -y**2 + nu*y**2 -2*nu*x*y - 2*x*y + 3 - nu)
f_vec = sy.Matrix([f_1, f_2])

components_equal = sy.simplify(grad_sigma + f_vec).equals(sy.Matrix([0, 0]))

if components_equal:
    print(f"The equation ∇ᵀσ(u) = -f has an analytical solution:")
    print("\tu(x, y) = [(x^2 - 1)(y^2 - 1), (x^2 - 1)(y^2 - 1)],\ngiven source vector f(x, y) = [f_1, f_2]:")
    print(f"\tf_1 = {sy.simplify(f_1)},\n\tf_2 = {sy.simplify(f_1)}.")
