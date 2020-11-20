README - TMA4220 Projects

Code is run as a module by calling
>>> python -m root
in the directory where the `root` folder lies. This calls the "main()" method in 
"__main__.py".

The source code for the 2D Poisson solver is located in "root/project_1/fem_2d_solver.py".
The solver is implemented in the class "Poisson2DSolver()", which stores the requisite
parameters and has the required functionality for constructing things such as stiffness
matrixes and applying boundary conditions.

The source code for the 2D Elasticity solver is located in "root/project_1/elasticity_solver.py".
Here all the functionality for solving the 2D elasticy equation with dirichlet conditions is implemented,
along with the functionality for finding eigen vibration modes and frequencies over a plate or circular domain.

Tested on:
    python 3.7.0
    numpy 1.18.2
    scipy 1.1.0
    matplotlib 3.1.1
