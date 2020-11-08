README - TMA4220 Projects
# TODO: UPDATE!
The source code for the Finite Element solver is located in "fem_2d_solver.py".
The solver is implemented in the class "Poisson2DSolver()", which stores the requisite
parameters and has the required functionality for constructing things such as stiffness
matrixes and applying boundary conditions.

In order to solve the problems from the project, one runs the file "tasks.py". 
There we have instantiated the solver with the different source functions and 
boundary conditions, and also display the solutions. 

Lastly, one can run the file "convergence_plots.py", which will generate the 
convergence plots for both a pure Dirichlet BC-problem, and a mixed BC-problem.
