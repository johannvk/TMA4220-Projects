from .project_1.tasks import run_tasks as run_task_1
from .project_1.tests import test_main
from .project_2.tasks import test_elasticity_solver


def main():
    print("The best FEM package there is!")
    # run_task_1()
    # test_main()
    test_elasticity_solver(N=50, area="plate")
    pass

if __name__ == "__main__":
    main()
