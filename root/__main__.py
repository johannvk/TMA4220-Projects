from .project_1.tasks import run_tasks as run_task_1
from .project_1.tests import test_main
from .project_2.elasticity_solver import Elasticity2DSolver, test_elasticity_solver


def main():
    print("The best FEM package there is!")
    # run_task_1()
    # test_main()
    test_elasticity_solver()
    pass

if __name__ == "__main__":
    main()
