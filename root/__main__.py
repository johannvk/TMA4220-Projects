from .project_1.tests import test_main as proj_1_test_main
from .project_2.tests import test_main as proj_2_test_main

from .project_1.tasks import run_tasks as run_tasks_1
from .project_2.tasks import run_tasks as run_tasks_2


def run_tests():
    print("\nProject 1 Tests:")
    proj_1_test_main()

    print("\n\nProject 2 Tests:")
    proj_2_test_main()


def main():
    print("The best FEM package there is!")
    
    # Un-comment the below line to run a barrage of tests:
    # run_tests()

    # Project 1 Tasks:
    run_tasks_1()

    # Project 2 Tasks:
    run_tasks_2()    


if __name__ == "__main__":
    main()
