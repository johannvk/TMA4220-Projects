from .project_1.tasks import run_tasks as run_task_1
from .project_1.tests import test_main
from .project_2.tasks import test_elasticity_solver, test_animation, test_animate_mesh_stress, \
                             aluminium_animation, steel_animation, timber_animation


def main():
    print("The best FEM package there is!")
    # run_task_1()
    # test_main()
    # test_elasticity_solver(N=10, area="plate")
    # test_elasticity_solver(N=25, area="plate")
    # test_animation(mode=10)
    steel_animation(N=15, area="plate", mode=9, max_modes=50)
    timber_animation(N=15, area="plate", mode=9, max_modes=50)
    pass

if __name__ == "__main__":
    main()
