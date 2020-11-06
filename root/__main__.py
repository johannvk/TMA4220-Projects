from .project_1.tasks import run_tasks as run_task_1
from .project_1.tests import test_main
from .project_2.tasks import test_elasticity_solver, test_animation, test_animate_mesh_stress, \
                             save_material_vibration_modes


def main():
    print("The best FEM package there is!")
    # run_task_1()
    # test_main()
    # test_elasticity_solver(N=10, area="plate")
    # test_elasticity_solver(N=25, area="plate")
    # test_animation(mode=10)
    # steel_animation(N=8, area="plate", mode=9, max_modes=10)
    # timber_animation(N=8, area="plate", mode=9, max_modes=50)
    # test_markov(N=10, area="plate")
    # test_animate_mesh_stress(N=10)
    save_material_vibration_modes(N=6, k_min=3, k_max=4)
    pass


if __name__ == "__main__":
    main()
