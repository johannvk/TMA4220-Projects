from .project_1.tasks import run_tasks as run_tasks_1
from .project_1.tests import test_main
from .project_2.tasks import save_material_vibration_modes

from .project_2.tests import test_elasticity_solver, test_animation, test_animate_mesh_stress, \
                             test_markov, test_full_solver, test_mosaic, test_show_frequencies

def main():
    print("The best FEM package there is!")
    # run_tasks_1()
    # test_main()
    # test_elasticity_solver(N=10, area="plate")
    # test_elasticity_solver(N=25, area="plate")
    # test_animation(mode=10)
    # test_markov(N=10, area="plate")
    # test_animate_mesh_stress(N=10)
    # save_material_vibration_modes(N=6, k_min=3, k_max=4)
    # test_full_solver(N=12)
    # test_mosaic(N=10, k=5, dims=(3,3), alpha=0.5, savename="mosaictest", dpi=200)
    test_show_frequencies(N=20, num=20, area="disc")

    pass


if __name__ == "__main__":
    main()
