from matplotlib.animation import FuncAnimation, ArtistAnimation, writers

from .elasticity_solver import *





def test_animation(N=3, area="plate"):

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=10)


    fig, ax = plt.subplots()

    artists = [solver.display_vibration_mode(k=4, show=False, ax=ax), solver.display_vibration_mode(k=5, show=False, ax=ax)]

    ani = ArtistAnimation(fig, artists, interval=500, repeat_delay=500, repeat=True, blit=True)


    #foo = plt.plot([0,1], [0,1])
    #print(foo)

    #ani.save()
    plt.show()
    

    return
