from matplotlib.animation import FuncAnimation, ArtistAnimation, writers

from .elasticity_solver import *





def test_animation(N=10, area="plate"):

    model_dict = {"N": N, "f": lambda p: 0.0, "g_D": lambda _: True, "g_N": lambda _: False,
                  "class_BC": 12.0, "E": 12.0, "nu": 0.22, "rho": 1.0, "area": area}
    solver = Elasticity2DSolver.from_dict(model_dict)
    solver.solve_vibration_modes(num=10)
    
    k = 9

    vibration_eigenvec = solver.vibration_eigenvectors[:, k]
        
    displacement_vec = np.zeros(solver.nodes.shape)
    
    for n, d in iter_product(range(solver.num_nodes), (0, 1)):
        displacement_vec[n, d] = vibration_eigenvec[2*n + d]

    alpha = 0.5
    N_frames = 100
    ts = np.linspace(0, 2*np.pi, N_frames)
    disp_vecs = [alpha * np.sin(t) * displacement_vec for t in ts]

    fig, ax = plt.subplots()

    artists = [solver.display_mesh(displacement=disp_vecs[i], show=False, ax=ax) for i in range(N_frames)]

    ani = ArtistAnimation(fig, artists, interval=50, repeat_delay=500, repeat=True, blit=True)


    #ani.save()
    plt.show()
    

    return
