import numpy as np
import matplotlib.pyplot as plt

from triangulation.getdisc import GetDisc

def display_triangulation(N):

    p, tri, edge = GetDisc(N)

    plt.triplot(p[:, 0], p[:, 1], triangles=tri)
    plt.show()

display_triangulation(5)
