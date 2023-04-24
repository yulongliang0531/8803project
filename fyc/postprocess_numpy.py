import os
import numpy as np
from sampling import Polynomial
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prefix = "results_2"
    npy = os.path.join(prefix, "dcoefs.npy")
    dconfs = np.load(npy)

    coefs = np.array([1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    modes = np.array([0, 2, 4, 6, 8, 10])
    for dconf in dconfs:
        px = Polynomial(coefs + dconf, modes)

        x = np.linspace(0, 1, 100)
        plt.plot(x, px.eval(x), color="blue")

    plt.show()
