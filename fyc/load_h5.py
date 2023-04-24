import desc.io
import desc.plotting as dplot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Load result
    eq_fam = desc.io.load("/Users/fyc/git/DESC/desc/examples/DSHAPE_CURRENT_output.h5")

    # # Plot
    # neqs = len(eq_fam)
    # fig, axs = plt.subplots(nrows=2, ncols=neqs, figsize=(6.4 * neqs, 9.6))
    # for i in range(neqs):
    #     dplot.plot_1d(eq=eq_fam[i], name="p", ax=axs[0][i])
    #     dplot.plot_section(
    #         eq=eq_fam[i], name="|F|", norm_F=True, log=True, ax=axs[1][i], nzeta=1
    #     )
    #     axs[0][i].set_title("step %d" % i)
    #     axs[1][i].set_title("step %d" % i)

    # dplot.plot_3d(eq=eq_fam[-1], name="|F|")
    # plt.show()

    eq_baseline = eq_fam[-1]

    # Perturb pressure distributions
    N = 20
    np.random.seed(0)

    fig, ax = plt.subplots()

    dplot.plot_1d(eq_baseline, "p", ax=ax, linecolor="red", label="baseline")

    for i in tqdm(range(N)):
        delta_p = np.zeros_like(eq_baseline.p_l)
        nnz = np.nonzero(eq_baseline.p_l)
        delta_p[nnz] = 0.1 * (np.random.rand(len(nnz)) - 0.5) * eq_baseline.p_l[nnz]
        eq_perturb = eq_baseline.perturb(deltas={"p_l": delta_p})
        # eq_perturb.solve(verbose=3, ftol=1e-8, maxiter=100)

        dplot.plot_1d(eq_perturb, "p", ax=ax, linecolor="blue", label="perturbed")

    ax.legend()
    plt.show()
