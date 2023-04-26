import argparse
import desc.io
import desc.plotting as dplot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    eqs = desc.io.load("/Users/fyc/git/DESC/desc/examples/W7-X_output.h5")
    eq = eqs[-1]

    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0))
    dplot.plot_section(eq, "p", nzeta=2, ax=axs)
    fig.savefig("figures/w7_pressure_sections.pdf")

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    dplot.plot_1d(eq, "p", ax=ax)
    fig.savefig("figures/w7_pressure_profile.pdf")

    fig, axs = plt.subplots(figsize=(9.0, 5.0), ncols=3, nrows=2)
    dplot.plot_section(eq, "|F|", nzeta=6, ax=axs)
    fig.savefig("figures/w7_F_sections.pdf")
