import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile

import desc.plotting as dplot
import matplotlib.pyplot as plt
import desc.io
import os
import argparse


def solve_baseline_equilibrium(verbose=3):
    """
    Perform HMD computation for a simple Tokamak configuration
    """
    eq_bl = Equilibrium(
        surface=FourierRZToroidalSurface(
            R_lmn=[10, 1],
            Z_lmn=[0, -1],
            modes_R=[[0, 0], [1, 0]],
            modes_Z=[[0, 0], [-1, 0]],
        ),
        pressure=PowerSeriesProfile(
            params=[1000, -1000, 0.0, 0.0, 0.0, 0.0], modes=[0, 2, 4, 6, 8, 10]
        ),
        iota=PowerSeriesProfile(params=[1, 1.5], modes=[0, 2]),
        Psi=1.0,  # flux (in Webers) within the last closed flux surface
        NFP=1,  # number of field periods
        L=7,  # radial spectral resolution
        M=7,  # poloidal spectral resolution
        N=0,  # toroidal spectral resolution
        L_grid=12,  # real space radial resolution, slightly oversampled
        M_grid=12,  # real space poloidal resolution, slightly oversampled
        N_grid=0,  # real space toroidal resolution
        sym=True,  # explicitly enforce stellarator symmetry
    )

    eq_bl.solve(
        verbose=verbose,
        ftol=1e-8,
        maxiter=100,
        objective="force",
        optimizer="lsq-exact",
    )

    return eq_bl


def test_direct_perturb():
    # Solve for the baseline equilibrium
    eq_bl = solve_baseline_equilibrium(verbose=3)

    # Create figure
    fig, ax = plt.subplots()

    # Plot pressure baseline pressure profile
    dplot.plot_1d(eq_bl, "p", ax=ax, linecolor="red", label="baseline")

    # Generate random perturbed pressure profiles
    np.random.seed(0)
    N = 10
    for i in range(N):
        delta_p = np.zeros_like(eq_bl.p_l)
        nnz = np.nonzero(eq_bl.p_l)
        delta_p[nnz] = 0.1 * (np.random.rand(len(nnz)) - 0.5) * eq_bl.p_l[nnz]
        eq_perturb = eq_bl.perturb(deltas={"p_l": delta_p})
        # eq_perturb.solve(verbose=3, ftol=1e-8, maxiter=100)

        dplot.plot_1d(eq_perturb, "p", ax=ax, linecolor="blue", label="perturbed")

    plt.show()
    return


def test_gaussian_process_perturb():
    return


def test_load_save():
    eq_bl = solve_baseline_equilibrium(verbose=3)
    pw = desc.io.PickleWriter(target="eq_bl.pkl")
    pw.write_obj(eq_bl)

    eq_loaded = desc.io.load("eq_bl.pkl")
    print(eq_loaded)
    print(type(eq_loaded))

    dplot.plot_section(eq=eq_loaded, name="|F|", norm_F=True, log=True)
    plt.show()

    return


class Polynomial:
    def __init__(self, coefs, modes) -> None:
        self.coefs = coefs
        self.modes = modes
        assert len(self.coefs) == len(self.modes)

    def eval(self, x):
        val = np.zeros_like(x)
        for coef, power in zip(self.coefs, self.modes):
            val += coef * x**power
        return val


def test_polynomial():
    coefs = np.array([1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    modes = np.array([0, 2, 4, 6, 8, 10])

    p = Polynomial(coefs, modes)

    x = np.linspace(0, 1, 100)
    plt.plot(x, p.eval(x), color="red")

    np.random.seed(0)
    for i in range(20):
        dcoefs = (
            0.1 * np.max(coefs) * np.random.uniform(low=-1.0, high=1.0, size=len(coefs))
        )
        dcoefs[0] -= np.sum(dcoefs)
        px = Polynomial(coefs + dcoefs, modes)
        plt.plot(x, px.eval(x), color="blue", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results", type=str)
    p.add_argument("--num-perturbs", default=500, type=int)
    p.add_argument("--random-seed", default=0, type=int)
    args = p.parse_args()

    # Create folder
    if os.path.isdir(args.prefix):
        print("folder %s exists, exiting..." % args.prefix)
        exit(-1)
    else:
        os.mkdir(args.prefix)

    # Save option values
    with open(os.path.join(args.prefix, "options.txt"), "w") as f:
        f.write("Options:\n")
        for k, v in vars(args).items():
            f.write(f"{k:<20}{v}\n")

    # Perturb pressure profiles
    np.random.seed(args.random_seed)
    modes = np.array([0, 2, 4, 6, 8, 10])
    scale = 100.0
    dcoefs = scale * np.random.uniform(-1.0, 1.0, size=(args.num_perturbs, 6))
    dcoefs[:, 0] -= np.sum(dcoefs, axis=1)

    # Save each perturbation
    with open(os.path.join(args.prefix, "dcoefs.npy"), "wb") as f:
        np.save(f, dcoefs)

    # Solve baseline polynomial
    eq = solve_baseline_equilibrium()

    # Perturb and save equilibriums
    eqdir = os.path.join(args.prefix, "eqs")
    if not os.path.isdir(eqdir):
        os.mkdir(eqdir)

    # Run cases
    for i in range(args.num_perturbs):
        eqp = eq.perturb(deltas={"p_l": dcoefs[i]}, copy=True)
        ret = eqp.solve(
            verbose=3,
            ftol=1e-8,
            maxiter=100,
            objective="force",
            optimizer="lsq-exact",
            copy=False,
        )

        # Save the equilibrium

        # Write to pkl
        pw = desc.io.PickleWriter(target=os.path.join(eqdir, "eq_%d.pkl" % i))
        pw.write_obj(ret)
