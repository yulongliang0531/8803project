import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import desc.plotting as dplot
import matplotlib.pyplot as plt
import desc.io
import os
import argparse


def solve_baseline_equilibrium(p_modes, p_coefs, verbose=3):
    """
    Perform HMD computation for a simple Tokamak configuration

    Args:
    p_modes: powers of the polynomial bases for pressure profile
    p_coefs: coefficients of the polynomial bases for pressure profile
    """

    assert len(p_modes) == len(p_coefs)

    eq_bl = Equilibrium(
        surface=FourierRZToroidalSurface(
            R_lmn=[10, 1],
            Z_lmn=[0, -1],
            modes_R=[[0, 0], [1, 0]],
            modes_Z=[[0, 0], [-1, 0]],
        ),
        pressure=PowerSeriesProfile(params=p_coefs, modes=p_modes),
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


def even_poly_fit(x, y, modes):
    """
    Fit a polynomial with only even power terms:
        p(x) = c0 + c[2]*x**2 + .. + c[deg] * x**deg

    Args:
        x: (M,), x coordinates
        y: (M, K), y coordinates
        modes: polynomial exponents, e.g. [0, 2, 4, 6, 8]

    Return:
        coefs: (len(modes), K), coefficient matrix
    """
    M = len(x)
    assert len(x) == y.shape[0]

    # Create matrix
    Mat = np.zeros((M, len(modes)))
    for i in modes:
        Mat[:, i // 2] = x**i

    # Solve the least-square problem
    coefs, _, _, _ = np.linalg.lstsq(Mat, y, rcond=None)
    return coefs


def perturb_poly_coef(nsamples, modes, upper=100.0, lower=-100.0, seed=0):
    """
    Generate a batch of perturbed polynomials by directly perturbing each coefficient
    Args:
        nsamples: number of perturbations
        modes: polynomial exponents, e.g. [0, 2, 4, 6, 8]
        upper: upper bound for each coefficient
        lower: lower bound for each coefficient
        seed: random seed

    Return:
        dcoefs: perturbation coefficients, (nsamples, deg + 1)
    """
    # Perturb pressure profiles
    np.random.seed(seed)

    dcoefs = np.random.uniform(lower, upper, size=(nsamples, len(modes)))
    dcoefs[:, 0] -= np.sum(dcoefs, axis=1)  # Such that p(1.0) = 0.0
    return dcoefs


def sample_gp(X, cov=1.0, n_samples=1, seed=0):
    kernel = RBF(length_scale=cov, length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(kernel)
    Y = gp.sample_y(X, n_samples=n_samples, random_state=seed)
    return Y


def perturb_poly_gp(nsamples, modes, length_scale=0.1, stddev=100.0, seed=0):
    """
    Generate a batch of perturbed polynomials using gaussian process

    Args:
        nsamples: number of perturbations
        modes: powers of coefficient
        length_scale: controls the smoothness of the perturbation
        stddev: controls the magnitude of the perturbation
        seed: random seed

    Return:
        dcoefs: perturbation coefficients, (nsamples, len(modes))
    """
    # We use 100 points to fit the Gaussian process samples
    X = np.linspace(0.0, 1.0, 100).reshape(-1, 1)

    Y = stddev * sample_gp(X, cov=length_scale, n_samples=nsamples, seed=seed)
    dcoefs = even_poly_fit(X.flatten(), Y, modes=modes).T
    dcoefs[:, 0] -= np.sum(dcoefs, axis=1)  # Such that p(1.0) = 0.0
    return dcoefs


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prefix", default="results", type=str, help="result folder")
    p.add_argument(
        "--num-perturbs",
        default=1000,
        type=int,
        help="number of samples by perturbation",
    )
    p.add_argument(
        "--seed", default=0, type=int, help="random seed for reproducibility"
    )
    p.add_argument(
        "--degree",
        default=10,
        type=int,
        help="polynomial degree for pressure profile, deg=1 is linear, deg=2 is quadratic, etc.",
    )
    p.add_argument(
        "--scale",
        default=20.0,
        type=float,
        help="absolute scaling factor that controls the magnitude of the perturbation of pressure profile",
    )
    p.add_argument(
        "--method",
        type=str,
        default="gp",
        choices=["direct", "gp"],
        help="how to generate functional perturbation of the pressure profile",
    )
    p.add_argument(
        "--gp-length-scale",
        type=float,
        default=0.1,
        help="length scale of Gaussian process samples",
    )
    p.add_argument("--plot-perturb-and-exit", action="store_true")
    args = p.parse_args()

    if args.degree < 2:
        print("[FATAL] --poly-degree must be at least 2, exiting...")
        exit(-1)

    # Create folder
    if os.path.isdir(args.prefix):
        print("[FATAL] folder %s exists, exiting..." % args.prefix)
        exit(-1)
    else:
        os.mkdir(args.prefix)

    # Save option values
    with open(os.path.join(args.prefix, "options.txt"), "w") as f:
        f.write("Options:\n")
        for k, v in vars(args).items():
            f.write(f"{k:<20}{v}\n")

    # Define the baseline pressure profile
    p_modes_bl = list(range(0, args.degree + 1, 2))  # 0, 2, 4, ..., deg
    p_coefs_bl = np.zeros_like(p_modes_bl)
    p_coefs_bl[0] = 1000.0
    p_coefs_bl[1] = -1000.0

    # Generate Perturbed polynomials
    if args.method == "direct":
        dcoefs = perturb_poly_coef(
            nsamples=args.num_perturbs,
            modes=p_modes_bl,
            upper=args.scale,
            lower=-args.scale,
            seed=args.seed,
        )
    else:  # args.method == "gp"
        dcoefs = perturb_poly_gp(
            nsamples=args.num_perturbs,
            modes=p_modes_bl,
            length_scale=args.gp_length_scale,
            stddev=args.scale,
            seed=args.seed,
        )

    with open(os.path.join(args.prefix, "dcoefs.npy"), "wb") as f:
        np.save(f, dcoefs)

    if args.plot_perturb_and_exit:
        nsamples = dcoefs.shape[0]
        fig, axs = plt.subplots(ncols=3, figsize=(9.6, 3.2))
        x = np.linspace(0.0, 1.0, 100)
        poly_bl = Polynomial(p_coefs_bl, p_modes_bl)

        axs[0].plot(x, poly_bl.eval(x), color="red", zorder=100, label="baseline")
        for i in range(nsamples):
            poly_pt = Polynomial(p_coefs_bl + dcoefs[i, :], p_modes_bl)
            axs[0].plot(x, poly_pt.eval(x), color="blue", alpha=0.3, zorder=10)
        axs[0].legend()
        axs[0].set_title("(a) Pressure profiles")
        axs[0].set_ylabel(r"$p(\rho)$")

        vals = np.zeros((nsamples, 100))
        for i in range(nsamples):
            poly_pt = Polynomial(dcoefs[i, :], p_modes_bl)
            vals[i, :] = poly_pt.eval(x)
            axs[1].plot(x, vals[i, :], color="blue", alpha=0.3, zorder=10)
        axs[1].set_title("(b) Pressure profile perturbations")
        axs[1].set_ylabel(r"$\Delta p(\rho)$")

        sigma = np.std(vals, axis=0)
        mean = np.mean(vals, axis=0)
        axt = axs[2].twinx()
        axs[2].plot(x, sigma, ".", color="blue", label="std. dev.")
        axt.plot(x, mean, ".", color="red", label="mean")
        axs[2].set_xlabel(r"$\rho$")
        axs[2].set_ylabel("std. dev. (max: %.2f)" % (sigma.max()))
        axt.set_ylabel("mean (max: %.2f)" % (mean.max()))
        axt.spines[["right"]].set_visible(True)
        axs[2].legend(loc="upper right")
        axt.legend(loc="lower left")
        axs[2].set_title("(c) Sample mean and median")

        for i in range(2):
            axs[i].set_xlabel(r"$\rho$")

        fig.savefig(os.path.join(args.prefix, "perturbations.pdf"))
        exit(0)

    # Solve the baseline equilibrium
    eq = solve_baseline_equilibrium(p_modes=p_modes_bl, p_coefs=p_coefs_bl, verbose=3)

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

        # Write to pkl
        pw = desc.io.PickleWriter(target=os.path.join(eqdir, "eq_%d.pkl" % i))
        pw.write_obj(ret)
