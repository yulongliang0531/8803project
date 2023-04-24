import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.optimize import Optimizer
from desc.objectives import (
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
    ForceBalance,
)

import desc.plotting as dplot
import matplotlib.pyplot as plt

"""
DESC requires 4 inputs to define an equilibrium problem:
    - Pressure profile
    - Rotational Transform profile (Geometry?)
    - Last Closed Flux Surface (LCFS) Boundary Shape
    - Total toroidal magnetic flux enclosed by the LCFS
"""

if __name__ == "__main__":
    # Define the fixed boundary
    surface = FourierRZToroidalSurface(
        R_lmn=[10, 1],  # Fourier coefficients for R in cylindrical coordinates
        modes_R=[[0, 0], [1, 0]],  # modes given as [m,n] for each coefficient
        Z_lmn=[0, -1],  # Fourier coefficients for Z in cylindrical coordinates
        modes_Z=[[0, 0], [-1, 0]],
    )

    # Define pressure profile: p(rho) = 0
    pressure = PowerSeriesProfile(params=[0, 0], modes=[0, 2])

    # Define the rotational transform profile
    iota = PowerSeriesProfile(params=[1, 1.5], modes=[0, 2])

    # Define the equilibrium
    eq = Equilibrium(
        surface=surface,
        pressure=pressure,
        iota=iota,
        Psi=1.0,  # flux (in Webers) within the last closed flux surface
        NFP=1,  # number of field periods
        L=7,  # radial spectral resolution
        M=7,  # poloidal spectral resolution
        N=0,  # toroidal spectral resolution (axisymmetric case, so we don't need any toroidal modes)
        L_grid=12,  # real space radial resolution, slightly oversampled
        M_grid=12,  # real space poloidal resolution, slightly oversampled
        N_grid=0,  # real space toroidal resolution (axisymmetric, so we don't need any grid points toroidally)
        sym=True,  # explicitly enforce stellarator symmetry
    )

    # Plot the force balance error before solve
    fig, axs = plt.subplots(nrows=2, ncols=3)
    dplot.plot_surfaces(eq=eq, ax=axs[0][0])
    dplot.plot_section(eq=eq, name="|F|", norm_F=True, log=True, ax=axs[1][0])

    # Solve the force equilibrium by minimizing the error
    optimizer = Optimizer("lsq-exact")
    constraints = (
        FixBoundaryR(),  # enforce fixed  LCFS for R
        FixBoundaryZ(),  # enforce fixed  LCFS for R
        FixPressure(),  # enforce that the pressure profile stay fixed
        FixIota(),  # enforce that the rotational transform profile stay fixed
        FixPsi(),  # enforce that the enclosed toroidal stay fixed
    )
    objectives = ForceBalance()
    obj = ObjectiveFunction(objectives=objectives)
    eq.solve(
        verbose=3,
        ftol=1e-8,
        maxiter=100,
        constraints=constraints,
        optimizer=optimizer,
        objective=obj,
    )

    #  Plot the force balance error after solve
    dplot.plot_surfaces(eq=eq, ax=axs[0][1])
    dplot.plot_section(eq=eq, name="|F|", norm_F=True, log=True, ax=axs[1][1])

    # Now we perturb the pressure profile by a bit and rerun simulation
    delta_p = np.zeros_like(eq.p_l)
    delta_p[0] = 1000.0
    delta_p[1] = -1000.0
    eqp = eq.perturb(
        deltas={"p_l": delta_p}, order=2, objective=obj, constraints=constraints
    )
    eqp.solve(
        verbose=3,
        ftol=1e-8,
        maxiter=100,
        constraints=constraints,
        optimizer=optimizer,
        objective=obj,
    )
    dplot.plot_surfaces(eq=eqp, ax=axs[0][2])
    dplot.plot_section(eq=eqp, name="|F|", norm_F=True, log=True, ax=axs[1][2])

    plt.show()
