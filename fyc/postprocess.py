import desc.io
import desc.plotting as dplot
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import numpy as np


def compile_pkl(pkls, out_pkl):
    print("Compiling all pkls into a single pkl")
    res = []
    for pkl in tqdm(pkls):
        res.append(desc.io.load(pkl))

    pw = desc.io.PickleWriter(target=out_pkl)
    pw.write_obj(res)
    return


if __name__ == "__main__":
    prefix = "results_2"
    pkls = glob(os.path.join(prefix, "eqs", "eq_*.pkl"))
    compiled_pkl = os.path.join(prefix, "equilibriums.pkl")

    if not os.path.isfile(compiled_pkl):
        compile_pkl(pkls, compiled_pkl)

    with open(compiled_pkl, "rb") as f:
        rets = pickle.load(f)

    nsucc = 0
    nfail = 0
    fig, ax = plt.subplots()

    F_norm = []

    for ret in tqdm(rets[0:20]):
        eq, res = ret
        if res.success:
            nsucc += 1
            # print(eq.solved)
            # print(eq.pressure)
            # dplot.plot_1d(eq, "p", ax=ax)
            fig, ax = dplot.plot_section(eq, "|F|", norm_F=True, log=True)
            F_norm.append(np.max(ax.data))

        else:
            nfail += 1

    print("converged cases:", nsucc)
    print("failed cases:   ", nfail)
    print("success rate:   ", nsucc / (nsucc + nfail))

    print(F_norm)
    exit()
    plt.scatter(F_norm, np.zeros(len(F_norm)))
    plt.show()
