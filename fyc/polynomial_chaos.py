import argparse
import pandas as pd
import chaospy as cp
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


def eval_polynomial(coefs):
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results_gp", type=str)
    p.add_argument("--plot-nn", action="store_true")
    args = p.parse_args()

    # Load data frame
    assert os.path.isdir(args.prefix)
    df_path = os.path.join(args.prefix, "df.pkl")
    df = pd.read_pickle(df_path)
    df = df[df["success"] == True].reset_index(drop=True)

    input_coeffs = df.loc[:, ["p_l_%d" % i for i in range(6)]].values

    # fig, axs = plt.subplots(ncols=3, nrows=2)
    # for icol in range(input_coeffs.shape[1]):
    #     sns.histplot(input_coeffs[:, icol], kde=True, ax=axs.flatten()[icol])
    # plt.show()

    poly_order = 4
    input_dim = input_coeffs.shape[1]
    joint_dist = cp.J(cp.Normal(0, 1), input_dim)
    orthogonal_polys = cp.Distribution.ttr(poly_order, joint_dist)

    # nodes, weights = cp.generate_quadrature(poly_order, joint_dist, rule="gaussian")

    # # 4. Perform a point collocation method to evaluate the model
    # poly_order = 4
    # input_dim = y_samples.shape[1]
    # joint_dist = cp.J(cp.Normal(0, 1), input_dim)
    # orthogonal_polys = cp.orth_ttr(poly_order, joint_dist)

    # nodes, weights = cp.generate_quadrature(poly_order, joint_dist, rule="gaussian")
    # evaluated_model = np.array([computational_model(coeff) for coeff in y_samples])

    # # 5. Fit the polynomial chaos expansion
    # poly_chaos_expansion = cp.fit_quadrature(
    #     orthogonal_polys, nodes, weights, evaluated_model
    # )

    # # 6. Calculate mean, standard deviation, and Sobol sensitivity indices
    # mean = cp.E(poly_chaos_expansion, joint_dist)
    # std_dev = cp.Std(poly_chaos_expansion, joint_dist)
    # sobol_indices = cp.Sens_t(poly_chaos_expansion, joint_dist)

    # print(f"Mean: {mean}, Standard Deviation: {std_dev}")
    # print(f"Sobol Sensitivity Indices: {sobol_indices}")
