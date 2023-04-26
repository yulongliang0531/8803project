import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, nfeatures):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(nfeatures, 256)
        self.layer2 = torch.nn.Linear(256, 256)
        self.layer3 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        y = self.layer3(x)
        return y


class NNRegressor:
    def __init__(self, nfeatures) -> None:
        self.net = Net(nfeatures)

        # Define a loss function and an optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

        return

    def train(self, maxit=1000):
        # Train the neural network
        loss_vals = []
        for i in tqdm(range(maxit)):
            running_loss = 0.0
            self.optimizer.zero_grad()
            outputs = self.net(torch.Tensor(X_train.values)).flatten()
            loss = self.criterion(outputs, torch.Tensor(Y_train.values))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            loss_vals.append(running_loss)

        return loss_vals


def plot_nn(prefix, X_train, X_test, Y_train, Y_test):
    Y_pred_test = np.zeros_like(Y_test)
    Y_pred_train = np.zeros_like(Y_train)
    for i, x in enumerate(X_test.values):
        Y_pred_test[i] = nnr.net(torch.Tensor(x))
    for i, x in enumerate(X_train.values):
        Y_pred_train[i] = nnr.net(torch.Tensor(x))

    print("R2 training:", r2_score(Y_pred_train, Y_train))
    print("R2 test    :", r2_score(Y_pred_test, Y_test))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].scatter(Y_pred_train, Y_train, color="blue", label="training set")
    axs[0].scatter(Y_pred_test, Y_test, color="red", label="test set")
    axs[0].legend()
    axs[1].plot(loss_vals)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Iteration")
    fig.savefig(os.path.join(prefix, "nn.pdf"))
    plt.close()
    return


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

    # Get input and output
    X = df.loc[:, ["p_l_%d" % i for i in range(6)]]
    Y = df["max_F"].astype(float)
    X_mean = X.mean()
    X_std = X.std()

    # Normalize X such that E(x) = 0, std(X) = 1.0
    Xn = (X - X_mean) / X_std

    # Split into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        Xn, Y, test_size=0.2, random_state=0
    )

    # Train the NN
    nfeatures = X_train.values.shape[1]
    nnr = NNRegressor(nfeatures)
    loss_vals = nnr.train()

    # Use the neural network to make predictions on new data
    nnr.net.eval()

    if args.plot_nn:
        plot_nn(args.prefix, X_train, X_test, Y_train, Y_test)
