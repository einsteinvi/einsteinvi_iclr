""" Bayesian neural networks for UCI regression benchmarks. Code based on
    https://docs.pymc.io/en/stable/pymc-examples/examples/variational_inference/bayesian_neural_network_advi.html """
import argparse
from collections import namedtuple
from pathlib import Path
from time import time

import numpy as np
import pymc3 as pm
from sklearn.model_selection import train_test_split
import theano
import theano.tensor as T

DATADIR = Path(__file__).parent
DataState = namedtuple("data", ["xtr", "xte", "ytr", "yte"])

floatX = theano.config.floatX


def load_data(name: str) -> DataState:
    data = np.loadtxt(DATADIR / f"{name}.txt")
    x, y = data[:, :-1], data[:, -1]

    return DataState(*train_test_split(x, y, train_size=0.90))


def normalize(val, mean=None, std=None):
    if mean is None and std is None:
        std = np.std(val, 0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        mean = np.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "boston_housing",
            "concrete",
            "energy_heating_load",
            "kin8nm",
            "naval_compressor_decay",
            "power",
            "protein",
            "wine",
            "yacht",
            "year_prediction_msd",
        ],
        default="boston_housing",
    )
    parser.add_argument("--subsample_size", type=int, default=100)  # Fixed
    args = parser.parse_args()

    data = load_data(args.dataset)
    xtr, xtr_mean, xtr_std = normalize(data.xtr)
    ytr, ytr_mean, ytr_std = normalize(data.ytr)

    def construct_nn(
        ann_input, ann_output, n_hidden, num_obs, num_feat
    ):  # n_hidden is fixed.
        init_1 = np.random.randn(num_feat, n_hidden).astype(floatX)
        init_out = np.random.randn(n_hidden).astype(floatX)

        with pm.Model() as neural_network:
            ann_input = pm.Data("ann_input", xtr)
            ann_output = pm.Data("ann_output", ytr)
            prec_obs = pm.Gamma("prec_obs", 1.0, 0.1)
            prec_nn = pm.Gamma("prec_nn", 1.0, 0.1)

            # Weights from input to hidden layer
            w1 = pm.Normal(
                "w_in_1",
                0,
                sigma=1 / prec_nn,
                shape=(num_feat, n_hidden),
                testval=init_1,
            )
            b1 = pm.Normal("b1", 0, 1 / prec_nn, shape=n_hidden)

            # Weights from hidden layer to output
            w2 = pm.Normal(
                "w_2_out", 0, sigma=1 / prec_nn, shape=n_hidden, testval=init_out
            )
            b2 = pm.Normal("b2", 0, 1 / prec_nn, shape=())

            # Build neural-network using tanh activation function
            act_1 = pm.math.maximum(pm.math.dot(ann_input, w1), 0) + b1
            act_out = pm.math.dot(act_1, w2) + b2

            out = pm.Normal(
                "out", act_out, sigma=prec_obs, observed=ann_output, total_size=num_obs
            )
        return neural_network

    mini_x = pm.Minibatch(xtr, batch_size=args.subsample_size)
    mini_y = pm.Minibatch(ytr, batch_size=args.subsample_size)
    neural_network = construct_nn(mini_x, mini_y, 50, *xtr.shape)

    times = []
    scores = []
    with neural_network:
        inference = pm.SVGD()
        start = time()
        approx = pm.fit(
            n=2_000, method=inference, obj_optimizer=pm.adagrad(learning_rate=1.0)
        )  # LR from grid-search
    times.append(time() - start)

    x = T.matrix("X")
    n = T.iscalar("n")
    xte = normalize(data.xte, xtr_mean, xtr_std)[0]
    x.tag.test_value = np.empty_like(xte)
    n.tag.test_value = args.subsample_size
    _sample_proba = approx.sample_node(
        neural_network.out.distribution.mu,
        size=n,
        more_replacements={neural_network["ann_input"]: x},
    )

    sample_proba = theano.function([x, n], _sample_proba)

    prds = []

    for i in range(xte.shape[0] // args.subsample_size):
        prds.append(
            sample_proba(
                xte[i * args.subsample_size : (i + 1) * args.subsample_size], 1
            ).mean(0)
        )

    if xte.shape[0] % args.subsample_size:
        remainder = xte.shape[0] % args.subsample_size
        prds.append(sample_proba(xte[-remainder:], 1).mean(0))

    pred = np.concatenate(prds)
    y_pred = pred * ytr_std + ytr_mean
    scores.append(np.sqrt(((y_pred - data.yte) ** 2).mean()))
    times = np.array(times)
    scores = np.array(scores)
    print(f"rmse: {scores.mean():.2f}")
    print(f"times: {times.mean():.2f}")
