import os

import numpy as np

import bradlyterry
import bradlyterry_scaled
import bradlyterry_scaled_r
import bradlyterry_scaled_rij

import raokupper
import raokupper_scaled
import raokupper_scaled_r
import raokupper_scaled_rij

import davidson
import davidson_scaled
import davidson_scaled_r
import davidson_scaled_rij
from load_data import load_chatbot_arena_data

working_dir = os.path.dirname(os.path.abspath(__file__))


def score_jsd(y_pred, y):
    eps = 1e-12
    y = np.maximum(y, eps)
    y_pred = np.maximum(y_pred, eps)
    y = y / y.sum(axis=-1, keepdims=True)

    # compute JS divergence
    m = (y_pred + y) / 2
    js = (y * np.log(y / m) + y_pred * np.log(y_pred / m)) / 2

    if np.isnan(js).any():
        raise AssertionError("JS divergence is NaN")

    return js.sum(axis=-1).mean(axis=0)


def pretty_print(weight, models):
    # associate models with their optimized parameters
    result_dict = {}
    n_models = len(models)
    for i, name in enumerate(models):
        if len(weight) >= 2 * n_models:
            result_dict[name] = (weight[i], weight[n_models + i])
        else:
            result_dict[name] = (weight[i], 0)
    # sort result by mean
    result_dict = dict(sorted(result_dict.items(), key=lambda x: x[1][0], reverse=True))  # print the result
    print("= Model ranking =".center(57, "="))
    print("=" * 57)
    print("Model".ljust(35), "Mean".rjust(10), "Std".rjust(10))
    print("-" * 57)
    for k, v in result_dict.items():
        print(k.ljust(35), f"{v[0]:.2f}".rjust(10), f"{v[1]:.2f}".rjust(10))
    print("=" * 57)


def evaluate_algorithm(algorithm, non_tie=False):
    from sklearn.model_selection import train_test_split
    data = load_chatbot_arena_data()
    xs, ys, models = np.array(data["X"]), np.array(data["Y"]), data["models"]

    # split data into training and testing sets with 80% training and 20% testing
    xs, xs_test, ys, ys_test = train_test_split(xs, ys, test_size=0.2, random_state=42)

    if non_tie:
        ys[:, -1] = 0
        ys_test[:, -1] = 0

    n_models = len(models)
    # start_time = time.time()
    weight = algorithm.train(xs, ys, n_models)
    # print(f"{algorithm.__name__} training time: {time.time() - start_time}")
    # pretty_print(weight)
    y_pred = algorithm.inference(weight, xs_test, n_models)

    pretty_print(weight, models)

    print(f"{algorithm.__name__} JSD {'[non-tied]' if non_tie else ''}: "
          f"{score_jsd(y_pred, ys_test)}")


def evaluate_algorithm_fit_data(algorithm, non_tie=False):
    data = load_chatbot_arena_data()
    xs, ys, models = np.array(data["X"]), np.array(data["Y"]), data["models"]
    if non_tie:
        ys[:, -1] = 0
    n_models = len(models)
    # start_time = time.time()
    weight = algorithm.train(xs, ys, n_models)
    # print(f"{algorithm.__name__} training time: {time.time() - start_time}")
    # pretty_print(weight)
    y_pred = algorithm.inference(weight, xs, n_models)

    print(f"{algorithm.__name__} JSD {'[non-tied]' if non_tie else ''}: "
          f"{score_jsd(y_pred, ys)}")


algorithms = [
    bradlyterry, bradlyterry_scaled, bradlyterry_scaled_r, bradlyterry_scaled_rij, raokupper, raokupper_scaled,
    raokupper_scaled_r, raokupper_scaled_rij, davidson, davidson_scaled, davidson_scaled_r, davidson_scaled_rij
]

print("=== Fitting the same data ===")

for algo in algorithms[:4]:
    evaluate_algorithm_fit_data(algo, non_tie=True)

for algo in algorithms:
    evaluate_algorithm_fit_data(algo)

print("=== Evaluating ===")
for algo in algorithms[:4]:
    evaluate_algorithm(algo, non_tie=True)

for algo in algorithms:
    evaluate_algorithm(algo)

# plot_win_rate_bt(bt_weight)
# plot_correlation(davidson_weight)
