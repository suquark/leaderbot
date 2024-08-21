import numba
import numpy as np


@numba.jit(nopython=True)
def iterative_solver(n_models: int, x, y, max_iter: int = 100, tol: float = 1e-7):
    """Iterative solver for the Bradley-Terry model. Inspired by 
        'Efficient Computation of Rankings from Pairwise Comparisons' by M. E. J. Newman.

    Args:
        n_models: Number of models.
        x: Array of shape (n_pairs, 3) where each row represents a pair of models.
        y: Array of shape (n_pairs, 3) where each row represents the outcome of a pair of models.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
    """
    w = np.zeros(n_models)
    i, j = x.T
    win_count, loss_count, tie_count = y.T
    w_ij = win_count + 0.5 * tie_count
    w_ji = loss_count + 0.5 * tie_count

    for _ in range(max_iter):
        last_w = w.copy()
        p = np.exp(w)
        a = np.zeros_like(p)
        b = np.zeros_like(p)
        pi, pj = p[i], p[j]

        s = pi + pj
        p_win_ij = pi / s
        a_i = w_ij * (1 - p_win_ij)
        b_i = w_ji / s
        a_j = w_ji * p_win_ij
        b_j = w_ij / s

        for k in range(x.shape[0]):
            a[i[k]] += a_i[k]
            b[i[k]] += b_i[k]
            a[j[k]] += a_j[k]
            b[j[k]] += b_j[k]

        # Use MAP prior to avoid instability with sparse data
        a += 1 / (1 + p)
        b += 1 / (1 + p)

        w = np.log(a / b)
        w -= w.mean()
        if np.allclose(w, last_w, atol=tol):
            return w

    return w


def inference(w, x, n_models: int):
    """Inference for the Bradley-Terry model.

    Args:
        w: Array of shape (n_models,) representing the weights.
        x: Array of shape (n_pairs, 3) where each row represents a pair of models.
        n_models: Number of models.
    """
    del n_models
    i, j = x.T
    z = w[i] - w[j]
    p_win = 1 / (1 + np.exp(-z))  # sigmoid
    p_loss = 1 - p_win
    p_tie = 1 - p_win - p_loss
    return np.column_stack((p_win, p_loss, p_tie))


def train(x, y, n_models: int):
    """Train the Bradley-Terry model.

    Args:
        x: Array of shape (n_pairs, 3) where each row represents a pair of models.
        y: Array of shape (n_pairs, 3) where each row represents the outcome of a pair of models.
        n_models: Number of models
    """
    return iterative_solver(n_models, x, y)
