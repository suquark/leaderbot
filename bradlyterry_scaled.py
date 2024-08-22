import numba
import numpy as np
from scipy.optimize import minimize


@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.jit(nopython=True)
def bt_loss(xi, xj, ti, tj, win_ij, loss_ij, tie_ij):
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2)
    z = (xi - xj) * scale

    p_win = sigmoid(z)
    p_loss = 1 - p_win

    w_ij = win_ij + 0.5 * tie_ij
    l_ij = loss_ij + 0.5 * tie_ij

    loss = -w_ij * np.log(p_win) - l_ij * np.log(p_loss)

    # grad_z = w_ij * (p_win - 1) + l_ij * p_win
    grad_z = (w_ij + l_ij) * p_win - w_ij

    grad_xi = grad_z * scale
    grad_xj = -grad_xi

    grad_scale = -0.5 * grad_z * z * scale ** 2
    grad_ti = grad_scale * 2 * ti
    grad_tj = grad_scale * 2 * tj

    return loss, grad_xi, grad_xj, grad_ti, grad_tj


def bt_jac(w, x, y, n_models):
    n = x.shape[0]
    loss = np.zeros(n)
    jac = np.zeros((n, w.shape[0]))

    i, j = x.T
    xi, xj = w[i], w[j]
    ti, tj = w[i + n_models], w[j + n_models]

    win_count, loss_count, tie_count = y.T
    count = win_count.sum() + loss_count.sum() + tie_count.sum()

    loss, grad_xi, grad_xj, grad_ti, grad_tj = bt_loss(xi, xj, ti, tj, win_count, loss_count, tie_count)

    if np.isnan(np.sum(loss)):
        print("Loss is nan")

    ax = np.arange(n)

    jac[ax, i] += grad_xi
    jac[ax, j] += grad_xj
    jac[ax, i + n_models] += grad_ti
    jac[ax, j + n_models] += grad_tj

    loss = loss.sum()
    jac = jac.sum(axis=0)

    loss /= count
    jac /= count

    # constraint
    constraint_diff = np.sum(np.exp(w[:n_models])) - 1
    constraint_loss = constraint_diff ** 2
    constraint_jac = 2 * constraint_diff * np.exp(w[:n_models])
    loss += constraint_loss
    jac[:n_models] += constraint_jac

    # print(f"Loss: {loss}. Constraint: {constraint_diff}")
    return loss, jac


def inference(w, x, n_models):
    i, j = x.T
    xi, xj = w[i], w[j]
    ti, tj = w[i + n_models], w[j + n_models]
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2)
    z = (xi - xj) * scale
    p_win = sigmoid(z)
    p_loss = 1 - p_win
    p_tie = 1 - p_win - p_loss
    return np.column_stack((p_win, p_loss, p_tie))


def train(x, y, n_models):
    w0 = np.full(n_models * 2, np.log(1 / n_models))
    w0[n_models:n_models * 2] = 0.5 ** 0.5
    result = minimize(bt_jac,
                      w0,
                      args=(x, y, n_models),
                      jac=True,
                      method='L-BFGS-B',
                      options={'maxiter': 1500},
                      tol=1e-7)
    return result.x
