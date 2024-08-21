import numba
import numpy as np
from scipy.optimize import minimize


@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.jit(nopython=True)
def bt_loss(xi, xj, ti, tj, rij, win_ij, loss_ij, tie_ij):
    s_rij = np.tanh(rij)
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_rij * np.abs(ti * tj))
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
    grad_ti = grad_scale * 2 * (ti - s_rij * np.sign(ti) * np.abs(tj))
    grad_tj = grad_scale * 2 * (tj - s_rij * np.sign(tj) * np.abs(ti))

    grad_rij = -grad_scale * 2 * np.abs(ti * tj) * (1 - s_rij ** 2)

    return loss, grad_xi, grad_xj, grad_ti, grad_tj, grad_rij


def bt_jac(w, x, y, n_models):
    n = x.shape[0]
    loss = np.zeros(n)
    jac = np.zeros((n, w.shape[0]))

    i, j = x.T
    k = j * (j - 1) // 2 + i

    xi, xj = w[i], w[j]
    ti, tj = w[i + n_models], w[j + n_models]
    rij = w[k + n_models * 2]

    win_count, loss_count, tie_count = y.T
    count = win_count.sum() + loss_count.sum() + tie_count.sum()

    loss, grad_xi, grad_xj, grad_ti, grad_tj, grad_rij = bt_loss(xi, xj, ti, tj, rij, win_count, loss_count, tie_count)

    if np.isnan(np.sum(loss)):
        print("Loss is nan")

    ax = np.arange(n)

    jac[ax, i] += grad_xi
    jac[ax, j] += grad_xj
    jac[ax, i + n_models] += grad_ti
    jac[ax, j + n_models] += grad_tj
    jac[ax, k + n_models * 2] += grad_rij

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
    pair_index = j * (j - 1) // 2 + i
    xi, xj = w[i], w[j]
    ti, tj = w[n_models + i], w[n_models + j]
    rij = w[n_models * 2 + pair_index]

    s_rij = np.tanh(rij)
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_rij * np.abs(ti * tj))
    z = (xi - xj) * scale

    p_win = sigmoid(z)
    p_loss = 1 - p_win
    p_tie = 1 - p_win - p_loss
    return np.column_stack((p_win, p_loss, p_tie))


def train(x, y, n_models):
    n_pairs = n_models * (n_models - 1) // 2
    P = np.zeros(n_models * 2 + n_pairs)
    P[:n_models] = np.full(n_models, np.log(1 / n_models))
    P[n_models:n_models * 2] = np.full(n_models, 0.5 ** 0.5)

    result = minimize(bt_jac,
                      P,
                      args=(x, y, n_models),
                      jac=True,
                      method='L-BFGS-B',
                      options={'maxiter': 1500},
                      tol=1e-7)

    return result.x
