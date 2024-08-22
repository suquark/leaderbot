import numba
import numpy as np
from scipy.optimize import minimize


@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments
@numba.jit(nopython=True)
def raokupper_loss(xi, xj, ti, tj, r, eta, win_ij, loss_ij, tie_ij):
    s_r = np.tanh(r)
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_r * np.abs(ti * tj))
    z = (xi - xj) * scale

    d_win = z - eta
    d_loss = -z - eta
    p_win = sigmoid(d_win)
    p_loss = sigmoid(d_loss)
    # p_tie = (np.exp(eta * 2) - 1) * p_win * p_loss
    loss = -(win_ij + tie_ij) * np.log(p_win) - (loss_ij +
                                                 tie_ij) * np.log(p_loss) - tie_ij * np.log(np.exp(2 * eta) - 1)

    # gradient of the loss function
    grad_dwin = (win_ij + tie_ij) * (1 - p_win)
    grad_dloss = (loss_ij + tie_ij) * (1 - p_loss)
    grad_z = -grad_dwin + grad_dloss

    grad_xi = grad_z * scale
    grad_xj = -grad_xi

    grad_scale = -0.5 * grad_z * z * scale ** 2
    grad_ti = grad_scale * 2 * ti
    grad_tj = grad_scale * 2 * tj
    grad_scale = -0.5 * grad_z * z * scale ** 2
    grad_ti = grad_scale * 2 * (ti - s_r * np.sign(ti) * np.abs(tj))
    grad_tj = grad_scale * 2 * (tj - s_r * np.sign(tj) * np.abs(ti))

    grad_r = -grad_scale * 2 * np.abs(ti * tj) * (1 - s_r ** 2)

    grad_eta = grad_dwin + grad_dloss - tie_ij * 2 / (1 - np.exp(-2 * eta))
    return loss, grad_xi, grad_xj, grad_ti, grad_tj, grad_r, grad_eta


def raokupper_jac(w, x, y, n_models):
    n = x.shape[0]
    loss = np.zeros(n)
    jac = np.zeros((n, w.shape[0]))

    i, j = x.T
    xi, xj = w[i], w[j]
    ti, tj = w[i + n_models], w[j + n_models]
    r = w[-2]
    w[-1] = np.maximum(w[-1], 1e-3)  # clip eta
    eta = w[-1]

    win_count, loss_count, tie_count = y.T
    count = win_count.sum() + loss_count.sum() + tie_count.sum()

    loss, grad_xi, grad_xj, grad_ti, grad_tj, grad_r, grad_eta = raokupper_loss(xi, xj, ti, tj, r, eta, win_count,
                                                                                loss_count, tie_count)

    if np.isnan(np.sum(loss)):
        print("Loss is nan")

    ax = np.arange(n)

    jac[ax, i] += grad_xi
    jac[ax, j] += grad_xj
    jac[ax, i + n_models] += grad_ti
    jac[ax, j + n_models] += grad_tj
    jac[ax, -2] += grad_r
    jac[ax, -1] += grad_eta

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

    # print(f"Loss: {loss}. Constraint: {constraint_diff}. Theta: {np.exp(eta)}. R: {r}")
    return loss, jac


def inference(w, x, n_models):
    i, j = x.T
    xi, xj = w[i], w[j]
    ti, tj = w[i + n_models], w[j + n_models]
    r = w[-2]
    eta = w[-1]
    s_r = np.tanh(r)
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_r * np.abs(ti * tj))
    z = (xi - xj) * scale
    d_win = z - eta
    d_loss = -z - eta
    p_win = sigmoid(d_win)
    p_loss = sigmoid(d_loss)
    p_tie = 1 - p_win - p_loss
    return np.column_stack([p_win, p_loss, p_tie])


def train(x, y, n_models):
    w0 = np.zeros(n_models * 2 + 2)
    w0[:n_models] = np.full(n_models, np.log(1 / n_models))
    w0[n_models:n_models * 2] = np.full(n_models, 0.5 ** 0.5)

    result = minimize(raokupper_jac,
                      w0,
                      args=(x, y, n_models),
                      jac=True,
                      method='L-BFGS-B',
                      options={'maxiter': 1500},
                      tol=1e-8)

    return result.x
