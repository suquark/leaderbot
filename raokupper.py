import numba
import numpy as np
from scipy.optimize import minimize


@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments
@numba.jit(nopython=True)
def raokupper_loss(xi, xj, eta, win_ij, loss_ij, tie_ij):
    d_win = xi - xj - eta
    d_loss = xj - xi - eta
    s_win = sigmoid(d_win)
    s_loss = sigmoid(d_loss)
    loss = -(win_ij + tie_ij) * np.log(s_win) - (loss_ij +
                                                 tie_ij) * np.log(s_loss) - tie_ij * np.log(np.exp(2 * eta) - 1)

    # gradient of the loss function
    grad_win = (win_ij + tie_ij) * (1 - s_win)
    grad_loss = (loss_ij + tie_ij) * (1 - s_loss)
    grad_xi = -grad_win + grad_loss
    grad_xj = -grad_xi
    grad_eta = grad_win + grad_loss - tie_ij * 2 / (1 - np.exp(-2 * eta))
    return loss, grad_xi, grad_xj, grad_eta


def raokupper_jac(w, x, y, n_models):
    n = x.shape[0]
    loss = np.zeros(n)
    jac = np.zeros((n, w.shape[0]))

    i, j = x.T
    xi, xj = w[i], w[j]
    w[-1] = np.maximum(w[-1], 1e-3)  # clip eta
    eta = w[-1]

    win_count, loss_count, tie_count = y.T
    count = win_count.sum() + loss_count.sum() + tie_count.sum()

    loss, grad_xi, grad_xj, grad_eta = raokupper_loss(xi, xj, eta, win_count, loss_count, tie_count)

    if np.isnan(np.sum(loss)):
        print("Loss is nan")

    ax = np.arange(n)

    jac[ax, i] += grad_xi
    jac[ax, j] += grad_xj
    jac[ax, n_models] += grad_eta

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

    # print(f"Loss: {loss}. Constraint: {constraint_diff}. Theta: {np.exp(eta)}")
    return loss, jac


def inference(w, x, n_models):
    del n_models
    i, j = x.T
    xi, xj = w[i], w[j]
    eta = w[-1]
    p_win = sigmoid(xi - xj - eta)
    p_loss = sigmoid(xj - xi - eta)
    p_tie = 1 - p_win - p_loss
    return np.column_stack((p_win, p_loss, p_tie))


def train(x, y, n_models):
    w = np.zeros(n_models + 1)
    w[:n_models] = np.full(n_models, np.log(1 / n_models))

    result = minimize(raokupper_jac,
                      w,
                      args=(x, y, n_models),
                      jac=True,
                      method='L-BFGS-B',
                      options={'maxiter': 1500},
                      tol=1e-7)

    return result.x
