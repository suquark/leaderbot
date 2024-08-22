import numba
import numpy as np
from scipy.optimize import minimize


@numba.jit(nopython=True)
def double_sigmoid(a, b):
    return 1 / (1 + np.exp(-a) + np.exp(-b))


# ds := double_sigmoid
# d -log[double_sigmoid(a, b)] = d log[1 + exp(-a) + exp(-b)]
#                              = d [1 + exp(-a) + exp(-b)] / (1 + exp(-a) + exp(-b))
#                              = (d exp(-a) + d exp(-b)) * double_sigmoid(a, b)

# dist(xi, xj, ti, tj, rij) := (xi - xj) / sqrt[ti ** 2 + tj ** 2 - 2 tanh(rij) * |ti*tj| ]

# z := dist(.)
# u := 0.5 * z - µ
# v := -0.5 * z - µ

# p_win = double_sigmoid(z, u)
# p_loss = double_sigmoid(-z, v)
# p_tie = double_sigmoid(-u, -v)

# d -log(p_win) = - [exp(-z) dz + exp(-u) du] * p_win
#               = - [exp(-z) dz + exp(-u) (0.5 dz - dµ)] * p_win
#               = {-[exp(-z) + 0.5 exp(-u)] dz + exp(-u) dµ)} * p_win
# d -log(p_loss) = [exp(z) dz - exp(-v) dv] * p_loss
#                = [exp(z) dz - exp(-v) (-0.5 dz - dµ)] * p_loss
#                = {[exp(z) + 0.5 exp(-v)] dz + exp(-v) dµ)} * p_loss
# d -log(p_tie) = [exp(u) du + exp(v) dv] * p_tie
#               = [exp(u) (0.5 dz - dµ) + exp(v) (-0.5 dz - dµ)] * p_tie
#               = {0.5 * [exp(u) - exp(v)] dz - [exp(u) + exp(v)] dµ} * p_tie


# On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments
@numba.jit(nopython=True)
def davidson_loss(xi, xj, ti, tj, rij, mu, win_ij, loss_ij, tie_ij):
    s_rij = np.tanh(rij)
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_rij * np.abs(ti * tj))
    z = (xi - xj) * scale
    u = 0.5 * z - mu
    v = -0.5 * z - mu
    p_win = double_sigmoid(z, u)
    p_loss = double_sigmoid(-z, v)
    p_tie = double_sigmoid(-u, -v)

    loss = -win_ij * np.log(p_win) - loss_ij * np.log(p_loss) - tie_ij * np.log(p_tie)

    grad_p_win_z = -(np.exp(-z) + 0.5 * np.exp(-u)) * p_win
    grad_p_loss_z = (np.exp(z) + 0.5 * np.exp(-v)) * p_loss
    grad_p_tie_z = 0.5 * (np.exp(u) - np.exp(v)) * p_tie
    grad_z = win_ij * grad_p_win_z + loss_ij * grad_p_loss_z + tie_ij * grad_p_tie_z

    grad_p_win_u = np.exp(-u) * p_win
    grad_p_loss_u = np.exp(-v) * p_loss
    grad_p_tie_u = -(np.exp(u) + np.exp(v)) * p_tie
    grad_mu = win_ij * grad_p_win_u + loss_ij * grad_p_loss_u + tie_ij * grad_p_tie_u

    grad_xi = grad_z * scale
    grad_xj = -grad_xi

    grad_scale = -0.5 * grad_z * z * scale ** 2
    grad_ti = grad_scale * 2 * (ti - s_rij * np.sign(ti) * np.abs(tj))
    grad_tj = grad_scale * 2 * (tj - s_rij * np.sign(tj) * np.abs(ti))

    grad_rij = -grad_scale * 2 * np.abs(ti * tj) * (1 - s_rij ** 2)

    return loss, grad_xi, grad_xj, grad_ti, grad_tj, grad_rij, grad_mu


def davidson_jac(w, x, y, n_models):
    n = x.shape[0]
    loss = np.zeros(n)
    jac = np.zeros((n, w.shape[0]))

    i, j = x.T
    k = j * (j - 1) // 2 + i

    xi, xj = w[i], w[j]
    ti, tj = w[i + n_models], w[j + n_models]
    rij = w[k + n_models * 2]
    mu = w[-1]

    win_count, loss_count, tie_count = y.T
    count = win_count.sum() + loss_count.sum() + tie_count.sum()

    loss, grad_xi, grad_xj, grad_ti, grad_tj, grad_rij, grad_mu = davidson_loss(xi, xj, ti, tj, rij, mu, win_count,
                                                                                loss_count, tie_count)

    if np.isnan(np.sum(loss)):
        print("Loss is nan")

    ax = np.arange(n)

    jac[ax, i] += grad_xi
    jac[ax, j] += grad_xj
    jac[ax, i + n_models] += grad_ti
    jac[ax, j + n_models] += grad_tj
    jac[ax, k + n_models * 2] += grad_rij
    jac[ax, -1] += grad_mu

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

    # print(f"Loss: {loss}. Constraint: {constraint_diff}. Nu: {np.exp(mu)}")
    return loss, jac


def inference(w, x, n_models):
    i, j = x.T
    pair_index = j * (j - 1) // 2 + i
    xi, xj = w[i], w[j]
    ti, tj = w[n_models + i], w[n_models + j]
    rij = w[n_models * 2 + pair_index]
    mu = w[-1]

    s_rij = np.tanh(rij)
    scale = 1 / np.sqrt(ti ** 2 + tj ** 2 - 2 * s_rij * np.abs(ti * tj))
    z = (xi - xj) * scale
    u = 0.5 * z - mu
    v = -0.5 * z - mu
    p_win = double_sigmoid(z, u)
    p_loss = double_sigmoid(-z, v)
    p_tie = double_sigmoid(-u, -v)

    return np.column_stack((p_win, p_loss, p_tie))


def train(x, y, n_models):
    n_pairs = n_models * (n_models - 1) // 2
    w0 = np.zeros(n_models * 2 + n_pairs + 1)
    w0[:n_models] = np.full(n_models, np.log(1 / n_models))
    w0[n_models:n_models * 2] = np.full(n_models, 0.5 ** 0.5)

    result = minimize(davidson_jac,
                      w0,
                      args=(x, y, n_models),
                      jac=True,
                      method='L-BFGS-B',
                      options={'maxiter': 1500},
                      tol=1e-8)

    return result.x
