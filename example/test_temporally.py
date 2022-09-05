from PyRobustPCA import VanillaPCA, RobustPCAOGK, RobustPCADetMCD
from PyRobustPCA import median as median_
from PyRobustPCA import mad as mad_
from PyRobustPCA import mahalanobis_distance
from PyRobustPCA import generate_correlation_matrix as generate_correlation_matrix_
from PyRobustPCA import covariance_ogk as covariance_ogk_
from scipy import stats
from itertools import combinations
from functools import partial

import numpy as np


def calculate_bisquare_weights(x, c):
    x = np.asarray(x)
    w = (1 - (x / c) ** 2) ** 2 * (np.abs(x) <= c)
    return w


def mad(data, scale_const=1.482602218505602):
    """scale_const = 1 / scipy.stats.Gaussian.ppf(3 / 4.0)"""
    center_med = np.median(data)
    return scale_const * np.median(np.abs(data - center_med))


def winsored_squared_mean(x, c):
    return np.minimum(x**2, c**2)


def calculate_robust_mean(data, cm=4.5):
    x = np.asarray(data)
    med_x = np.median(x)
    xdm = x - med_x
    mad_x = mad(x)
    wm = calculate_bisquare_weights(xdm / mad_x, cm)
    mean = (wm * x).sum() / wm.sum()
    return mean


def scale_tau(data, cm=4.5, cs=3.0):
    """tau estimator of univariate scale"""
    x = np.asarray(data)
    n_obs = len(x)
    mad_x = mad(x)
    robust_mean = calculate_robust_mean(data, cm)
    scale_squared = mad_x**2 * winsored_squared_mean((x - robust_mean) / mad_x, cs).sum() / n_obs
    return np.sqrt(scale_squared)


def generate_correlation_matrix(data):
    """Calculating Corrilation Matrix"""
    x = np.asarray(data)
    _, n_feature = x.shape
    corr = np.diag(np.ones(n_feature, dtype=np.float32))
    for j, i in combinations(np.arange(n_feature), 2):
        vec_ipj = x[:, i] + x[:, j]
        vec_imj = x[:, i] - x[:, j]
        s1 = scale_tau(vec_ipj)
        s2 = scale_tau(vec_imj)
        cij = (s1**2 - s2**2) / 4
        corr[i, j] = corr[j, i] = cij
    return corr


def covariance_ogk(data, cm=4.5, cs=3.0):
    """Calculating Covariance Matrix with the Orthogonal Gnanadesikan-Kettenring method"""
    assert data.ndim == 2

    n_obs, n_feature = data.shape
    x = data
    # _, scale_x = scale_tau(x)
    scale_x = np.apply_along_axis(partial(scale_tau, cm=cm, cs=cs), 0, x)

    D = np.diag(scale_x)
    D_inv = np.linalg.inv(D)
    z = x.dot(D_inv.T)
    U = generate_correlation_matrix(z)
    eigvalues, E = np.linalg.eigh(U)
    indices_argsort = np.argsort(eigvalues)[::-1]
    eigvalues = eigvalues[indices_argsort]
    E = E[:, indices_argsort]
    V = z.dot(E)

    # coarse parameter estimation
    scale_v = np.apply_along_axis(partial(scale_tau, cm=cm, cs=cs), 0, V)
    sigma_z = (E.dot(np.diag(scale_v**2))).dot(E.T)
    m_vec = np.apply_along_axis(partial(calculate_robust_mean, cm=cm), 0, V).reshape(-1, 1)
    mu_z = E.dot(m_vec)
    mu_rawogk = D.dot(mu_z).flatten()
    sigma_rawogk = D.dot(sigma_z).dot(D.T)

    # parameter refinement
    dist_mahalanobis = mahalanobis(x, mu_rawogk, sigma_rawogk)
    d_squared_med = np.median(dist_mahalanobis**2)
    cutoff = d_squared_med * stats.chi2.ppf(0.90, n_feature) / stats.chi2.ppf(0.5, n_feature)
    mask = dist_mahalanobis <= cutoff
    sample = data[mask, :]
    loc = sample.mean(0)
    cov = np.cov(sample.T)

    return loc, cov


def mahalanobis(x, mu, sigma_mat):
    assert x.ndim == 2
    sigma_mat_inv = np.linalg.inv(sigma_mat)
    data_centrized_tensor = (x - mu).reshape(-1, 1, 3)
    distances_ = np.matmul(np.matmul(data_centrized_tensor, sigma_mat_inv), data_centrized_tensor.transpose(0, 2, 1))
    distances = np.sqrt(np.abs(distances_)).flatten()
    return distances


X = np.random.rand(120).reshape(-1, 3).astype(np.float64)
mu = np.mean(X, axis=0)
sigma_mat = np.cov(X.T)
print("mdist diff: ", np.mean(mahalanobis(X, mu, sigma_mat) - mahalanobis_distance(X, mu, sigma_mat)))

print("cov")
print(covariance_ogk(X))
print(covariance_ogk_(X))

print("vanilla")
pca = VanillaPCA()
pca.fit(X)
print(pca.get_mean())
print(pca.get_principal_components())


print("ogk")
rpca_ogk = RobustPCAOGK()
rpca_ogk.fit(X)
print(rpca_ogk.get_mean())
print(rpca_ogk.get_principal_components())
print(rpca_ogk.get_scores())

print("detmcd")
rpca_detmcd = RobustPCADetMCD()
rpca_detmcd.fit(X)
print(rpca_detmcd.get_mean())
print(rpca_detmcd.get_principal_components())
print(rpca_detmcd.get_scores())
