from scipy.stats import gaussian_kde
import numpy as np

def silverman_bandwidth(x):
    n = len(x)
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34)
    return 0.9 * sigma * n ** (-1 / 5)

def bootstrap_non_parametric(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    n = len(data)
    indexes = random_generator.choice(n, size=(n, bootstrap_size), replace=True)
    return data[indexes]


def bootstrap_bayesian(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    n = len(data)
    
    weights = random_generator.dirichlet(np.ones(n), size=bootstrap_size)
    return (data.T * weights).T * n

def bootstrap_with_jitter(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    
    silverman_h = silverman_bandwidth(data).reshape(-1, 1)

    n = len(data)
    indexes = random_generator.choice(n, size=(n, bootstrap_size), replace=True)
    jitter = random_generator.normal(0, silverman_h, size=(n, bootstrap_size))
    return data[indexes] + jitter

def bootstrap_kde(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    bandwidth = silverman_bandwidth(data) / np.std(data, ddof=1)
    kde = gaussian_kde(data, bw_method=bandwidth)
    n = len(data)
    return np.array([
        kde.resample(n, seed=random_generator)
        for _ in range(bootstrap_size)
    ]).reshape(n, bootstrap_size)
