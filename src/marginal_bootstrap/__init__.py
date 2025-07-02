from scipy.stats import gaussian_kde
import numpy as np

def bootstrap_non_parametric(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    n = len(data)
    indexes = random_generator.choice(n, size=(n, bootstrap_size), replace=True)
    return data[indexes]


def bootstrap_bayesian(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    n = len(data)
    
    weights = random_generator.dirichlet(np.ones(n), size=bootstrap_size)
    return (data.T * weights).T

def bootstrap_with_jitter(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    
    n = len(data)
    indexes = random_generator.choice(n, size=(n, bootstrap_size), replace=True)
    jitter = random_generator.normal(0, np.std(data) / 10, size=(n, bootstrap_size))
    return data[indexes] + jitter

def bootstrap_kde(data, bootstrap_size=1000, random_generator=None):
    random_generator = random_generator or np.random.default_rng()
    kde = gaussian_kde(data)
    n = len(data)
    return np.array([
        kde.resample(n, seed=random_generator)
        for _ in range(bootstrap_size)
    ]).reshape(n, bootstrap_size)
