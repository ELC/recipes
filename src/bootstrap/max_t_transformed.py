import numpy as np
import xarray as xr
from scipy.special import logit, expit

xr.set_options(keep_attrs=True)

from .models import Interval, Sample

def transform(x):
    lower, upper = x.attrs.get("bounds", (None, None))
    if lower is None and upper is None:
        return x                     # Unbounded → no transform
    elif lower == 0 and upper is None:
        return np.log(x)            # Positive-only → log
    elif lower == 0 and upper == 1:
        return logit(x)             # Proportion → logit
    
    # Generic affine rescaling to [0,1], then logit
    z = (x - lower) / (upper - lower)
    return logit(z)

def inverse_transform(y):
    lower, upper = y.attrs.get("bounds", (None, None))
    if lower is None and upper is None:
        return y                    # Unbounded → no transform
    elif lower == 0 and upper is None:
        return np.exp(y)            # Positive-only → log
    elif lower == 0 and upper == 1:
        return expit(y)             # Proportion → logit

    # Inverse logit, then rescale
    z = expit(y)
    return lower + z * (upper - lower)

def max_t_transformed_ci(
    metrics: xr.Dataset,
    alpha: float = 0.05
) -> Interval:
    """
    Adaptive-width CIs using percentile of max-t statistic.
    """
    transformed_metrics = metrics.map(transform)
    means = transformed_metrics.mean()
    stds = transformed_metrics.std(ddof=1)

    studentized = (transformed_metrics - means) / stds
    max_t = studentized.to_array().max(dim="variable")

    t_star = np.quantile(max_t, 1 - alpha)

    lower = means - t_star * stds
    upper = means + t_star * stds

    transformed_lower = lower.map(inverse_transform)
    transformed_upper = upper.map(inverse_transform)

    return xr.concat([transformed_lower, transformed_upper], dim="ci")
