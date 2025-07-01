import numpy as np
import xarray as xr


from .models import Interval, Sample

def max_t_ci(
    metrics: xr.Dataset,
    alpha: float = 0.05
) -> Interval:
    """
    Adaptive-width CIs using percentile of max-t statistic.
    """
    means = metrics.mean()
    stds = metrics.std(ddof=1)

    studentized = (metrics - means) / stds
    max_t = studentized.to_array().max(dim="variable")

    t_star = np.quantile(max_t, 1 - alpha)

    lower = means - t_star * stds
    upper = means + t_star * stds

    return xr.concat([lower, upper], dim="ci")
