import numpy as np
import xarray as xr

from .models import Interval, Sample

def max_statistic_ci(
    metrics: xr.Dataset,
    alpha: float = 0.05
) -> Interval:
    """
    Same-width CIs using percentile of max absolute deviations from mean.
    """
    means = metrics.mean()

    centered = metrics - means
    max_dev = centered.to_array().max(dim="variable")

    q = np.quantile(max_dev, 1 - alpha)

    lower = means - q
    upper = means + q

    return xr.concat([lower, upper], dim="ci")
