from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Sample: TypeAlias = npt.NDArray[np.float64]

Interval = tuple[float, float]
