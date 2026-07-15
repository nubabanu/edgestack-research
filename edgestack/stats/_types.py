"""Shared NumPy array aliases used by the typed research kernel."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
DateArray = NDArray[np.datetime64]
