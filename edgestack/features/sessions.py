"""Session-return decomposition.

All functions deliberately leave the first unavailable observation as ``NaN``.
They never backfill it, which is important when these returns later become
features in a causal research view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ArrayLike = NDArray[np.float64] | pd.Series | pd.DataFrame


def _validate_positive(values: ArrayLike, name: str) -> None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size and np.any(finite <= 0.0):
        raise ValueError(f"{name} must be strictly positive where finite")


def _ratio_return(numerator: ArrayLike, denominator: ArrayLike, log: bool) -> ArrayLike:
    ratio = numerator / denominator
    return np.log(ratio) if log else ratio - 1.0


def close_to_close_returns(close: ArrayLike, *, log: bool = False) -> ArrayLike:
    """Compute close-to-close returns without using a future close.

    Parameters
    ----------
    close
        One- or two-dimensional close-price data. Pandas labels are preserved.
    log
        Return log returns instead of simple returns.
    """

    _validate_positive(close, "close")
    if isinstance(close, (pd.Series, pd.DataFrame)):
        previous_pandas = close.shift(1)
        return _ratio_return(close, previous_pandas, log)
    array = np.asarray(close, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError("close must be one- or two-dimensional")
    previous_array = np.full_like(array, np.nan, dtype=float)
    previous_array[1:] = array[:-1]
    return _ratio_return(array, previous_array, log)


def overnight_returns(
    open_: ArrayLike, close: ArrayLike, *, log: bool = False
) -> ArrayLike:
    """Compute prior-close to current-open returns.

    This return is available after the current session's opening print. It must
    not be joined to a pre-open decision timestamp unless that print is already
    known.
    """

    _validate_positive(open_, "open")
    _validate_positive(close, "close")
    if np.shape(open_) != np.shape(close):
        raise ValueError("open and close must have identical shapes")
    if isinstance(close, (pd.Series, pd.DataFrame)):
        if not isinstance(open_, type(close)) or not open_.index.equals(close.index):
            raise ValueError("pandas open and close must have matching type and index")
        previous_pandas = close.shift(1)
        return _ratio_return(open_, previous_pandas, log)
    else:
        close_array = np.asarray(close, dtype=float)
        previous_array = np.full_like(close_array, np.nan, dtype=float)
        previous_array[1:] = close_array[:-1]
        return _ratio_return(open_, previous_array, log)


def intraday_returns(
    open_: ArrayLike, close: ArrayLike, *, log: bool = False
) -> ArrayLike:
    """Compute same-session open-to-close returns."""

    _validate_positive(open_, "open")
    _validate_positive(close, "close")
    if np.shape(open_) != np.shape(close):
        raise ValueError("open and close must have identical shapes")
    return _ratio_return(close, open_, log)


@dataclass(frozen=True, slots=True)
class SessionReturns:
    """Three mutually consistent session-return series."""

    overnight: ArrayLike
    intraday: ArrayLike
    close_to_close: ArrayLike
    log_returns: bool = False


def decompose_sessions(
    open_: ArrayLike, close: ArrayLike, *, log: bool = True
) -> SessionReturns:
    """Decompose total returns into overnight and intraday components.

    Log returns are the default because they are exactly additive:
    ``overnight + intraday == close_to_close`` apart from missing observations.
    """

    return SessionReturns(
        overnight=overnight_returns(open_, close, log=log),
        intraday=intraday_returns(open_, close, log=log),
        close_to_close=close_to_close_returns(close, log=log),
        log_returns=log,
    )


# Singular aliases are retained because they read naturally in feature configs.
overnight_return = overnight_returns
intraday_return = intraday_returns
close_to_close_return = close_to_close_returns
