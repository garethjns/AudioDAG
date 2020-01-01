import numpy as np


def ms_to_pts(t_ms: int, fs: int) -> int:
    """Convert time in ms to time in pts."""
    return int(fs * t_ms / 1000)


def pts_to_ms(t_pts: int, fs: int) -> int:
    """Convert time in pts to time in ms (to nearest whole)"""
    return int(np.round(t_pts * 1000 / fs))


def lin_to_db(ref: float, ratio: float) -> float:
    """
    Convert linear ratio to dB change.

    :param ref: Reference value
    :param ratio: Linear ratio change.
    :return: dB change.
    """
    return 100 * np.log10((ref * ratio) / ref)


def db_to_lin(ref: float, db_change: float) -> float:
    """
    Convert dB change to linear ratio.

    :param ref: Reference value
    :param db_change: dB change.
    :return: Linear ratio change.
    """
    return 10 ** (db_change / 100) / ref ** 2
