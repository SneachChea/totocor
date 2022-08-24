from enum import auto
from warnings import WarningMessage
import numpy as np
from .utils import frame, get_window_fn
import warnings

def autocorr(
    signal: np.ndarray,
    window_size: int,
    hop_length: int = None,
    window_fn: str = "hann",
    center: bool = True,
    pad_mode: str = "constant", normalization : bool = True,
):
    """_summary_

    Args:
        signal (np.ndarray): _description_
        window_size (int): _description_
        hop_length (int, optional): _description_. Defaults to None.
        window_fn (str, optional): _description_. Defaults to "hann".
        center (bool, optional): _description_. Defaults to True.
        pad_mode (str, optional): _description_. Defaults to "constant".
        normalization (bool, optional): _description_. Defaults to True.

    Raises:
        AttributeError: _description_

    Returns:
        _type_: _description_
    """

    if hop_length is None:
        hop_length = window_size // 4

    if center is True:
        pad = [(0, 0)] * signal.ndim
        pad[-1] = (window_size // 2, window_size // 2)
        if pad_mode not in ("constant", "reflect"):
            raise AttributeError("padding mode should be either constant or reflect")
        signal = np.pad(signal, pad_width=pad, mode=pad_mode)

    signal_framed = frame(signal, window_size=window_size, hop_length=hop_length)
    window_frame = get_window_fn(M=window_size, name=window_fn)
    signal_framed = signal_framed*window_frame
    X = np.fft.rfft(signal_framed, axis=-1)
    autocorr = np.fft.irfft(X * np.conj(X), axis=-1)
    if np.any(autocorr[..., 0]==0.):
        warnings.warn("At least one frame seems to be silent, expect NaN if normalization flag is raised", RuntimeWarning)
    if normalization:
        autocorr = autocorr/autocorr[..., 0, None]
    return autocorr