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
    """
    function that compute the autocorrelation of a signal

    Args:
        signal (np.ndarray): signal array
        window_size (int): Window size of the frame for the autocorrelation
        hop_length (int, optional): Hop length between frames. If None, hop_length = window_size //4.
        window_fn (str, optional): Windowing function. Can be "rectangular", "hann" or "hamming". Defaults to "hann".
        center (bool, optional): If true, the signal is padded so the frame at time T is centered at signal[.., T*hop_length]. Defaults to True.
        pad_mode (str, optional): How the padding is done if the frame is centered. Can be "constant" or "reflect". Defaults to "constant".
        normalization (bool, optional): If True, the autocorrelation is normalized by the variance of the frame. A warning is raised if the frame is completely silent. Defaults to True.


    Returns:
        autocorr (np.ndarray): The autocorrelation signal
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