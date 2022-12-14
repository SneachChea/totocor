import numpy as np




def frame(x: np.ndarray, window_size: int, hop_length: int):
    """Return the frame version of signal x.

    Args:
        x (np.ndarray): signal, dimension should be [..., T] where T is the temporal dimension
        window_size (int): Window size that would be used for computing autocorrelation
        hop_length (int): hop length
    """

    if hop_length < 1 or window_size < 1:
        raise ValueError("hop length and window size should be a positive integer")

    if x.shape[-1] < window_size:
        raise ValueError(
            f"Temporal size of the signal should be greater than the window size. Got a temporal size of {x.shape[-1]} and a window size of {window_size}"
        )
    slided_signal = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_size, axis=-1)
    target_axis = -2
    slices = [slice(None)]*slided_signal.ndim
    slices[target_axis] = slice(0, None, hop_length)

    return slided_signal[tuple(slices)]


def get_window_fn(M: int, name: str = None):
    """function that facilitate the window function. Can be "rectangular", "hann" or "hamming".

    Args:
        M (int): Size of the frame
        name (str, optional): Name of the function. Can be "rectangular" (or None), "hann" or "hamming". Defaults to None.

    Raises:
        ValueError: If the argument name is not "rectangular" (or None), "hann" or "hamming", raise ValueError

    Returns:
        window function (np.ndarray): An array of length M.
    """
    if name is None or name == "rectangular":
        return np.ones(M)
    elif name == "hann":
        return np.hanning(M)
    elif name == "hamming":
        return np.hamming(M)
    else:
        raise ValueError(f"Window function should be either rectangular, hann our hamming. Got {name}")