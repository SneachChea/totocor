import numpy as np
from totocor.utils import frame

def test_1d_frame_fn():
    target = np.array([[0, 1, 2], [2, 3, 4]])
    input = np.array([0, 1, 2, 3, 4])
    res = frame(input, window_size=3, hop_length=2)
    assert np.array_equal(res,target)

def test_2d_frame_fn():
    target = np.array([[[0, 1, 2], [2, 3, 4]]])
    input = np.array([[0, 1, 2, 3, 4]])
    res = frame(input, window_size=3, hop_length=2)
    assert np.array_equal(res,target)