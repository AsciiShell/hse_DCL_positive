import torch
import numpy as np
import pytest
from loss import get_target_mask


@pytest.mark.parametrize("test_input,expected", [
    ([1, 2, 3], [[1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1],
                 [1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1]]),
    ([2, 2, 2], [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]),
    ([-1, -1, -1], [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]),
    ([1, 2, 1], [[1, 0, 1, 1, 0, 1],
                 [0, 1, 0, 0, 1, 0],
                 [1, 0, 1, 1, 0, 1],
                 [1, 0, 1, 1, 0, 1],
                 [0, 1, 0, 0, 1, 0],
                 [1, 0, 1, 1, 0, 1]]),
])
def test_get_target_mask(test_input, expected):
    out = get_target_mask(torch.Tensor(test_input)).numpy()
    assert np.all(out == np.array(expected))
