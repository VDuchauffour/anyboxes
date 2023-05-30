import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

from anyboxes.implementations.torch.boxes import TorchBoxes


@pytest.mark.usefixtures("top_left_tensor", "top_left_dict")
def test_as_dict(top_left_tensor, top_left_dict):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    assert b.as_dict == top_left_dict


@pytest.mark.usefixtures("top_left_tensor", "top_left_tuple")
def test_as_tuple(top_left_tensor, top_left_tuple):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    assert b.as_tuple == top_left_tuple


@pytest.mark.usefixtures("top_left_tensor", "top_left_numpy")
def test_as_numpy(top_left_tensor, top_left_numpy):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    b = b.to_top_left_corner()
    np.testing.assert_equal(b.as_numpy, top_left_numpy)


@pytest.mark.usefixtures("top_left_tensor", "top_left_array")
def test_as_array(top_left_tensor, top_left_array):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    b = b.to_top_left_corner()
    assert jnp.array_equal(b.as_array, top_left_array)


@pytest.mark.usefixtures("top_left_tensor", "top_left_tf_tensor")
def test_as_tf_tensor(top_left_tensor, top_left_tf_tensor):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    b = b.to_top_left_corner()
    tf.assert_equal(b.as_tf_tensor, top_left_tf_tensor)


@pytest.mark.usefixtures("top_left_tensor")
def test_as_tensor(top_left_tensor):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    b = b.to_top_left_corner()
    assert torch.equal(b.as_tensor, top_left_tensor)
