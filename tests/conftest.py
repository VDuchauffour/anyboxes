import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch


@pytest.fixture
def top_left_data():
    return [[0.0, 0.0, 3.0, 2.0], [10.0, 10.0, 10.0, 10.0]]


@pytest.fixture
def top_left_dict():
    return {
        "1": {"x": [0.0, 10.0], "y": [0.0, 10.0]},
        "2": {"x": [3.0, 20.0], "y": [0.0, 10.0]},
        "3": {"x": [3.0, 20.0], "y": [2.0, 20.0]},
        "4": {"x": [0.0, 10.0], "y": [2.0, 20.0]},
        "c": {"x": [1.5, 15.0], "y": [1.0, 15.0]},
        "size": {"w": [3.0, 10.0], "h": [2.0, 10.0]},
    }


@pytest.fixture
def top_left_tuple():
    return (
        [0.0, 10.0],
        [0.0, 10.0],
        [3.0, 20.0],
        [0.0, 10.0],
        [3.0, 20.0],
        [2.0, 20.0],
        [0.0, 10.0],
        [2.0, 20.0],
        [1.5, 15.0],
        [1.0, 15.0],
        [3.0, 10.0],
        [2.0, 10.0],
    )


@pytest.fixture
def top_left_numpy(top_left_data):
    return np.asarray(top_left_data)


@pytest.fixture
def top_left_array(top_left_data):
    return jnp.asarray(top_left_data)


@pytest.fixture
def top_left_tf_tensor(top_left_data):
    return tf.convert_to_tensor(top_left_data)


@pytest.fixture
def top_left_tensor(top_left_data):
    return torch.tensor(top_left_data)


@pytest.fixture
def bottom_left_data():
    return [[0.0, 2.0, 3.0, 2.0], [10.0, 20.0, 10.0, 10.0]]


@pytest.fixture
def bottom_left_tensor(bottom_left_data):
    return torch.tensor(bottom_left_data)


@pytest.fixture
def center_data():
    return [[1.5, 1.0, 3.0, 2.0], [15.0, 15.0, 10.0, 10.0]]


@pytest.fixture
def center_tensor(center_data):
    return torch.tensor(center_data)


@pytest.fixture
def two_corners_data():
    return [[0.0, 0.0, 3.0, 2.0], [10.0, 10.0, 20.0, 20.0]]


@pytest.fixture
def two_corners_tensor(two_corners_data):
    return torch.tensor(two_corners_data)


@pytest.fixture
def squared_center_data():
    return [[1.5, 1.0, 3.0, 3.0], [15.0, 15.0, 10.0, 10.0]]


@pytest.fixture
def squared_center_tensor(squared_center_data):
    return torch.tensor(squared_center_data)


@pytest.fixture
def image_height():
    return 30


@pytest.fixture
def top_left_data_with_bottom_left_origin():
    return [[0.0, 30.0, 3.0, 2.0], [10.0, 20.0, 10.0, 10.0]]


@pytest.fixture
def top_left_tensor_with_bottom_left_origin(top_left_data_with_bottom_left_origin):
    return torch.tensor(top_left_data_with_bottom_left_origin)


@pytest.fixture
def center_data_to_mask():
    return [[1.5, 1.0, 3.0, 2.0]]


@pytest.fixture
def center_tensor_to_mask(center_data_to_mask):
    return torch.tensor(center_data_to_mask)


@pytest.fixture
def mask_dimension():
    return (10, 10)


@pytest.fixture
def mask_data():
    return [
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ]


@pytest.fixture
def mask_tensor(mask_data):
    return torch.tensor(mask_data)
