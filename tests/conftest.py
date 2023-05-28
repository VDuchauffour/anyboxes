import pytest
import torch


@pytest.fixture
def top_left_data():
    return [[0.0, 0.0, 3.0, 2.0], [10.0, 10.0, 10.0, 10.0]]


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
