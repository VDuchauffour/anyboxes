import pytest

from anyboxes._errors import MissingToMethodError
from anyboxes.implementations.torch.boxes import TorchBoxes


@pytest.mark.usefixtures("top_left_tensor")
def test_missing_to_method_as_numpy(top_left_tensor):
    with pytest.raises(
        MissingToMethodError,
        match=(
            "One of the method `to_top_left_corner`, `to_bottom_left_corner`,"
            " `to_two_corners` or `to_center` must be called."
        ),
    ):
        b = TorchBoxes.from_top_left_corner(top_left_tensor)
        b.as_numpy


@pytest.mark.usefixtures("top_left_tensor")
def test_missing_to_method_as_array(top_left_tensor):
    with pytest.raises(
        MissingToMethodError,
        match=(
            "One of the method `to_top_left_corner`, `to_bottom_left_corner`,"
            " `to_two_corners` or `to_center` must be called."
        ),
    ):
        b = TorchBoxes.from_top_left_corner(top_left_tensor)
        b.as_array


@pytest.mark.usefixtures("top_left_tensor")
def test_missing_to_method_as_tf_tensor(top_left_tensor):
    with pytest.raises(
        MissingToMethodError,
        match=(
            "One of the method `to_top_left_corner`, `to_bottom_left_corner`,"
            " `to_two_corners` or `to_center` must be called."
        ),
    ):
        b = TorchBoxes.from_top_left_corner(top_left_tensor)
        b.as_tf_tensor


@pytest.mark.usefixtures("top_left_tensor")
def test_missing_to_method_as_tensor(top_left_tensor):
    with pytest.raises(
        MissingToMethodError,
        match=(
            "One of the method `to_top_left_corner`, `to_bottom_left_corner`,"
            " `to_two_corners` or `to_center` must be called."
        ),
    ):
        b = TorchBoxes.from_top_left_corner(top_left_tensor)
        b.as_tensor
