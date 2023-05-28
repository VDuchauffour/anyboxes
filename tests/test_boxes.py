# type: ignore
import pytest
import torch

from anyboxes.implementations.torch.boxes import TorchBoxes


@pytest.mark.usefixtures("top_left_tensor")
def test_from_top_left_corner_to_top_left_corner(top_left_tensor):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    assert torch.equal(b.to_top_left_corner().as_tensor, top_left_tensor)


@pytest.mark.usefixtures("top_left_tensor", "bottom_left_tensor")
def test_from_top_left_corner_to_bottom_left_corner(
    top_left_tensor, bottom_left_tensor
):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bottom_left_tensor)


@pytest.mark.usefixtures("top_left_tensor", "center_tensor")
def test_from_top_left_corner_to_center(top_left_tensor, center_tensor):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    assert torch.equal(b.to_center().as_tensor, center_tensor)


@pytest.mark.usefixtures("top_left_tensor", "two_corners_tensor")
def test_from_top_left_corner_to_two_corners(top_left_tensor, two_corners_tensor):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    assert torch.equal(b.to_two_corners().as_tensor, two_corners_tensor)


@pytest.mark.usefixtures("bottom_left_tensor", "top_left_tensor")
def test_from_bottom_left_corner_to_top_left_corner(
    bottom_left_tensor, top_left_tensor
):
    b = TorchBoxes.from_bottom_left_corner(bottom_left_tensor)
    assert torch.equal(b.to_top_left_corner().as_tensor, top_left_tensor)


@pytest.mark.usefixtures("bottom_left_tensor")
def test_from_bottom_left_corner_to_bottom_left_corner(bottom_left_tensor):
    b = TorchBoxes.from_bottom_left_corner(bottom_left_tensor)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bottom_left_tensor)


@pytest.mark.usefixtures("bottom_left_tensor", "center_tensor")
def test_from_bottom_left_corner_to_center(bottom_left_tensor, center_tensor):
    b = TorchBoxes.from_bottom_left_corner(bottom_left_tensor)
    assert torch.equal(b.to_center().as_tensor, center_tensor)


@pytest.mark.usefixtures("bottom_left_tensor", "two_corners_tensor")
def test_from_bottom_left_corner_to_two_corners(bottom_left_tensor, two_corners_tensor):
    b = TorchBoxes.from_bottom_left_corner(bottom_left_tensor)
    assert torch.equal(b.to_two_corners().as_tensor, two_corners_tensor)


@pytest.mark.usefixtures("center_tensor", "top_left_tensor")
def test_from_center_to_top_left_corner(center_tensor, top_left_tensor):
    b = TorchBoxes.from_center(center_tensor)
    assert torch.equal(b.to_top_left_corner().as_tensor, top_left_tensor)


@pytest.mark.usefixtures("center_tensor", "bottom_left_tensor")
def test_from_center_to_bottom_left_corner(center_tensor, bottom_left_tensor):
    b = TorchBoxes.from_center(center_tensor)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bottom_left_tensor)


@pytest.mark.usefixtures("center_tensor")
def test_from_center_to_center(center_tensor):
    b = TorchBoxes.from_center(center_tensor)
    assert torch.equal(b.to_center().as_tensor, center_tensor)


@pytest.mark.usefixtures("center_tensor", "two_corners_tensor")
def test_from_center_to_two_corners(center_tensor, two_corners_tensor):
    b = TorchBoxes.from_center(center_tensor)
    assert torch.equal(b.to_two_corners().as_tensor, two_corners_tensor)


@pytest.mark.usefixtures("two_corners_tensor", "top_left_tensor")
def test_from_two_corners_to_top_left_corner(two_corners_tensor, top_left_tensor):
    b = TorchBoxes.from_two_corners(two_corners_tensor)
    assert torch.equal(b.to_top_left_corner().as_tensor, top_left_tensor)


@pytest.mark.usefixtures("two_corners_tensor", "bottom_left_tensor")
def test_from_two_corners_to_bottom_left_corner(two_corners_tensor, bottom_left_tensor):
    b = TorchBoxes.from_two_corners(two_corners_tensor)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bottom_left_tensor)


@pytest.mark.usefixtures("two_corners_tensor", "center_tensor")
def test_from_two_corners_to_center(two_corners_tensor, center_tensor):
    b = TorchBoxes.from_two_corners(two_corners_tensor)
    assert torch.equal(b.to_center().as_tensor, center_tensor)


@pytest.mark.usefixtures("two_corners_tensor")
def test_from_two_corners_to_two_corners(two_corners_tensor):
    b = TorchBoxes.from_two_corners(two_corners_tensor)
    assert torch.equal(b.to_two_corners().as_tensor, two_corners_tensor)


@pytest.mark.usefixtures("center_tensor", "squared_center_tensor")
def test_from_center_squared_to_center(center_tensor, squared_center_tensor):
    b = TorchBoxes.from_center(center_tensor)
    assert torch.equal(b.square().to_center().as_tensor, squared_center_tensor)


@pytest.mark.usefixtures("center_tensor_to_mask", "mask_dimension", "mask_tensor")
def test_from_center_get_mask(center_tensor_to_mask, mask_dimension, mask_tensor):
    b = TorchBoxes.from_center(center_tensor_to_mask)
    assert torch.equal(b.get_binary_mask(*mask_dimension), mask_tensor)


@pytest.mark.usefixtures(
    "top_left_tensor", "image_height", "top_left_tensor_with_bottom_left_origin"
)
def test_top_left_flip_origin(
    top_left_tensor, image_height, top_left_tensor_with_bottom_left_origin
):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    b.flip_origin(image_height)
    assert torch.equal(
        b.to_top_left_corner().as_tensor, top_left_tensor_with_bottom_left_origin
    )


@pytest.mark.usefixtures("top_left_tensor", "image_height")
def test_top_left_origin_is_conserved(top_left_tensor, image_height):
    b = TorchBoxes.from_top_left_corner(top_left_tensor)
    b.flip_origin(image_height)
    b.flip_origin(image_height)
    assert torch.equal(b.to_top_left_corner().as_tensor, top_left_tensor)


@pytest.mark.usefixtures("bottom_left_tensor", "image_height")
def test_bottom_left_origin_is_conserved(bottom_left_tensor, image_height):
    b = TorchBoxes.from_bottom_left_corner(bottom_left_tensor)
    b.flip_origin(image_height)
    b.flip_origin(image_height)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bottom_left_tensor)
