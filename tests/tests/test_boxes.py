# type: ignore
TYPE_CHECKING = False
import pytest
import torch

from anyboxes.implementations.torch.boxes import TorchBoxes

from .data import (
    BOXES_BOTTOM_LEFT_ORIGIN,
    BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE,
    BOXES_CENTER,
    BOXES_CENTER_SQUARED,
    BOXES_CENTER_SQUARED_UNIQUE,
    BOXES_CENTER_UNIQUE,
    BOXES_TOP_LEFT_ORIGIN,
    BOXES_TOP_LEFT_ORIGIN_UNIQUE,
    BOXES_TWO_CORNERS,
    BOXES_TWO_CORNERS_UNIQUE,
    EXPECTED_BOXES_TOP_LEFT_WITH_MODIFY_ORIGIN,
    IMAGE_HEIGHT,
    MASK,
    MASK_DIMENSION,
)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TOP_LEFT_ORIGIN_UNIQUE, BOXES_TOP_LEFT_ORIGIN_UNIQUE),
        (BOXES_TOP_LEFT_ORIGIN, BOXES_TOP_LEFT_ORIGIN),
    ],
)
def test_from_top_left_corner_to_top_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_top_left_corner(bboxes_input)
    assert torch.equal(b.to_top_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TOP_LEFT_ORIGIN_UNIQUE, BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE),
        (BOXES_TOP_LEFT_ORIGIN, BOXES_BOTTOM_LEFT_ORIGIN),
    ],
)
def test_from_top_left_corner_to_bottom_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_top_left_corner(bboxes_input)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TOP_LEFT_ORIGIN_UNIQUE, BOXES_CENTER_UNIQUE),
        (BOXES_TOP_LEFT_ORIGIN, BOXES_CENTER),
    ],
)
def test_from_top_left_corner_to_center(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_top_left_corner(bboxes_input)
    assert torch.equal(b.to_center().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TOP_LEFT_ORIGIN_UNIQUE, BOXES_TWO_CORNERS_UNIQUE),
        (BOXES_TOP_LEFT_ORIGIN, BOXES_TWO_CORNERS),
    ],
)
def test_from_top_left_corner_to_two_corners(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_top_left_corner(bboxes_input)
    assert torch.equal(b.to_two_corners().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE, BOXES_TOP_LEFT_ORIGIN_UNIQUE),
        (BOXES_BOTTOM_LEFT_ORIGIN, BOXES_TOP_LEFT_ORIGIN),
    ],
)
def test_from_bottom_left_corner_to_top_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_bottom_left_corner(bboxes_input)
    assert torch.equal(b.to_top_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE, BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE),
        (BOXES_BOTTOM_LEFT_ORIGIN, BOXES_BOTTOM_LEFT_ORIGIN),
    ],
)
def test_from_bottom_left_corner_to_bottom_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_bottom_left_corner(bboxes_input)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE, BOXES_CENTER_UNIQUE),
        (BOXES_BOTTOM_LEFT_ORIGIN, BOXES_CENTER),
    ],
)
def test_from_bottom_left_corner_to_center(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_bottom_left_corner(bboxes_input)
    assert torch.equal(b.to_center().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE, BOXES_TWO_CORNERS_UNIQUE),
        (BOXES_BOTTOM_LEFT_ORIGIN, BOXES_TWO_CORNERS),
    ],
)
def test_from_bottom_left_corner_to_two_corners(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_bottom_left_corner(bboxes_input)
    assert torch.equal(b.to_two_corners().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_CENTER_UNIQUE, BOXES_TOP_LEFT_ORIGIN_UNIQUE),
        (BOXES_CENTER, BOXES_TOP_LEFT_ORIGIN),
    ],
)
def test_from_center_to_top_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_center(bboxes_input)
    assert torch.equal(b.to_top_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_CENTER_UNIQUE, BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE),
        (BOXES_CENTER, BOXES_BOTTOM_LEFT_ORIGIN),
    ],
)
def test_from_center_to_bottom_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_center(bboxes_input)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_CENTER_UNIQUE, BOXES_CENTER_UNIQUE),
        (BOXES_CENTER, BOXES_CENTER),
    ],
)
def test_from_center_to_center(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_center(bboxes_input)
    assert torch.equal(b.to_center().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_CENTER_UNIQUE, BOXES_TWO_CORNERS_UNIQUE),
        (BOXES_CENTER, BOXES_TWO_CORNERS),
    ],
)
def test_from_center_to_two_corners(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_center(bboxes_input)
    assert torch.equal(b.to_two_corners().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TWO_CORNERS_UNIQUE, BOXES_TOP_LEFT_ORIGIN_UNIQUE),
        (BOXES_TWO_CORNERS, BOXES_TOP_LEFT_ORIGIN),
    ],
)
def test_from_two_corners_to_top_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_two_corners(bboxes_input)
    assert torch.equal(b.to_top_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TWO_CORNERS_UNIQUE, BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE),
        (BOXES_TWO_CORNERS, BOXES_BOTTOM_LEFT_ORIGIN),
    ],
)
def test_from_two_corners_to_bottom_left_corner(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_two_corners(bboxes_input)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TWO_CORNERS_UNIQUE, BOXES_CENTER_UNIQUE),
        (BOXES_TWO_CORNERS, BOXES_CENTER),
    ],
)
def test_from_two_corners_to_center(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_two_corners(bboxes_input)
    assert torch.equal(b.to_center().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_TWO_CORNERS_UNIQUE, BOXES_TWO_CORNERS_UNIQUE),
        (BOXES_TWO_CORNERS, BOXES_TWO_CORNERS),
    ],
)
def test_from_two_corners_to_two_corners(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_two_corners(bboxes_input)
    assert torch.equal(b.to_two_corners().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "bboxes_expected"],
    [
        (BOXES_CENTER_UNIQUE, BOXES_CENTER_SQUARED_UNIQUE),
        (BOXES_CENTER, BOXES_CENTER_SQUARED),
    ],
)
def test_from_center_squared_to_center(bboxes_input, bboxes_expected):
    b = TorchBoxes.from_center(bboxes_input)
    assert torch.equal(b.square().to_center().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "mask_dimension", "mask_expected"],
    [(BOXES_CENTER_UNIQUE, MASK_DIMENSION, MASK)],
)
def test_from_center_get_mask(bboxes_input, mask_dimension, mask_expected):
    b = TorchBoxes.from_center(bboxes_input)
    assert torch.equal(b.get_binary_mask(*mask_dimension), mask_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "image_height", "bboxes_expected"],
    [
        (
            BOXES_TOP_LEFT_ORIGIN,
            IMAGE_HEIGHT,
            EXPECTED_BOXES_TOP_LEFT_WITH_MODIFY_ORIGIN,
        ),
    ],
)
def test_top_left_flip_origin(bboxes_input, image_height, bboxes_expected):
    b = TorchBoxes.from_top_left_corner(bboxes_input)
    b.flip_origin(image_height)
    assert torch.equal(b.to_top_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "image_height", "bboxes_expected"],
    [
        (BOXES_TOP_LEFT_ORIGIN, IMAGE_HEIGHT, BOXES_TOP_LEFT_ORIGIN),
    ],
)
def test_top_left_origin_is_conserved(bboxes_input, image_height, bboxes_expected):
    b = TorchBoxes.from_top_left_corner(bboxes_input)
    b.flip_origin(image_height)
    b.flip_origin(image_height)
    assert torch.equal(b.to_top_left_corner().as_tensor, bboxes_expected)


@pytest.mark.parametrize(
    ["bboxes_input", "image_height", "bboxes_expected"],
    [
        (BOXES_BOTTOM_LEFT_ORIGIN, IMAGE_HEIGHT, BOXES_BOTTOM_LEFT_ORIGIN),
    ],
)
def test_bottom_left_origin_is_conserved(bboxes_input, image_height, bboxes_expected):
    b = TorchBoxes.from_bottom_left_corner(bboxes_input)
    b.flip_origin(image_height)
    b.flip_origin(image_height)
    assert torch.equal(b.to_bottom_left_corner().as_tensor, bboxes_expected)
