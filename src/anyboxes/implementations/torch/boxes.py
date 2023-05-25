from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from anyboxes._errors import MissingToMethodError
from anyboxes.implementations.origin import Origin

from .coordinates import Coordinates, Size

if TYPE_CHECKING:
    from jaxtyping import Float
    from numpy.typing import NDArray

    from ._typing import (
        BoxesTensorType,
        BoxTensorType,
        CoordTensorType,
        FourCornersCoordinates,
        MaskTensorType,
    )


class TorchBoxes:
    """Represent a collection of bounding boxes. Coordinates are represented as follow:
    1: top-left corner
    2: top-right corner
    3: bottom-right corner
    3: bottom-left corner.

    The boxes will all be stored as it has been given.
    The `to` methods are inplace and produce a `boxes_` attribute.
    """

    def __init__(
        self,
        corners_coordinates: FourCornersCoordinates,
        center_coordinates: Coordinates,
        size: Size,
        origin: Origin,
        device: torch.device,
    ):
        """Make a Boxes object. Object instanced is expected by classmethod
        `from_center`, `from_top_left_corner` or `from_two_corners`.

        Args:
            corners_coordinates (FourCornersCoordinates): All corners coordinates.
            center_coordinates (Coordinates): Center coordinates.
            size (Size): Size of Boxes.
            origin (Origin): Origin of Boxes.
            device (torch.device): PyTorch device.
        """
        self.corners_coordinates = corners_coordinates
        self.center_coordinates = center_coordinates
        self.size = size
        self._origin = origin
        self.device = device

    @property
    def dimensions(self) -> Float[torch.Tensor, "5"]:  # noqa
        """Return dimensions of FourCornersCoordinates and center_coordinates.

        Returns:
            TensorType[5]: Dimensions of FourCornersCoordinates and center_coordinates.
        """
        c_1, c_2, c_3, c_4 = self.corners_coordinates
        return torch.tensor(
            [
                c_1.batch_size,
                c_2.batch_size,
                c_3.batch_size,
                c_4.batch_size,
                self.center_coordinates.batch_size,
            ]
        )

    def __len__(self) -> int:
        """Return the number of Boxes.

        Raises:
            RuntimeError: Some bboxes dimension aren't the same.

        Returns:
            int: Number of Boxes.
        """
        dimensions = torch.unique(self.dimensions)
        if len(dimensions) != 1:
            raise RuntimeError("Not same boxes dimension!")
        return dimensions.item()

    @property
    def origin(self) -> str:
        """Return the value of the origin of the Boxes.

        Returns:
            str: name of the origin
        """
        return self._origin.value

    def flip_origin(self, height: int):
        """Flip the origin of the Boxes given an image height. Work in place.

        Args:
            height (int): height of the image.

        Returns:
            self
        """
        if height < self.size.h.min():
            raise ValueError("`width` or `height` must be higher than boxes.")

        for i, _ in enumerate(self.corners_coordinates):
            self.corners_coordinates[i].y = height - self.corners_coordinates[i].y
        self._origin = (
            Origin.BOTTOM_LEFT if self._origin == Origin.TOP_LEFT else Origin.TOP_LEFT
        )
        return self

    @staticmethod
    def __recast_tensor(
        tensor: Float[torch.Tensor, "batch 1"]  # noqa: F722
    ) -> CoordTensorType:
        """Recast torch.Tensor to 1 dimension in order to handle batch size
        cases.

        Args:
            tensor (TensorType["batch", 1]): Tensor split from Boxes tensor

        Returns:
            CoordTensorType: Tensor of 1 dimension.
        """
        tensor = tensor.squeeze()
        return tensor.unsqueeze(0) if tensor.dim() == 0 else tensor

    @staticmethod
    def __extract_coordinates_from_tensors(
        boxes: BoxesTensorType,
    ) -> tuple[CoordTensorType, CoordTensorType, CoordTensorType, CoordTensorType,]:
        """Extract coordinates from `torch.Tensor`.

        Args:
            boxes (BoxesTensorType): Bounding boxes of shape (b, 4).

        Returns:
            Tuple[CoordTensorType, CoordTensorType, CoordTensorType, CoordTensorType,]: Tuple of coordinates.
        """
        x_1, x_2, x_3, x_4 = map(
            TorchBoxes.__recast_tensor, torch.tensor_split(boxes, 4, dim=1)
        )
        return x_1, x_2, x_3, x_4

    @staticmethod
    def _extract_coordinates_from_tensor(
        boxes: BoxTensorType,
    ) -> tuple[CoordTensorType, CoordTensorType, CoordTensorType, CoordTensorType,]:
        """Extract coordinates from `torch.Tensor`.

        Args:
            boxes (BoxesTensorType): Bounding box of shape (4).

        Returns:
            Tuple[CoordTensorType, CoordTensorType, CoordTensorType, CoordTensorType,]: Tuple of coordinates.
        """
        x_1, x_2, x_3, x_4 = map(
            TorchBoxes.__recast_tensor, torch.tensor_split(boxes.unsqueeze(0), 4, dim=1)
        )
        return x_1, x_2, x_3, x_4

    @staticmethod
    def __compute_corners_from_center(
        center_coordinates: Coordinates, size: Size, device: torch.device
    ) -> FourCornersCoordinates:
        """Generate corners coordinates given center coordinates and size.

        Args:
            center_coordinates (Coordinates): Center coordinates.
            size (Size): Size of Boxes.
            device (torch.device): PyTorch device.

        Returns:
            FourCornersCoordinates: All corners coordinates.
        """
        x_c, y_c = (center_coordinates.x, center_coordinates.y)
        w, h = (
            size.w,
            size.h,
        )
        x_1, x_2 = x_c - w / 2, x_c + w / 2
        x_3, x_4 = x_c + w / 2, x_c - w / 2
        y_1, y_2 = y_c - h / 2, y_c - h / 2
        y_3, y_4 = y_c + h / 2, y_c + h / 2
        return (
            Coordinates(x_1, y_1, device),
            Coordinates(x_2, y_2, device),
            Coordinates(x_3, y_3, device),
            Coordinates(x_4, y_4, device),
        )

    @staticmethod
    def __compute_corners_from_top_left_corner_and_size(
        top_left_corner: Coordinates, size: Size, device: torch.device
    ) -> FourCornersCoordinates:
        """Generate corners coordinates from top-left corner and size.

        Args:
            top_left_corner (Coordinates): Top-left corner.
            size (Size): Size of Boxes.
            device (torch.device): PyTorch device.

        Returns:
            FourCornersCoordinates: All corners coordinates.
        """
        x_1, y_1 = (
            top_left_corner.x,
            top_left_corner.y,
        )
        w, h = size.w, size.h
        x_2, y_2 = x_1 + w, y_1
        x_3, y_3 = x_1 + w, y_1 + h
        x_4, y_4 = x_1, y_1 + h
        return (
            Coordinates(x_1, y_1, device),
            Coordinates(x_2, y_2, device),
            Coordinates(x_3, y_3, device),
            Coordinates(x_4, y_4, device),
        )

    @staticmethod
    def __compute_corners_from_bottom_left_corner(
        bottom_left_corner: Coordinates, size: Size, device: torch.device
    ) -> FourCornersCoordinates:
        """Generate corners coordinates from bottom-left corner.

        Args:
            bottom_left_corner (Coordinates): Bottom-left corner.
            size (Size): Size of Boxes.
            device (torch.device): PyTorch device.

        Returns:
            FourCornersCoordinates: All corners coordinates.
        """
        x_4, y_4 = bottom_left_corner.x, bottom_left_corner.y
        w, h = size.w, size.h
        x_1, y_1 = x_4, y_4 - h
        x_2, y_2 = x_4 + w, y_4 - h
        x_3, y_3 = x_4 + w, y_4
        return (
            Coordinates(x_1, y_1, device),
            Coordinates(x_2, y_2, device),
            Coordinates(x_3, y_3, device),
            Coordinates(x_4, y_4, device),
        )

    @staticmethod
    def __compute_corners_from_two_corners(
        top_left_corner: Coordinates,
        bottom_right_corner: Coordinates,
        device: torch.device,
    ) -> FourCornersCoordinates:
        """Generate corners coordinates from two corners.

        Args:
            top_left_corner (Coordinates): Top-left corner.
            bottom_right_corner (Coordinates): Bottom-right corner.
            device (torch.device): PyTorch device.

        Returns:
            FourCornersCoordinates: All corners coordinates.
        """
        return (
            top_left_corner,
            Coordinates(bottom_right_corner.x, top_left_corner.y, device),
            bottom_right_corner,
            Coordinates(top_left_corner.x, bottom_right_corner.y, device),
        )

    @staticmethod
    def __compute_size_from_two_corners(
        top_left_corner: Coordinates,
        bottom_right_corner: Coordinates,
        device: torch.device,
    ) -> Size:
        """Generate size from two corners.

        Args:
            top_left_corner (Coordinates): Top-left corner.
            bottom_right_corner (Coordinates): Bottom-right corner.
            device (torch.device): PyTorch device.

        Returns:
            size (Size): Size of BBoxes.
        """
        return Size(
            bottom_right_corner.x - top_left_corner.x,
            bottom_right_corner.y - top_left_corner.y,
            device,
        )

    @staticmethod
    def __compute_center_from_corners(
        corners_coordinates: FourCornersCoordinates, device: torch.device
    ) -> Coordinates:
        """Generate center coordinates given corners coordinates.

        Args:
            corners_coordinates (FourCornersCoordinates): All corners coordinates.
            device (torch.device): PyTorch device.

        Returns:
            Coordinates: Center coordinates.
        """
        c_1, c_2, c_3, c_4 = corners_coordinates
        x_c = (c_1.x + c_2.x) / 2
        y_c = (c_1.y + c_3.y) / 2
        return Coordinates(x_c, y_c, device)

    @classmethod
    def from_center(
        cls, boxes: BoxesTensorType, origin: Origin = Origin.TOP_LEFT
    ) -> TorchBoxes:
        """Generate Boxes from torch.Tensor containing center coordinates and
        size.

        Args:
            boxes (torch.Tensor) : boxes of size (n, 4),
                containing the (x_c, y_c, w, h) for all boxes
            origin (Origin): default is `top-left`

        Returns:
            Boxes : object of class Boxes.
        """
        x_c, y_c, w, h = TorchBoxes.__extract_coordinates_from_tensors(boxes)
        device = boxes.device
        center_coordinates = Coordinates(x_c, y_c, device)
        size = Size(w, h, device)
        corners_coordinates = TorchBoxes.__compute_corners_from_center(
            center_coordinates, size, device
        )
        return cls(
            corners_coordinates,
            center_coordinates,
            size,
            origin,
            device,
        )

    @classmethod
    def from_top_left_corner(cls, boxes: BoxesTensorType) -> TorchBoxes:
        """Generate Boxes from torch.Tensor containing top-left coordinates and
        size.

        Args:
            boxes (torch.Tensor) : boxes of size (n, 4),
                containing the (x_1, y_1, w, h) for all boxes

        Returns:
            Boxes : object of class Boxes
        """
        x_1, y_1, w, h = TorchBoxes.__extract_coordinates_from_tensors(boxes)
        device = boxes.device
        size = Size(w, h, device)
        corners_coordinates = (
            TorchBoxes.__compute_corners_from_top_left_corner_and_size(
                Coordinates(x_1, y_1, device), size, device
            )
        )
        center_coordinates = TorchBoxes.__compute_center_from_corners(
            corners_coordinates, device
        )
        return cls(
            corners_coordinates,
            center_coordinates,
            size,
            Origin.TOP_LEFT,
            device,
        )

    @classmethod
    def from_bottom_left_corner(cls, boxes: BoxesTensorType) -> TorchBoxes:
        """Generate Boxes from torch.Tensor containing bottom-left coordinates
        and size.

        Args:
            boxes (torch.Tensor) : boxes of size (n, 4),
                containing the (x_4, y_4, w, h) for all boxes

        Returns:
            Boxes : object of class Boxes
        """
        x_4, y_4, w, h = TorchBoxes.__extract_coordinates_from_tensors(boxes)
        device = boxes.device
        size = Size(w, h, device)
        corners_coordinates = TorchBoxes.__compute_corners_from_bottom_left_corner(
            Coordinates(x_4, y_4, device), size, device
        )
        center_coordinates = TorchBoxes.__compute_center_from_corners(
            corners_coordinates, device
        )
        return cls(
            corners_coordinates,
            center_coordinates,
            size,
            Origin.BOTTOM_LEFT,
            device,
        )

    @classmethod
    def from_two_corners(
        cls, boxes: BoxesTensorType, origin: Origin = Origin.TOP_LEFT
    ) -> TorchBoxes:
        """Generate Boxes from torch.Tensor containing top-left and bottom-
        right coordinates.

        Args:
            boxes (torch.Tensor) : boxes of size (n, 4),
                containing the (x_1, y_1, x_3, y_3) for all boxes
            origin (Origin): default is `top-left`

        Returns:
            Boxes : object of class Boxes
        """
        x_1, y_1, x_3, y_3 = TorchBoxes.__extract_coordinates_from_tensors(boxes)
        device = boxes.device
        c_1, c_3 = Coordinates(x_1, y_1, device), Coordinates(x_3, y_3, device)
        size = TorchBoxes.__compute_size_from_two_corners(c_1, c_3, device)
        corners_coordinates = TorchBoxes.__compute_corners_from_two_corners(
            c_1, c_3, device
        )
        center_coordinates = TorchBoxes.__compute_center_from_corners(
            corners_coordinates, device
        )
        return cls(
            corners_coordinates,
            center_coordinates,
            size,
            origin,
            device,
        )

    def to_center(self):
        """Generate boxes (inplace method) : boxes of size (n, 4),
        containing the (x_c, y_c, w, h) for all boxes.

        Returns:
            self
        """
        self.boxes_ = torch.stack(
            [
                self.center_coordinates.x,
                self.center_coordinates.y,
                self.size.w,
                self.size.h,
            ],
            dim=1,
        ).to(self.device)
        return self

    def to_top_left_corner(self):
        """Generate boxes (inplace method) : boxes of size (n, 4),
        containing the (x_1, y_1, w, h) for all boxes.

        Returns:
            self
        """
        self.boxes_ = torch.stack(
            [
                self.corners_coordinates[0].x,
                self.corners_coordinates[0].y,
                self.size.w,
                self.size.h,
            ],
            dim=1,
        ).to(self.device)
        return self

    def to_bottom_left_corner(self):
        """Generate boxes (inplace method): boxes of size (n, 4), containing
        the (x_4, y_4, w, h) for all boxes.

        Returns:
            self
        """
        self.boxes_ = torch.stack(
            [
                self.corners_coordinates[3].x,
                self.corners_coordinates[3].y,
                self.size.w,
                self.size.h,
            ],
            dim=1,
        ).to(self.device)
        return self

    def to_two_corners(self):
        """Generate boxes (inplace method) : boxes of size (n, 4), containing the (x_1, y_1, x_3, y_3) for all boxes.

        Returns:
            self
        """
        self.boxes_ = torch.stack(
            [
                self.corners_coordinates[0].x,
                self.corners_coordinates[0].y,
                self.corners_coordinates[2].x,
                self.corners_coordinates[2].y,
            ],
            dim=1,
        ).to(self.device)
        return self

    def square(self):
        """Pad the box so that they become squares. Works in place.

        Returns:
            self
        """
        self.size, self.corners_coordinates = self.squared()
        return self

    def squared(self) -> tuple[Size, FourCornersCoordinates]:
        """Pad the box so that they become squares.

        Returns:
            size : Size of the Boxes.
            Boxes : A different Boxes object containing the squared boxes.
        """
        max_w_h, _ = torch.stack((self.size.w, self.size.h)).max(dim=0)
        size = Size(max_w_h, max_w_h, self.device)
        return size, TorchBoxes.__compute_corners_from_center(
            self.center_coordinates, size, self.device
        )

    def get_binary_mask(self, width: int, height: int) -> MaskTensorType:
        """Build a mask to hide the parts of the image inside the bounding
        boxes.

        Args:
            width (int) : desired mask width
            height (int) : desired mask height
        Returns:
            torch.Tensor : binary tensor or binary array, containing the mask
                (has ones everywhere, with zeroes inside the bounding boxes), size (n_boxes, height, width)
        """
        if width < self.size.w.min() or height < self.size.h.min():
            raise ValueError("`width` or `height` must be higher than boxes.")

        masks = torch.ones(len(self), height, width, device=self.device)

        for i, mask in enumerate(masks):
            y_1 = int(self.corners_coordinates[0].y[i].to(torch.int).item())
            y_3 = int(self.corners_coordinates[2].y[i].to(torch.int).item())
            x_1 = int(self.corners_coordinates[0].x[i].to(torch.int).item())
            x_3 = int(self.corners_coordinates[2].x[i].to(torch.int).item())
            mask[
                y_1:y_3,
                x_1:x_3,
            ] = 0
        return masks.to(torch.int)

    @property
    def as_dict(self) -> dict[str, dict[str, CoordTensorType]]:
        """Return boxes' data as dictionnary.

        Returns:
            dict[str, dict[str, torch.Tensor]]: dict containing coordinates and size of Boxes.
        """
        return {
            "1": self.corners_coordinates[0].to_dict(),
            "2": self.corners_coordinates[1].to_dict(),
            "3": self.corners_coordinates[2].to_dict(),
            "4": self.corners_coordinates[3].to_dict(),
            "c": self.center_coordinates.to_dict(),
            "size": self.size.to_dict(),
        }

    @property
    def as_tuple(self):
        """Return boxes' data as tuple.

        Returns:
            tuple[CoordTensorType]: tuple containing coordinates and size of Boxes.
        """
        return (
            self.corners_coordinates[0].x,
            self.corners_coordinates[0].y,
            self.corners_coordinates[1].x,
            self.corners_coordinates[1].y,
            self.corners_coordinates[2].x,
            self.corners_coordinates[2].y,
            self.corners_coordinates[3].x,
            self.corners_coordinates[3].y,
            self.center_coordinates.x,
            self.center_coordinates.y,
            self.size.w,
            self.size.h,
        )

    @property
    def as_numpy(self) -> NDArray:
        """Return boxes' data as `np.ndarray`.

        Returns:
            np.ndarray: Return boxes as Numpy array.
        """
        if hasattr(self, "boxes_"):
            return self.boxes_.to(torch.float).detach().cpu().numpy()
        else:
            raise MissingToMethodError

    @property
    def as_tensor(self) -> BoxesTensorType:
        """Return boxes' data as `torch.Tensor`.

        Returns:
            BboxesTensor: Return bboxes as Torch tensor.
        """
        if hasattr(self, "boxes_"):
            return self.boxes_.to(torch.float)
        else:
            raise MissingToMethodError
