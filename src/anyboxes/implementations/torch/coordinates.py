from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._typing import CoordTensorType


class Coordinates:
    """Represent a set of corner coordinates in 2D space."""

    def __init__(self, x: CoordTensorType, y: CoordTensorType, device: torch.device):
        """Make a Coordinates object.

        Args:
            x (CoordTensorType): torch.tensor containing `x` coordinates.
            y (CoordTensorType): torch.tensor containing `y` coordinates.
            device (torch.device): device
        """
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch_size: int = len(x)

    def to_dict(self) -> dict[str, CoordTensorType]:
        """Return x and y as dict.

        Returns:
            dict[str, CoordTensorType]: dict containing coordinates tensors.
        """
        return {"x": self.x, "y": self.y}


class Size:
    """Represent size of a bbox in 2D space."""

    def __init__(self, w: CoordTensorType, h: CoordTensorType, device: torch.device):
        """Make a Size object.

        Args:
            w (CoordTensorType): torch.tensor containing `w` coordinates.
            h (CoordTensorType): torch.tensor containing `h` coordinates.
            device (torch.device): device
        """
        self.w = w.to(device)
        self.h = h.to(device)
        self.batch_size: int = len(w)

    def to_dict(self) -> dict[str, CoordTensorType]:
        """Return w and h as dict.

        Returns:
            dict[str, CoordTensorType]: dict containing size tensors.
        """
        return {"w": self.w, "h": self.h}
