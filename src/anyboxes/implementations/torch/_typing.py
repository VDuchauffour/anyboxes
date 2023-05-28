from typing import Tuple

from jaxtyping import Shaped
from torch import Tensor

from .coordinates import Coordinates

CoordTensorType = Shaped[Tensor, "batch"]

BoxTensorType = Shaped[Tensor, "4"]
BoxesTensorType = Shaped[Tensor, "batch 4"]
MaskTensorType = Shaped[Tensor, "batch height width"]

ImageTensorType = Shaped[Tensor, "channel height width"]
BatchImageTensorType = Shaped[Tensor, "batch channel height width"]

FourCornersCoordinates = Tuple[
    Coordinates,
    Coordinates,
    Coordinates,
    Coordinates,
]
