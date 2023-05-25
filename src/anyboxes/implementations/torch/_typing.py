from typing import Tuple

from jaxtyping import Float
from torch import Tensor

from .coordinates import Coordinates

CoordTensorType = Float[Tensor, "batch"]

BoxTensorType = Float[Tensor, "4"]
BoxesTensorType = Float[Tensor, "batch 4"]
MaskTensorType = Float[Tensor, "batch height width"]

ImageTensorType = Float[Tensor, "channel height width"]
BatchImageTensorType = Float[Tensor, "batch channel height width"]

FourCornersCoordinates = Tuple[
    Coordinates,
    Coordinates,
    Coordinates,
    Coordinates,
]
