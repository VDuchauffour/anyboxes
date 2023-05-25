"""Package."""
from importlib.metadata import version

__version__ = version(__name__.split(".", 1)[0])

from anyboxes.implementations.torch.boxes import TorchBoxes
