try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .aslnn import AslNNModel
from .aslrest import AslRestModel

__all__ = [
    "AslNNModel",
    "AslRestModel",
    "__version__"
]
