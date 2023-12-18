try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .aslnn import AslNNModel
from .aslrest import AslRestModel
from .aslrest_disp import AslRestDisp

__all__ = ["AslNNModel", "AslRestModel", "AslRestDisp", "__version__"]
