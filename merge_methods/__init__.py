from .base import MergeMethod
from .linear import LinearMerge
from .passthrough import PassthroughMerge
from .slerp import SlerpMerge
from .ties import TiesMerge


def get(method: str) -> MergeMethod:
    if method == "ties":
        return TiesMerge()
    elif method == "linear":
        return LinearMerge()
    elif method == "slerp":
        return SlerpMerge()
    elif method == "passthrough":
        return PassthroughMerge()
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeMethod",
    "get",
    "LinearMerge",
    "TiesMerge",
    "SlerpMerge",
    "PassthroughMerge",
]
