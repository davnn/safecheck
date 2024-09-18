"""Re-exports a subset of the functionality of the following packages.

- beartype
- jaxtyping

We additionally rename some of the functions to be independent of the underlying packages. For example,
it should be easily possible to switch from beartype to typeguard for runtime type checking.
"""

from warnings import filterwarnings

# re-export everything necessary from beartype, never use beartype itself.
# additionally ignore the default beartype deprecation warnings.
from beartype.door import (
    die_if_unbearable as assert_instance,
    is_bearable as is_instance,
)
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from beartype.vale import (
    Is,
    IsAttr,
    IsEqual,
    IsInstance,
    IsSubclass,
)

# re-export everything necessary from jaxtyping, never use jaxtyping itself
from jaxtyping import (
    AbstractDtype,  # type: ignore[reportGeneralTypeIssues]
    Bool,  # type: ignore[reportGeneralTypeIssues]
    Complex,  # type: ignore[reportGeneralTypeIssues]
    Complex64,  # type: ignore[reportGeneralTypeIssues]
    Complex128,  # type: ignore[reportGeneralTypeIssues]
    Float,  # type: ignore[reportGeneralTypeIssues]
    Float16,  # type: ignore[reportGeneralTypeIssues]
    Float32,  # type: ignore[reportGeneralTypeIssues]
    Float64,  # type: ignore[reportGeneralTypeIssues]
    Inexact,  # type: ignore[reportGeneralTypeIssues]
    Int,  # type: ignore[reportGeneralTypeIssues]
    Int8,  # type: ignore[reportGeneralTypeIssues]
    Int16,  # type: ignore[reportGeneralTypeIssues]
    Int32,  # type: ignore[reportGeneralTypeIssues]
    Int64,  # type: ignore[reportGeneralTypeIssues]
    Integer,  # type: ignore[reportGeneralTypeIssues]
    Num,  # type: ignore[reportGeneralTypeIssues]
    Real,  # type: ignore[reportGeneralTypeIssues]
    Shaped,  # type: ignore[reportGeneralTypeIssues]
    UInt,  # type: ignore[reportGeneralTypeIssues]
    UInt8,  # type: ignore[reportGeneralTypeIssues]
    UInt16,  # type: ignore[reportGeneralTypeIssues]
    UInt32,  # type: ignore[reportGeneralTypeIssues]
    UInt64,  # type: ignore[reportGeneralTypeIssues]
)

from ._overload import typecheck_overload
from ._typecheck import typecheck

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

__all__ = [
    # decorators (runtime type-checking)
    "typecheck",
    "typecheck_overload",
    # introspection
    "is_instance",  # like "isinstance(...)"
    "assert_instance",  # like "assert isinstance(...)"
    # validators (runtime only, see https://beartype.readthedocs.io/en/latest/api_vale/)
    "Is",  # Annotated[str, Is[lambda x: x > 0)]]
    "IsAttr",  # Annotated[NumpyArray, IsAttr["ndim", IsEqual[1]]]
    "IsEqual",  # Annotated[list, IsEqual[list(range(42))]]
    "IsSubclass",  # Annotated[type, IsSubclass[str, bytes]]
    "IsInstance",  # Annotated[object, IsInstance[str, bytes]]
    # abstract type
    "AbstractDtype",
    # union array types
    "Shaped",  # Any type at all (e.g. object or string)
    "Num",  # Any integer, unsigned integer, floating, or complex
    "Real",  # Any integer, unsigned integer or floating
    "Inexact",  # Any floating or complex
    "Float",  # Any floating point
    "Complex",  # Any complex
    "Integer",  # Any integer or unsigned integer
    "UInt",  # Any unsigned integer
    "Int",  # Any signed integer
    # concrete array types
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float16",
    "Float32",
    "Float64",
    "Bool",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Complex64",
    "Complex128",
]

try:
    from numpy import ndarray as NumpyArray  # noqa: N812

    __all__ += ["NumpyArray"]
except ImportError:  # pragma: no cover
    ...

try:
    from torch import Tensor as TorchArray

    __all__ += ["TorchArray"]
except ImportError:  # pragma: no cover
    ...

try:
    from jax import Array as JaxArray

    __all__ += ["JaxArray"]
except ImportError:  # pragma: no cover
    ...


def get_version() -> str:
    """Return the package version or "unknown" if no version can be found."""
    from importlib import metadata

    try:
        return metadata.version(__name__)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
