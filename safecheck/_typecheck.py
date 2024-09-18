from beartype import beartype as _typecheck
from beartype.typing import Any, Callable, TypeVar, Union
from jaxtyping import jaxtyped as _shapecheck

__all__ = [
    "typecheck",
    "CallableAnyT",
    "CheckableT",
]


class MissingAnnotationError(Exception): ...


CallableAnyT = Callable[..., Any]
CheckableT = TypeVar(
    "CheckableT",
    # Equal to ``BeartypeableT``, but without allowing class decorators
    bound=Union[
        # An arbitrary callable *OR*...
        CallableAnyT,
        # A C-based unbound class method descriptor (i.e., a pure-Python unbound
        # function decorated by the builtin @classmethod decorator) *OR*...
        classmethod,
        # A C-based unbound property method descriptor (i.e., a pure-Python
        # unbound function decorated by the builtin @property decorator) *OR*...
        property,
        # A C-based unbound static method descriptor (i.e., a pure-Python
        # unbound function decorated by the builtin @staticmethod decorator).
        staticmethod,
    ],
)


def raise_if_missing_annotation(fn: CheckableT) -> None:  # type: ignore[reportInvalidTypeVarUse]
    """Any function or method should have an ``__annotations__`` attribute, we therefore assume it's a function."""
    if not hasattr(fn, "__annotations__"):
        msg = (
            f"@typecheck-decorated objects must have an '__annotations__' attribute, but '{fn}' has "
            f" no '__annotations__' attribute. Did you try to use @typecheck on an unsupported object? "
            f"In contrast to beartype, we do not allow implicit typechecking of entire classes and @typecheck "
            f"must not be used as a class decorator."
        )
        raise MissingAnnotationError(msg)


def typecheck(fn: CheckableT, *, skip_annotation_check: bool = False) -> CheckableT:
    """Typecheck a function without jaxtyping annotations, otherwise additionally shapecheck the function.

    :param fn: Any function or method.
    :param skip_annotation_check: Disable the missing annotation check if you already know it exists.
    :return: Typechecked function or method.
    :raises: MissingAnnotationError if ``fn`` does not contain ``__annotations__`` (is no function or method).
    :raises: BeartypeException if a call to the function does not satisfy the typecheck.
    """
    # check if there is any annotation requiring a shapecheck, i.e. any jaxtyping annotation that is not "..."
    # this check is significantly slower than the string-based check implemented below (~+50%), but this should
    # only be relevant in tight loops.
    # for annotation in fn.__annotations__.values():
    #     if getattr(annotation, "dim_str", "") != "...":

    if not skip_annotation_check:
        raise_if_missing_annotation(fn)

    # simply check if there is any mention of jaxtyping in the annotations, this adds barely any overhead to
    # a base call of beartype's @beartype
    if "jaxtyping" in str(fn.__annotations__):
        # shapecheck implies typecheck
        return _shapecheck(typechecker=_typecheck)(fn)  # type: ignore[reportGeneralTypeIssues]

    return _typecheck(fn)
