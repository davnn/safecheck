from beartype import beartype as _typecheck
from beartype._data.hint.datahinttyping import BeartypeableT, BeartypeReturn
from jaxtyping import jaxtyped as _shapecheck


class MissingAnnotationError(Exception):
    ...


def typecheck(fn: BeartypeableT) -> BeartypeReturn:
    """Typecheck a function without jaxtyping annotations, otherwise additionally shapecheck the function.

    :param fn: Any function or method.
    :return: Typechecked function or method.
    :raises: MissingAnnotationError if ``fn`` does not contain ``__annotations__`` (is no function or method).
    :raises: BeartypeException if a call to the function does not satisfy the typecheck.
    """
    # check if there is any annotation requiring a shapecheck, i.e. any jaxtyping annotation that is not "..."
    # this check is significantly slower than the string-based check implemented below (~+50%), but this should
    # only be relevant in tight loops.
    # for annotation in fn.__annotations__.values():
    #     if getattr(annotation, "dim_str", "") != "...":

    # simply check if there is any mention of jaxtyping in the annotations, this adds barely any overhead to
    # a base call of beartype's @beartype
    if not hasattr(fn, "__annotations__"):
        msg = (
            f"@typecheck-decorated objects must have an '__annotations__' attribute, but '{fn.__qualname__}' has "
            f" no '__annotations__' attribute. Did you try to use @typecheck on an unsupported object? "
            f"In contrast to beartype, we do not allow implicit typechecking of entire classes and @typecheck "
            f"must not be used as a class decorator."
        )
        raise MissingAnnotationError(msg)

    if "jaxtyping" in str(fn.__annotations__):
        # shapecheck implies typecheck
        return _shapecheck(typechecker=_typecheck)(fn)  # type: ignore[reportGeneralTypeIssues]

    return _typecheck(fn)
