from beartype import beartype as _typecheck
from beartype._data.hint.datahinttyping import BeartypeableT, BeartypeReturn
from jaxtyping import jaxtyped as _shapecheck


def typecheck(fn: BeartypeableT) -> BeartypeReturn:
    """Typecheck a function without jaxtyping annotations, otherwise additionally shapecheck the function.

    :param fn: Any function or method.
    :return: Typechecked function or method.
    :raises: BeartypeException if a call to the function does not satisfy the typecheck.
    """
    # check if there is any annotation requiring a shapecheck, i.e. any jaxtyping annotation that is not "..."
    # this check is significantly slower than the string-based check implemented below (~+50%), but this should
    # only be relevant in tight loops.
    # for annotation in fn.__annotations__.values():
    #     if getattr(annotation, "dim_str", "") != "...":

    # simply check if there is any mention of jaxtyping in the annotations, this adds barely any overhead to
    # a base call of beartype's @beartype
    if "jaxtyping" in str(fn.__annotations__):
        # shapecheck implies typecheck
        return _shapecheck(_typecheck(fn))

    return _typecheck(fn)
