"""Enable typechecking of @overload-decorated functions.

Previously, this library supported [plum](https://github.com/beartype/plum) as a multiple dispatch library, but it
turns out that it's really difficult to implement multiple dispatch in a generic way, see:
https://discuss.python.org/t/multiple-dispatch-based-on-typing-overload/26197/17.

Even overloads are tricky to determine and very slightly between type checking libraries, see for example:
https://microsoft.github.io/pyright/#/type-concepts-advanced?id=overloads

For the stated reasons, we choose to implement a very simple form of typechecking for overloaded functions that
works as follows:

1. The existing overloads are determined using ``get_overloads``
2. The overloads are iterated in definition-order until a matching overload is found.
3. If a matching overload is found (typecheck successful) return the result, otherwise raise an error.
"""

import sys
import warnings
from functools import wraps
from inspect import signature

from beartype._util.func.arg.utilfuncargiter import ArgKind, iter_func_args
from beartype.roar import BeartypeCallHintParamViolation, BeartypeCallHintReturnViolation
from beartype.typing import Any, Iterable, Sequence, Tuple
from jaxtyping import TypeCheckError  # type: ignore[reportGeneralTypeIssues]
from typing_extensions import get_overloads

from ._typecheck import CallableAnyT, raise_if_missing_annotation, typecheck

__all__ = [
    "typecheck_overload",
]


class UnavailableOverloadError(Exception): ...


class MissingOverloadError(Exception): ...


class IncompatibleOverloadError(Exception): ...


def typecheck_overload(fn: CallableAnyT) -> CallableAnyT:
    """Ensure that an implementing function satisfied at least one of its defined overloads.

    To check if the implementing function satisfies one of the overloads, we iterate over all overloads (in definition
    order) and check if the signature matches the provided ``*args`` and ``**kwargs``. If the signature matches,
    the implementing function ``fn`` is called and the return value is cached because we also type-check the return
    value and the first return value might not satisfy the return type for the current overload. If further signatures
    match an already computed ``fn``, we directly return the cached return value and check if the return type matches
    the current overload.

    :param fn: An implementing function for @overload-decorated specifications.
    :return: Typechecked function or method.
    :raises: MissingAnnotationError if ``fn`` does not contain ``__annotations__`` (is no function or method).
    :raises: MissingOverloadError if no overload can be determined for ``fn``.
    :raises: IncompatibleOverloadError if one of the @overload functions is incompatible with the signature of ``fn``.
    :raises: UnavailableOverloadError if no overload matches the input arguments to ``fn``.
    """
    # this check must be before the ``len(fn.__annotations__)``, to ensure ``fn`` has ``__annotations__``
    raise_if_missing_annotation(fn)

    # make sure that the implementing fn is not annotated
    if len(fn.__annotations__) > 0:
        warn_if_unexpected_annotation(fn)

    # we are not yet sure if we are dealing with a function, therefore we wrap ``get_overloads`` safely
    overloads = safe_get_overloads(fn)

    # make sure that at least one overload is found (registered)
    if len(overloads) == 0:
        raise_if_no_overload(fn)

    # make sure that overloads contain equal parameters as in the implementation
    raise_if_incompatible_overload(fn, overloads)

    # override the empty overload functions with the implementing function body
    # this is required for later type checking of the return values
    for f in overloads:
        f.__code__ = fn.__code__

    # pre-wrap the overloads for later retrieval in the perf-critical wrapper function
    typed_overloads = tuple(typecheck(f, skip_annotation_check=True) for f in overloads)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for f in typed_overloads:
            try:
                return f(*args, **kwargs)
            except (BeartypeCallHintParamViolation, BeartypeCallHintReturnViolation, TypeCheckError):
                continue

        raise_unavailable_overload(fn, overloads, args, kwargs)

    return wrapper


def raise_unavailable_overload(
    fn: CallableAnyT,
    overloads: Sequence[CallableAnyT],
    args: Any,
    kwargs: Any,
) -> None:
    arg_msg = "without arguments"
    if (has_args := len(args) > 0) ^ (len(kwargs) > 0):
        arg_msg = f"with arguments '{args if has_args else kwargs}'"

    if len(args) > 0 and len(kwargs) > 0:
        arg_msg = f"with arguments '{args}' and keywords '{kwargs}'"

    msg = (
        f"No suitable overload was found for @typecheck_overload-decorated function '{fn.__qualname__}' {arg_msg}, "
        f"the available overloads are:\n" + "\n".join(str(signature(o)) for o in overloads)
    )
    raise UnavailableOverloadError(msg)


def warn_if_unexpected_annotation(fn: CallableAnyT) -> None:
    msg = (
        f"The function implementing @overload-decorated definitions should not contain type annotations, "
        f"because the behaviour is not defined for @typecheck_overload, but found annotations '{fn.__annotations__}' "
        f"which are ignored."
    )
    warnings.warn(msg, stacklevel=2)


def raise_if_no_overload(fn: CallableAnyT) -> None:
    if sys.version_info < (3, 11):
        additional_help = (
            "Did you use 'from typing import overload'? If this is the case, use 'from typing_extensions import "
            "overload' instead. Using @overload from typing before Python 3.11 does not make the overload visible "
            "to safecheck. When you upgrade to Python 3.11, you'll be able to use 'from typing import overload'."
        )
    else:
        additional_help = "Did you forget to use '@overload'?"

    msg = f"Could not find any overload for '{fn.__qualname__}'." + additional_help
    raise MissingOverloadError(msg)


def raise_if_incompatible_overload(fn: CallableAnyT, overloads: Iterable[CallableAnyT]) -> None:
    args_fn = tuple(iter_func_args_no_default_value(fn))
    for fn_overload in overloads:
        args_ov = tuple(iter_func_args_no_default_value(fn_overload))
        if args_ov != args_fn:
            signature_fn = signature(fn)
            signature_overload = signature(fn_overload)
            msg = (
                "Signature of this @overload-decorated function is not compatible with the implementation, found an "
                f"implementation with signature '{signature_fn}' and an incompatible overload with signature "
                f"'{signature_overload}', make sure that the @overload-decorated functions share the signature "
                f"with the implementing function."
            )
            raise IncompatibleOverloadError(msg)


def iter_func_args_no_default_value(fn: CallableAnyT) -> Iterable[Tuple[ArgKind, str]]:
    # it should be safe to use is_unwrap=False, because we don't use the default value
    for kind, name, *_ in iter_func_args(fn, is_unwrap=False):
        # extract the argument kind and name from the resulting tuple
        yield kind, name


def safe_get_overloads(fn: CallableAnyT) -> Sequence[CallableAnyT]:
    try:
        return get_overloads(fn)
    except AttributeError:
        msg = (
            f"Could not determine overloads for '{fn}' because of missing attributes, maybe the object is no "
            f"function or method?"
        )
    raise MissingOverloadError(msg)
