from typing import Any

import pytest

from safecheck import *


def valid_protocol():
    def pos_kw(a: int, b: float) -> float:
        return a + b

    def kw_only(*, a: int, b: float) -> float:
        ...

    def pos_only(a: int, /, b: float) -> float:
        ...

    def pos_varargs(*args: Any) -> Any:
        ...

    def kw_varargs(**kwargs: Any) -> Any:
        ...

    return [pos_kw, kw_only, pos_only, pos_varargs, kw_varargs]


def valid_implementation():
    # same specification as the protocol
    def f_equal(a: int, b: float) -> float:
        ...

    # implementation with default parameter
    def f_default(a: int, b: float = 0.0) -> float:
        ...

    # implementation without annotations
    def f_empty(a, b):
        ...

    # implementation with partial annotations
    def f_partial(a: int, b) -> float:
        ...

    return [f_equal, f_default, f_empty, f_partial]


@pytest.mark.parametrize("f", valid_implementation())
def test_valid_implementation(f):
    @protocol
    def p(a: int, b: float) -> float:
        ...

    implements(p)(f)


def invalid_protocol():
    def missing_parameter_annotation(a, b: float) -> float:
        ...

    def missing_return_annotation(a: int, b: float):
        ...

    def unexpected_default_value(a: int, b: float = 0.0) -> float:
        ...

    return [missing_parameter_annotation, missing_return_annotation, unexpected_default_value]


@pytest.mark.parametrize("f", invalid_protocol())
def test_invalid_protocol(f):
    from safecheck._protocol import InvalidProtocolError

    with pytest.raises(InvalidProtocolError):
        protocol(f)


def invalid_implementation():
    def wrong_parameter(a: int, b: int) -> float:
        ...

    def wrong_return(a: int, b: float) -> int:
        ...

    def wrong_kind_kw(a, *, b):
        ...

    def wrong_kind_pos(a, /, b):
        ...

    def additional_missing_default(a, b, c):
        ...

    return [wrong_parameter, wrong_return, wrong_kind_kw, wrong_kind_pos, additional_missing_default]


@pytest.mark.parametrize("f", invalid_implementation())
def test_invalid_implementation(f):
    from safecheck._protocol import ProtocolImplementationError

    @protocol
    def p(a: int, b: float) -> float:
        ...

    with pytest.raises(ProtocolImplementationError):
        implements(p)(f)


def test_invalid_implementation_call():
    from safecheck._protocol import ProtocolImplementationError

    with pytest.raises(ProtocolImplementationError):
        implements(None)
