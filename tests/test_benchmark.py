from typing_extensions import overload

import numpy
from beartype import beartype

from safecheck import *

args = list(range(10))
args_shaped = numpy.random.randn(10, 100)  # dim0=number of args, dim1=size of arg


def decorate(f):
    return f


def f_basic(*_: int) -> None:
    ...


def f_shaped(*_: Shaped[NumpyArray, "n"]) -> None:
    ...


@overload
def f_overload(*_: int) -> None:
    ...


def f_overload(*_):
    ...


@overload
def f_overload_shaped(*_: Shaped[NumpyArray, "n"]) -> None:
    ...


def f_overload_shaped(*_):
    ...


def test_no_overhead(benchmark):
    benchmark(f_basic, *args)


def test_no_overhead_shaped(benchmark):
    benchmark(f_shaped, *args_shaped)


def test_minimal_overhead(benchmark):
    f = decorate(f_basic)
    benchmark(f, *args)


def test_minimal_overhead_shaped(benchmark):
    f = decorate(f_shaped)
    benchmark(f, *args_shaped)


def test_beartype(benchmark):
    f = beartype(f_basic)
    benchmark(f, *args)


def test_beartype_shaped(benchmark):
    f = beartype(f_shaped)
    benchmark(f, *args_shaped)


def test_typecheck(benchmark):
    f = typecheck(f_basic)
    benchmark(f, *args)


def test_typecheck_shaped(benchmark):
    f = typecheck(f_shaped)
    benchmark(f, *args_shaped)


def test_typecheck_overload(benchmark):
    f = typecheck_overload(f_overload)
    benchmark(f, *args)


def test_typecheck_overload_shaped(benchmark):
    f = typecheck_overload(f_overload_shaped)
    benchmark(f, *args_shaped)
