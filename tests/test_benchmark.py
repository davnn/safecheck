import numpy
from beartype import beartype

from safecheck import *

args = list(range(10))
args_shaped = numpy.random.randn(10, 100)  # dim0=number of args, dim1=size of arg


def decorate(f):
    return f


def f(*_: int) -> None:
    ...


def f_shaped(*_: Shaped[NumpyArray, "n"]) -> None:
    ...


def test_no_overhead(benchmark):
    benchmark(f, *args)


def test_no_overhead_shaped(benchmark):
    benchmark(f_shaped, *args_shaped)


def test_minimal_overhead(benchmark):
    benchmark(decorate(f), *args)


def test_minimal_overhead_shaped(benchmark):
    benchmark(decorate(f_shaped), *args_shaped)


def test_beartype(benchmark):
    benchmark(beartype(f), *args)


def test_beartype_shaped(benchmark):
    benchmark(beartype(f_shaped), *args_shaped)


def test_typecheck(benchmark):
    benchmark(typecheck(f), *args)


def test_typecheck_shaped(benchmark):
    benchmark(typecheck(f_shaped), *args_shaped)


def test_dispatch(benchmark):
    dispatch = Dispatcher()
    benchmark(dispatch(f), *args)


def test_dispatch_shaped(benchmark):
    benchmark(dispatch(f_shaped), *args_shaped)


def test_protocol(benchmark):
    benchmark(implements(protocol(f))(f), *args)


def test_protocol_shaped(benchmark):
    benchmark(implements(protocol(f_shaped))(f_shaped), *args_shaped)
