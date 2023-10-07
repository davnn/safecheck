import time

import pytest
from beartype.roar import BeartypeCallHintParamViolation

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

from safecheck import *

benchmark_sleep = 0.001  # time in s per benchmark trial run
benchmark_args = numpy.random.randn(10, 100)  # dim0=number of args, dim1=size of arg

np_array = numpy.random.randint(low=0, high=1, size=(1,))
torch_array = torch.randint(low=0, high=1, size=(1,))
jax_array = jax.random.randint(key=jax.random.PRNGKey(0), minval=0, maxval=1, shape=(1,))

array_types = {TorchArray: torch_array, NumpyArray: np_array, JaxArray: jax_array}
array_types_str = {TorchArray: "torch", NumpyArray: "numpy", JaxArray: "jax"}
data_types = {
    TorchArray: {Float: torch_array.float(), Integer: torch_array.int(), Bool: torch_array.bool()},
    NumpyArray: {Float: np_array.astype(float), Integer: np_array.astype(int), Bool: np_array.astype(bool)},
    JaxArray: {Float: jax_array.astype(float), Integer: jax_array.astype(int), Bool: jax_array.astype(bool)},
}
data_types_str = {
    TorchArray: {Float: "torch_float", Integer: "torch_integer", Bool: "torch_bool"},
    NumpyArray: {Float: "numpy_float", Integer: "numpy_integer", Bool: "numpy_bool"},
    JaxArray: {Float: "jax_float", Integer: "jax_integer", Bool: "jax_bool"},
}


@pytest.mark.parametrize("array_type", array_types.keys())
def test_array_type(array_type):
    @shapecheck
    def f(array: Int[array_type, "..."]) -> Int[array_type, "..."]:
        return array

    array = array_types[array_type]
    assert f(array) == array

    # check that all other array types raise
    for k, other in array_types.items():
        if k == array_type:
            continue

        with pytest.raises(BeartypeCallHintParamViolation):
            f(other)


@pytest.mark.parametrize("array_type", data_types.keys())
@pytest.mark.parametrize("data_type", next(iter(data_types.values())).keys())
def test_data_type(array_type, data_type):
    @shapecheck
    def f(array: data_type[array_type, "..."]) -> data_type[array_type, "..."]:
        return array

    array = data_types[array_type][data_type]
    assert f(array) == array

    # check that all other array types raise
    for current_array_type, current_data_types in data_types.items():
        for current_data_type, current_array in current_data_types.items():
            if current_array_type == array_type and current_data_type == data_type:
                continue

            with pytest.raises(BeartypeCallHintParamViolation):
                f(current_array)


@pytest.mark.parametrize("array_type", data_types.keys())
def test_array_type_dispatch(array_type):
    dispatch = Dispatcher()

    @dispatch
    def f(_: Shaped[NumpyArray, "..."]) -> str:
        return "numpy"

    @dispatch
    def f(_: Shaped[TorchArray, "..."]) -> str:
        return "torch"

    @dispatch
    def f(_: Shaped[JaxArray, "..."]) -> str:
        return "jax"

    assert array_types_str[array_type] == f(array_types[array_type])


@pytest.mark.parametrize("array_type", data_types.keys())
def test_array_type_dispatch_with_typecheck(array_type):
    dispatch = Dispatcher()

    @dispatch
    @typecheck
    def f(_: Shaped[NumpyArray, "..."]) -> str:
        return "numpy"

    @dispatch
    @typecheck
    def f(_: Shaped[TorchArray, "..."]) -> str:
        return "torch"

    @dispatch
    @typecheck
    def f(_: Shaped[JaxArray, "..."]) -> str:
        return "jax"

    assert array_types_str[array_type] == f(array_types[array_type])


@pytest.mark.parametrize("array_type", data_types.keys())
def test_array_type_dispatch_with_shapecheck(array_type):
    dispatch = Dispatcher()

    @dispatch
    @shapecheck
    def f(_: Shaped[NumpyArray, "..."]) -> str:
        return "numpy"

    @dispatch
    @shapecheck
    def f(_: Shaped[TorchArray, "..."]) -> str:
        return "torch"

    @dispatch
    @shapecheck
    def f(_: Shaped[JaxArray, "..."]) -> str:
        return "jax"

    assert array_types_str[array_type] == f(array_types[array_type])


@pytest.mark.parametrize("array_type", data_types.keys())
@pytest.mark.parametrize("data_type", next(iter(data_types.values())).keys())
def test_data_type_dispatch(array_type, data_type):
    dispatch = Dispatcher()

    @dispatch
    def f(_: Float[NumpyArray, "..."]) -> str:
        return "numpy_float"

    @dispatch
    def f(_: Integer[NumpyArray, "..."]) -> str:
        return "numpy_integer"

    @dispatch
    def f(_: Bool[NumpyArray, "..."]) -> str:
        return "numpy_bool"

    @dispatch
    def f(_: Float[TorchArray, "..."]) -> str:
        return "torch_float"

    @dispatch
    def f(_: Integer[TorchArray, "..."]) -> str:
        return "torch_integer"

    @dispatch
    def f(_: Bool[TorchArray, "..."]) -> str:
        return "torch_bool"

    @dispatch
    def f(_: Float[JaxArray, "..."]) -> str:
        return "jax_float"

    @dispatch
    def f(_: Integer[JaxArray, "..."]) -> str:
        return "jax_integer"

    @dispatch
    def f(_: Bool[JaxArray, "..."]) -> str:
        return "jax_bool"

    assert data_types_str[array_type][data_type] == f(data_types[array_type][data_type])


def test_no_overhead(benchmark):
    def f(*_):
        time.sleep(benchmark_sleep)

    benchmark(f, *benchmark_args)


def test_minimal_decorator(benchmark):
    def decorate(f):
        return f

    @decorate
    def f(*_):
        time.sleep(benchmark_sleep)

    benchmark(f, *benchmark_args)


def test_typecheck_decorator(benchmark):
    @typecheck
    def f(*_: NumpyArray):
        time.sleep(benchmark_sleep)

    benchmark(f, *benchmark_args)


def test_shapecheck_decorator(benchmark):
    @shapecheck
    def f(*_: Shaped[NumpyArray, "n"]) -> None:
        time.sleep(benchmark_sleep)

    benchmark(f, *benchmark_args)


def test_dispatch_decorator(benchmark):
    dispatch = Dispatcher()

    @dispatch
    def f(*_: Shaped[NumpyArray, "n"]):
        time.sleep(benchmark_sleep)

    benchmark(f, *benchmark_args)
