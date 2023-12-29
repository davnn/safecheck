from types import GenericAlias

from enum import Enum

from typing import Union, Literal, TypedDict, Callable

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from typing_extensions import overload, runtime_checkable, Protocol, Annotated, ParamSpec, Concatenate

from jaxtyping import TypeCheckError

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

from safecheck import *
from safecheck._overload import MissingOverloadError

np_array = numpy.random.randint(low=0, high=1, size=(1,))
torch_array = torch.randint(low=0, high=1, size=(1,))
jax_array = jax.random.randint(key=jax.random.PRNGKey(0), minval=0, maxval=1, shape=(1,))


class SomeTypedDict(TypedDict):
    x: int
    y: int


class SomeLiteralEnum(Enum):
    x = 1


@runtime_checkable
class SomeProtocol(Protocol):
    __some_random_attribute__: str


class SomeProtocolClass:
    __some_random_attribute__ = "name"


SomeAnnotatedType = Annotated[tuple[float], Is[lambda x: x[0] < 0]]
OtherAnnotatedType = Annotated[tuple[float], Is[lambda x: x[0] > 0]]

P = ParamSpec("P", bound=int)
SomeCallableWithParamSpec = Callable[Concatenate[int, P], int]

basic_types = {
    int: 1,
    float: 1.0,
    complex: complex(1, 1),
    str: "1",
    bytes: b"1",
    Literal[SomeLiteralEnum.x]: SomeLiteralEnum.x,
    GenericAlias(list, (float,)): [1.0],
    SomeProtocol: SomeProtocolClass(),
    list[str]: ["1.0"],
    None: None,
    SomeTypedDict: {"x": 1, "y": 1},
    SomeAnnotatedType: tuple([-1.0]),
    OtherAnnotatedType: tuple([1.0]),
    SomeCallableWithParamSpec: lambda x, _: x,
}

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


@pytest.mark.parametrize("basic_type", basic_types.keys())
def test_basic_type_plain_function(basic_type):
    @typecheck
    def f(x: basic_type) -> basic_type:
        return x

    check_basic_type(f, basic_type)


@pytest.mark.parametrize("basic_type", basic_types.keys())
def test_basic_type_plain_method(basic_type):
    class A:
        @typecheck
        def f(self, x: basic_type) -> basic_type:
            return x

    check_basic_type(A().f, basic_type)


@pytest.mark.parametrize("basic_type", basic_types.keys())
def test_basic_type_overload_function(basic_type):
    @overload
    def f(x: basic_type) -> basic_type:
        ...

    @typecheck_overload
    def f(x):
        return x

    check_basic_type(f, basic_type)


@pytest.mark.parametrize("basic_type", basic_types.keys())
def test_basic_type_overload_method(basic_type):
    class A:
        @overload
        def f(self, x: basic_type) -> basic_type:
            ...

        @typecheck_overload
        def f(self, x):
            return x

    check_basic_type(A().f, basic_type)


def test_union_type_plain_function():
    union_type = Union[tuple(basic_types.keys())]

    @typecheck
    def f(array: union_type) -> union_type:
        return array

    for value in basic_types.values():
        assert f(value) == value


def test_parametrized_generic_type_plain_function():
    union_type = Union[tuple(basic_types.keys())]

    @typecheck
    def f(array: union_type) -> union_type:
        return array

    for value in basic_types.values():
        assert f(value) == value


@pytest.mark.parametrize("array_type", data_types.keys())
@pytest.mark.parametrize("data_type", next(iter(data_types.values())).keys())
def test_array_type_plain_function(array_type, data_type):
    @typecheck
    def f(array: data_type[array_type, "..."]) -> data_type[array_type, "..."]:
        return array

    check_array_type(f, array_type, data_type)


@pytest.mark.parametrize("array_type", data_types.keys())
@pytest.mark.parametrize("data_type", next(iter(data_types.values())).keys())
def test_array_type_overload_function(array_type, data_type):
    @overload
    def f(array: data_type[array_type, "..."]) -> data_type[array_type, "..."]:
        ...

    @typecheck_overload
    def f(array):
        return array

    check_array_type(f, array_type, data_type)


@pytest.mark.parametrize("array_type", data_types.keys())
@pytest.mark.parametrize("data_type", next(iter(data_types.values())).keys())
def test_array_type_plain_method(array_type, data_type):
    class A:
        @typecheck
        def f(self, array: data_type[array_type, "..."]) -> data_type[array_type, "..."]:
            return array

    check_array_type(A().f, array_type, data_type)


@pytest.mark.parametrize("array_type", data_types.keys())
@pytest.mark.parametrize("data_type", next(iter(data_types.values())).keys())
def test_array_type_overload_method(array_type, data_type):
    class A:
        @overload
        def f(self, array: data_type[array_type, "..."]) -> data_type[array_type, "..."]:
            ...

        @typecheck_overload
        def f(self, array):
            return array

    check_array_type(A().f, array_type, data_type)


def check_basic_type(f, basic_type):
    value = basic_types[basic_type]
    assert f(value) == value

    # check that all other basic types raise
    for k, other in basic_types.items():
        if k == basic_type:
            continue

        with pytest.raises(
            (
                TypeCheckError,
                BeartypeCallHintParamViolation,
                MissingOverloadError,
            )
        ):
            f(other)


def check_array_type(f, array_type, data_type):
    array = data_types[array_type][data_type]
    assert f(array) == array

    # check that all other array types raise
    for current_array_type, current_data_types in data_types.items():
        for current_data_type, current_array in current_data_types.items():
            if current_array_type == array_type and current_data_type == data_type:
                continue

            with pytest.raises((TypeCheckError, BeartypeCallHintParamViolation, MissingOverloadError)):
                f(current_array)
