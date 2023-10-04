import pytest
from beartype.roar import BeartypeCallHintParamViolation

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

from safecheck import *

np_array = numpy.random.randint(low=0, high=1, size=(1,))
torch_array = torch.randint(low=0, high=1, size=(1,))
jax_array = jax.random.randint(key=jax.random.PRNGKey(0), minval=0, maxval=1, shape=(1,))

array_types = {TorchArray: torch_array, NumpyArray: np_array, JaxArray: jax_array}

data_types = {
    TorchArray: {Float: torch_array.float(), Integer: torch_array.int(), Bool: torch_array.bool()},
    NumpyArray: {Float: np_array.astype(float), Integer: np_array.astype(int), Bool: np_array.astype(bool)},
    JaxArray: {Float: jax_array.astype(float), Integer: jax_array.astype(int), Bool: jax_array.astype(bool)},
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
