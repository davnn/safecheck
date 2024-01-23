[![Check Status](https://github.com/davnn/safecheck/actions/workflows/check.yml/badge.svg)](https://github.com/davnn/safecheck/actions?query=workflow%3Acheck)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/davnn/safecheck/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/davnn/safecheck/releases)
![Coverage Report](https://raw.githubusercontent.com/davnn/safecheck/main/assets/coverage.svg)

# safecheck

Opinionated combination of typechecking libraries. Safecheck is a (very) minimal wrapper of the following libraries to
provide a unified and simple-to-use interface:

- typechecking [beartype](https://github.com/beartype/)
- shapechecking [jaxtyping](https://github.com/google/jaxtyping)

Safecheck configures a *unified* ``typecheck`` decorator that invokes ``beartype.beartype`` if the function annotations
do not contain any ``jaxtyping``-related types. If the function contains ``jaxtyping``-related types ``typecheck``
invokes ``jaxtyping.jaxtyped`` with ``beartype.beartype`` as a runtime type-checker.
``safecheck`` is highly-efficient, it adds no measurable overhead to the underlying type and shape checking logic.

One of the goals of ``safecheck`` is to abstract over the runtime-typechecker and -shapechecker such that the concrete
implementation can be swapped without requiring changes to the codebase.

We re-export most of the functionality of ``beartype`` and ``jaxtyping``, and it might be a good idea to disallow
imports from ``beartype`` and ``jaxtyping`` if you are using safecheck, e.g. using
[ruff](https://docs.astral.sh/ruff/rules/banned-api/) or [Flake8](https://pypi.org/project/flake8-tidy-imports/).

To unify the ``jaxtyping.Array`` interface, we export ``jax.Array as JaxArray`` if
[Jax](https://github.com/google/jax) is available, ``torch.Tensor as TorchArray`` if
[PyTorch](https://github.com/pytorch/pytorch) is available and ``numpy.ndarray as NumpyArray`` if
[NumPy](https://github.com/numpy/numpy) is available.

In addition to the unified ``typecheck``, the library provides a ``typecheck_overload`` decorator.

### API

#### decorators

    typecheck(fn)

typechecks a function without jaxtyping annotations, otherwise additionally shapecheck the function.

    typecheck_overload(fn)

ensures that an implementing function satisfied at least one of its defined overloads.

#### introspection

    is_instance(obj, hint)

like ``isinstance(...)``, but [better](https://beartype.readthedocs.io/en/latest/api_door/#beartype.door.is_bearable).

    assert_instance(obj, hint)

like ``assert isinstance(...)``, but
[better](https://beartype.readthedocs.io/en/latest/api_door/#beartype.door.die_if_unbearable).

    is_subtype(subhint, superhint)

tests if a type is a subtype of
[another type](https://beartype.readthedocs.io/en/latest/api_door/#beartype.door.is_subhint).

#### validators

Validators enable runtime validation using ``typing.Annotated``, but these annotations are not enforced by any static
type checker and always require a runtime ``@typecheck``.

    Is

for example: ``Annotated[str, Is[lambda x: x > 0)]]``

    IsAttr

for example: ``Annotated[NumpyArray, IsAttr["ndim", IsEqual[1]]]``

    IsEqual

for example: ``Annotated[list, IsEqual[list(range(42))]]``

    IsSubclass

for example: ``Annotated[type, IsSubclass[str, bytes]]``

    IsInstance

for example: ``Annotated[object, IsInstance[str, bytes]]``

#### union array types

Exported union array types from ``safecheck``.

    Shaped      # Any type at all (e.g. object or string)
    Num         # Any integer, unsigned integer, floating, or complex
    Real        # Any integer, unsigned integer or floating
    Inexact     # Any floating or complex
    Float       # Any floating point
    Complex     # Any complex
    Integer     # Any integer or unsigned integer
    UInt        # Any unsigned integer
    Int         # Any signed integer

#### concrete array types

Exported array types from ``safecheck``.

    Int8
    Int16
    Int32
    Int64
    Float16
    Float32
    Float64
    Bool
    UInt8
    UInt16
    UInt32
    UInt64
    Complex64
    Complex128

### Examples

Type-checking a simple function.

```python
from safecheck import typecheck


@typecheck
def f(x: int) -> int:
    return x

# f(1) -> 1
# f("1") -> fails
```

Type-checking a simple method.

```python
from safecheck import typecheck


class A:
    @typecheck
    def f(self, x: int) -> int:
        return x

# A().f(1) -> 1
# A().f("1") -> fails
```

Shape-checking a simple function.

```python
from safecheck import typecheck, NumpyArray, Integer


@typecheck
def f(x: Integer[NumpyArray, "n"]) -> Integer[NumpyArray, "n"]:
    return x

# import numpy as np
# f(np.array([1, 2, 3, 4, 5])) -> array([1, 2, 3, 4, 5])
# f(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) -> fails
# f(np.array([[1], [2], [3], [4], [5]])) -> fails
```

Shape-checking a simple method.

```python
from safecheck import typecheck, NumpyArray, Integer


class A:
    @typecheck
    def f(self, x: Integer[NumpyArray, "n"]) -> Integer[NumpyArray, "n"]:
        return x

# import numpy as np
# A().f(np.array([1, 2, 3, 4, 5])) -> array([1, 2, 3, 4, 5])
# A().f(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) -> fails
# A().f(np.array([[1], [2], [3], [4], [5]])) -> fails
```

Type-checking an overloaded function.

```python
from typing_extensions import overload  # python < 3.11, otherwise ``from typing import overload``
from safecheck import typecheck_overload


@overload
def f(x: int) -> int:
    ...


@typecheck_overload
def f(x):
    return x

# f(1) -> 1
# f("1") -> fails
```

Type-checking an overloaded method.

```python
from typing_extensions import overload  # python < 3.11, otherwise ``from typing import overload``
from safecheck import typecheck_overload


class A:
    @overload
    def f(self, x: int) -> int:
        ...

    @typecheck_overload
    def f(self, x):
        return x

# A().f(1) -> 1
# A().f("1") -> fails
```

Shape-checking an overloaded function.

```python
from typing_extensions import overload  # python < 3.11, otherwise ``from typing import overload``
from safecheck import typecheck_overload, NumpyArray, Integer


@overload
def f(x: Integer[NumpyArray, "n"]) -> Integer[NumpyArray, "n"]:
    ...


@typecheck_overload
def f(x):
    return x

# import numpy as np
# f(np.array([1, 2, 3, 4, 5])) -> array([1, 2, 3, 4, 5])
# f(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) -> fails
# f(np.array([[1], [2], [3], [4], [5]])) -> fails
```

Shape-checking an overloaded method.

```python
from typing_extensions import overload  # python < 3.11, otherwise ``from typing import overload``
from safecheck import typecheck_overload, NumpyArray, Integer


class A:
    @overload
    def f(self, x: Integer[NumpyArray, "n"]) -> Integer[NumpyArray, "n"]:
        ...

    @typecheck_overload
    def f(self, x):
        return x

# import numpy as np
# A().f(np.array([1, 2, 3, 4, 5])) -> array([1, 2, 3, 4, 5])
# A().f(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) -> fails
# A().f(np.array([[1], [2], [3], [4], [5]])) -> fails
```
