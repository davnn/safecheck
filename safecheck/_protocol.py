from collections.abc import Callable
from inspect import Parameter, _empty, signature  # type: ignore[reportPrivateUsage]
from typing import Any

from ._typecheck import typecheck

__all__ = [
    "implements",
    "protocol",
]

CallableAny = Callable[..., Any]


class FunctionProtocol:
    def __init__(
        self,
        return_annotation: type,
        parameters: list[Parameter],
    ) -> None:
        super().__init__()
        self.return_annotation = return_annotation
        self.parameters = parameters


class InvalidProtocolError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class ProtocolImplementationError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def protocol(func: CallableAny) -> FunctionProtocol:
    sig = signature(func)
    params = list(sig.parameters.values())
    if sig.return_annotation is _empty:
        msg = "Cannot construct a protocol with missing return type annotation."
        raise InvalidProtocolError(msg)

    for parameter in params:
        if parameter.annotation is _empty:
            msg = f"Cannot construct a protocol with missing type annotation, found {parameter}."
            raise InvalidProtocolError(msg)

        if parameter.default is not _empty:
            msg = f"Unexpected default value found in protocol definition, found {parameter}."
            raise InvalidProtocolError(msg)

    return FunctionProtocol(sig.return_annotation, params)


def implements(protocol: FunctionProtocol) -> Callable[[CallableAny], CallableAny]:
    if not isinstance(protocol, FunctionProtocol):  # type: ignore[reportUnnecessaryIsInstance]
        msg = (
            f"A protocol implementation using `implements` expects a FunctionProtocol parameter, "
            f"but found {type(protocol)}. Did you use `@implements` without parameters? Use "
            f"@implements(protocol) instead."
        )
        raise ProtocolImplementationError(msg)

    def decorator(func: CallableAny) -> CallableAny:
        sig = signature(func)
        size = len(protocol.parameters)

        # check if the updated return annotation matches the protocol return annotation
        return_annotation = protocol.return_annotation if sig.return_annotation is _empty else sig.return_annotation
        if return_annotation != (proto_return := protocol.return_annotation):
            msg = (
                f"Cannot implement a protocol without matching return types, but found return type "
                f"{return_annotation} for a protocol with return type {proto_return}."
            )
            raise ProtocolImplementationError(msg)

        # check if the updated shared parameters exactly match the protocol parameters
        sig_params = list(sig.parameters.values())
        shared_params = update_annotations(protocol.parameters, sig_params)
        if strip_defaults(shared_params[:size]) != (proto_params := protocol.parameters):
            msg = (
                f"Cannot implement a protocol without matching parameter types, but found parameters "
                f"{sig_params} for a protocol with parameters {proto_params}."
            )
            raise ProtocolImplementationError(msg)

        # check if the other parameters all have default values
        other_params = sig_params[size:]
        if any(p.default is _empty for p in other_params):
            msg = (
                f"Cannot implement a protocol that requires substitution, if any parameters not "
                f"included in the protocol do not have a default value, found: {other_params}."
            )
            raise ProtocolImplementationError(msg)

        # replace the function signature
        final_parameters = shared_params + other_params
        func.__signature__ = sig.replace(  # type: ignore[reportFunctionMemberAccess]
            parameters=final_parameters,
            return_annotation=return_annotation,
        )

        # replace the function annotations (used by runtime type checker)
        param_annotations = {p.name: p.annotation for p in final_parameters if p.annotation is not _empty}
        return_annotation = {} if return_annotation is _empty else {"return": return_annotation}
        func.__annotations__ = param_annotations | return_annotation
        return typecheck(func)

    return decorator


def strip_defaults(params: list[Parameter]) -> list[Parameter]:
    params = params.copy()
    """Strip the default values for the parameters in the list, which are irrelevant for the comparison."""
    for param in params:
        setattr(param, "_default", _empty)  # noqa[B010]

    return params


def update_annotations(reference: list[Parameter], params: list[Parameter]) -> list[Parameter]:
    for ref, param in zip(reference, params):
        if param.annotation is _empty:
            setattr(param, "_annotation", ref.annotation)  # noqa[B010]
    return params
