from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin


class ValidationError(Exception):
    pass


@dataclass
class FieldInfo:
    default: Any = None
    default_factory: Optional[Callable[[], Any]] = None


def Field(default: Any = None, default_factory: Optional[Callable[[], Any]] = None) -> FieldInfo:
    return FieldInfo(default=default, default_factory=default_factory)


Validator = Tuple[str, Callable[[Type, Any], Any], bool]


def validator(field_name: str, *, pre: bool = False):
    def decorator(func: Callable[[Type, Any], Any]) -> Callable[[Type, Any], Any]:
        func.__validator_config__ = (field_name, func, pre)
        return func

    return decorator


class BaseModelMeta(type):
    def __new__(mcls, name, bases, namespace):
        validators: List[Validator] = []
        for attr, value in namespace.items():
            config = getattr(value, "__validator_config__", None)
            if config:
                validators.append(config)
        namespace["__validators__"] = validators
        return super().__new__(mcls, name, bases, namespace)


class BaseModel(metaclass=BaseModelMeta):
    __validators__: List[Validator]

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        for field, annotation in annotations.items():
            value = data.get(field, _missing)
            default = getattr(self.__class__, field, _missing)
            if isinstance(default, FieldInfo):
                if value is _missing:
                    if default.default_factory is not None:
                        value = default.default_factory()
                    else:
                        value = default.default
                default = _missing
            elif value is _missing and default is not _missing:
                value = default
            if value is _missing:
                raise ValidationError(f"Field '{field}' is required")
            value = _convert_type(value, annotation)
            value = self._apply_validators(field, value)
            setattr(self, field, value)

    def _apply_validators(self, field: str, value: Any) -> Any:
        for name, func, _ in self.__class__.__validators__:
            if name == field:
                value = func(self.__class__, value)
        return value

    def dict(self) -> Dict[str, Any]:
        return {
            key: _export_value(getattr(self, key))
            for key in getattr(self.__class__, "__annotations__", {})
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dict()!r})"


_missing = object()


def _convert_type(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if isinstance(value, annotation):
                return value
            if isinstance(value, dict):
                return annotation(**value)
            raise ValidationError(f"Cannot convert {value} to {annotation}")
        if annotation in (int, float, str):
            try:
                return annotation(value)
            except Exception as exc:
                raise ValidationError(f"Failed to cast {value} to {annotation}") from exc
        return value
    if origin in (list, List):
        (inner,) = get_args(annotation)
        return [_convert_type(item, inner) for item in value]
    if origin is dict:
        key_type, value_type = get_args(annotation) or (Any, Any)
        return {
            _convert_type(k, key_type) if key_type is not Any else k: _convert_type(v, value_type) if value_type is not Any else v
            for k, v in value.items()
        }
    if origin is tuple:
        args = get_args(annotation)
        return tuple(_convert_type(item, args[idx]) for idx, item in enumerate(value))
    if origin is Union:
        args = get_args(annotation)
        for inner in args:
            if inner is type(None) and value is None:
                return None
            try:
                return _convert_type(value, inner)
            except ValidationError:
                continue
        raise ValidationError(f"Value {value} does not match union {annotation}")
    return value


def _export_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.dict()
    if isinstance(value, list):
        return [_export_value(item) for item in value]
    return value
