import configparser
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")


def _parse_value(value: str, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is tuple:
        items = [item.strip() for item in value.split(",")]

        if args and args[-1] is Ellipsis:
            item_type = args[0]
            return tuple(_parse_value(item, item_type) for item in items)

        if args:
            if len(items) != len(args):
                raise ValueError(
                    f"Expected {len(args)} tuple elements, got {len(items)}"
                )
            return tuple(
                _parse_value(item, item_type) for item, item_type in zip(items, args)
            )

        return tuple(items)

    if annotation is bool:
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")

    if annotation in (str, int, float):
        return annotation(value.strip())

    raise TypeError(f"Unsupported field type: {annotation}")


def parse_dataclass(section: Mapping[str, str], cls: type[T]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")

    type_hints = get_type_hints(cls)
    values = {}

    for field in fields(cls):
        if field.name not in section:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue

            raise ValueError(
                f"Missing required field {field.name!r} " f"for {cls.__name__}"
            )

        try:
            values[field.name] = _parse_value(
                section[field.name],
                type_hints[field.name],
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Error parsing {cls.__name__}.{field.name}: "
                f"{section[field.name]!r}"
            ) from exc

    return cls(**values)


class ModelConfig:
    def __init__(
        self,
        path: str | Path,
        schemas: Mapping[str, type],
    ):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path, encoding="utf-8")

        self._models = {
            name: parse_dataclass(parser[name], schema)
            for name, schema in schemas.items()
            if name in parser
        }

    def __getitem__(self, model_name: str) -> Any:
        return self._models[model_name]

    def __contains__(self, model_name: str) -> bool:
        return model_name in self._models

    def keys(self):
        return self._models.keys()
