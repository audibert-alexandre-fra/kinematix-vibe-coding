"""Registre des backends : permet de remplacer le modèle de pose sans toucher au reste."""

from __future__ import annotations

from collections.abc import Callable

from .base import PoseBackend


def _instanthmr(**kw) -> PoseBackend:
    from .instanthmr import InstantHMRBackend

    return InstantHMRBackend(**kw)


_BACKENDS: dict[str, Callable[..., PoseBackend]] = {"instanthmr": _instanthmr}


def register_backend(name: str, factory: Callable[..., PoseBackend]) -> None:
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str, **kwargs) -> PoseBackend:
    if name not in _BACKENDS:
        raise KeyError(f"Backend inconnu {name!r}. Disponibles : {available_backends()}")
    return _BACKENDS[name](**kwargs)
