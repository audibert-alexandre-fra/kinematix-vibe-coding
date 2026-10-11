"""Registre des backends : permet de remplacer le modèle de pose sans toucher au reste."""

from __future__ import annotations

from collections.abc import Callable

from .base import PoseBackend


def _instanthmr(**kw) -> PoseBackend:
    from .instanthmr import InstantHMRBackend

    return InstantHMRBackend(**kw)


def _sam3d(**kw) -> PoseBackend:
    from .sam3d import Sam3DBackend

    return Sam3DBackend(**kw)


def _metrabs(**kw) -> PoseBackend:
    from .metrabs import MetrabsBackend

    return MetrabsBackend(**kw)


def _cached(name: str) -> Callable[..., PoseBackend]:
    def factory(**kw) -> PoseBackend:
        from .cached import CachedBackend

        return CachedBackend(name, **kw)

    return factory


_BACKENDS: dict[str, Callable[..., PoseBackend]] = {
    "instanthmr": _instanthmr, "sam3d": _sam3d, "metrabs": _metrabs,
    # modèles exécutés dans un environnement séparé (tools/run_<nom>.py) puis relus depuis le cache
    "mediapipe": _cached("mediapipe"), "rtmw3d": _cached("rtmw3d"), "nlf": _cached("nlf"),
}


def register_backend(name: str, factory: Callable[..., PoseBackend]) -> None:
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str, **kwargs) -> PoseBackend:
    if name not in _BACKENDS:
        raise KeyError(f"Backend inconnu {name!r}. Disponibles : {available_backends()}")
    return _BACKENDS[name](**kwargs)
