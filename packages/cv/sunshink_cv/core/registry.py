from typing import Callable, Dict, Generic, Tuple, TypeVar


T = TypeVar("T")


class NamedComponentRegistry(Generic[T]):
    def __init__(self, domain: str):
        self.domain = domain
        self._entries: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, builder: Callable[..., T]) -> None:
        self._entries[name] = builder

    def build(self, name: str, *args, **kwargs) -> T:
        if name not in self._entries:
            raise KeyError(f"Unknown {self.domain} component: {name}")
        return self._entries[name](*args, **kwargs)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._entries.keys()))
