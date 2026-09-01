from collections import defaultdict
from typing import Callable, Dict

_REG: Dict[str, Dict[str, type]] = defaultdict(dict)

def register(kind: str, name: str | None = None) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        key = name or cls.__name__.lower()
        _REG[kind][key] = cls
        return cls
    return deco

def build(kind: str, name: str, cfg: Dict) -> Callable[[type], type]:
    try:
        cls = _REG[kind][name]
    except KeyError:
        raise KeyError(
            f"{kind} '{name}' not found. Available: {list(_REG[kind].keys())}"
        )
    # Classes own their construction:
    return cls.build_from_config(cfg)


