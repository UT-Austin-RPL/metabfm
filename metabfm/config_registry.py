from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class ConfigRegistry:
    """Enumerates .gin configs in a directory and resolves them by name.

    Usage:
        registry = ConfigRegistry(configs_dir)
        registry.list()             # ['agent/tdjepa', 'env/wind', ...]
        registry.get("env/wind")    # '/abs/path/to/env/wind.gin'
    """

    def __init__(self, configs_dir: str):
        self._root = Path(configs_dir)
        if not self._root.is_dir():
            raise FileNotFoundError(f"Config directory not found: {self._root}")

    def list(self) -> list[str]:
        names = []
        for gin_file in sorted(self._root.rglob("*.gin")):
            rel = gin_file.relative_to(self._root)
            name = str(rel.with_suffix(""))
            names.append(name)
        return names

    def get(self, name: str) -> str:
        if not name.endswith(".gin"):
            name = name + ".gin"
        path = self._root / name
        if not path.is_file():
            available = self.list()
            raise KeyError(
                f"Config '{name}' not found in {self._root}.\n"
                f"Available configs: {available}"
            )
        return str(path.resolve())

    def __contains__(self, name: str) -> bool:
        if not name.endswith(".gin"):
            name = name + ".gin"
        return (self._root / name).is_file()

    def completer(self, prefix, **kwargs):
        """argcomplete completer — returns matching config names."""
        return [n for n in self.list() if n.startswith(prefix)]

    def __repr__(self) -> str:
        return f"ConfigRegistry({self._root}, configs={self.list()})"
