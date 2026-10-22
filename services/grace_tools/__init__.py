"""Grace's tool modules (ADR 0021). Every module here is imported at start and registers its
tools in registry.TOOLS; adding a tool group means adding a module, nothing else."""

import importlib
import pkgutil

from . import registry


def load():
    for m in pkgutil.iter_modules(__path__):
        if m.name != "registry" and not m.name.startswith("_"):
            importlib.import_module(f"{__name__}.{m.name}")
    return registry.TOOLS
