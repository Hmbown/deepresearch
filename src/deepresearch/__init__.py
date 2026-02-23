"""deepresearch package exports."""

from __future__ import annotations

from .env import bootstrap_env

# Load project-local .env once for package consumers (CLI, imports, notebooks).
bootstrap_env(override=False)

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from .graph import app

        return app
    raise AttributeError(name)
