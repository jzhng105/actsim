"""Tool groups. Each module exposes ``register(server)``."""

from . import analytics, artifacts, claims, distributions, fitting, simulation

REGISTRARS = (
    artifacts.register,
    distributions.register,
    fitting.register,
    simulation.register,
    claims.register,
    analytics.register,
)


def register_all(server) -> None:
    """Attach every tool group to the server."""
    for registrar in REGISTRARS:
        registrar(server)
