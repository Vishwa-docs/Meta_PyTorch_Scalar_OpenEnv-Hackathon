"""Validator compatibility entrypoint."""

from backend.main import app


def main():
    """Return ASGI app for validator multi-mode checks."""
    return app


if __name__ == "__main__":
    main()

