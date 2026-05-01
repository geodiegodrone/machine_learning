"""Build a lightweight catalog of notebooks and datasets."""

from __future__ import annotations

from pathlib import Path


def build_catalog(root: Path = Path(".")) -> list[dict[str, str]]:
    """Return repository artifacts grouped by type."""
    catalog: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or path.is_dir():
            continue
        if path.suffix.lower() in {".ipynb", ".csv"}:
            catalog.append(
                {
                    "path": path.as_posix(),
                    "type": "notebook" if path.suffix.lower() == ".ipynb" else "dataset",
                }
            )
    return catalog


def main() -> None:
    """Print catalog to stdout."""
    for item in build_catalog():
        print(f"{item['type']:8} {item['path']}")


if __name__ == "__main__":
    main()
