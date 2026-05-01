"""Repository integrity tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "catalog_repository", Path("scripts/catalog_repository.py")
)
catalog_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(catalog_module)
build_catalog = catalog_module.build_catalog


def test_catalog_finds_notebooks_and_datasets() -> None:
    """Catalog should find original coursework assets."""
    catalog = build_catalog(Path("."))
    paths = {item["path"] for item in catalog}
    assert "07MBID_Práctica_2_diego_pulido.ipynb" in paths
    assert "actividad1/dengue_features_train.csv" in paths
    assert "actividad1/dengue_labels_train.csv" in paths


def test_dengue_train_label_files_exist() -> None:
    """Dengue practice data files should remain available."""
    assert Path("actividad1/dengue_features_train.csv").exists()
    assert Path("actividad1/dengue_labels_train.csv").exists()
