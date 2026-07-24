"""Parity tests for scripts/state_toml_loader.py against cell_load.

Skipped automatically where `cell_load` is not installed (e.g. the base CellFlow
test env); runs inside the expansion image where `cell-load` is a dependency.
"""

import os
import sys
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cell_load")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from state_toml_loader import load_state_toml_adatas  # noqa: E402

GENES = ["non-targeting", "g1", "g2", "g3", "g4"]


def _make_h5ad(path, celltypes, n_per=30, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    gene, ct = [], []
    for c in celltypes:
        for g in GENES:
            gene += [g] * n_per
            ct += [c] * n_per
    n = len(gene)
    obs = pd.DataFrame({
        "gene": pd.Categorical(gene),
        "cell_type": pd.Categorical(ct),
        "gem_group": pd.Categorical(["b0"] * n),
    })
    a = ad.AnnData(X=rng.random((n, 5)).astype("float32"), obs=obs)
    a.obsm["X_hvg"] = rng.random((n, dim)).astype("float32")
    a.write_h5ad(path)


@pytest.fixture()
def toml_and_files(tmp_path):
    fA, fB = tmp_path / "dsA.h5ad", tmp_path / "dsB.h5ad"
    _make_h5ad(fA, ["neuron", "other"], seed=1)
    _make_h5ad(fB, ["glia"], seed=2)
    toml_path = tmp_path / "split.toml"
    toml_path.write_text(f'''
[datasets]
dsA = "{fA}"
dsB = "{fB}"
[training]
dsA = "train"
dsB = "train"
[zeroshot]
[fewshot]
[fewshot."dsA.neuron"]
val = ["g1"]
test = ["g2"]
''')
    return str(toml_path)


def _cell_load_counts(toml_path):
    from cell_load.data_modules import PerturbationDataModule

    dm = PerturbationDataModule(
        toml_config_path=toml_path, embed_key="X_hvg", pert_col="gene",
        cell_type_key="cell_type", control_pert="non-targeting", batch_col="gem_group",
        random_seed=42, should_yield_control_cells=True,
    )
    dm.setup()
    counts = {}
    for split, subs in [("train", dm.train_datasets), ("val", dm.val_datasets), ("test", dm.test_datasets)]:
        per_file = defaultdict(set)
        for s in subs:
            per_file[str(s.dataset.h5_path)].update(int(i) for i in np.asarray(s.indices))
        counts[split] = sum(len(v) for v in per_file.values())
    return counts


def test_split_parity(toml_and_files):
    data = load_state_toml_adatas(toml_and_files, embed_key="X_hvg", seed=42)
    ref = _cell_load_counts(toml_and_files)
    assert data.n_cells == ref  # identical cell partition to cell_load


def test_fewshot_holdout_correct(toml_and_files):
    data = load_state_toml_adatas(toml_and_files, embed_key="X_hvg", seed=42)
    val_perts = set(data.val.obs.loc[~data.val.obs.is_control, "gene"])
    test_perts = set(data.test.obs.loc[~data.test.obs.is_control, "gene"])
    assert val_perts == {"g1"}
    assert test_perts == {"g2"}


def test_controls_shared_and_embed_present(toml_and_files):
    data = load_state_toml_adatas(toml_and_files, embed_key="X_hvg", seed=42)
    assert "X_hvg" in data.train.obsm
    assert "is_control" in data.train.obs
    val_ctrl = set(data.val.obs_names[data.val.obs.is_control])
    train_ctrl = set(data.train.obs_names[data.train.obs.is_control])
    assert len(val_ctrl & train_ctrl) > 0  # controls shared across splits
