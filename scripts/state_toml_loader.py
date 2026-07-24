"""Build CellFlow-ready train/val/test AnnDatas from a STATE-style TOML.

This reuses STATE's ``cell_load`` package to compute the *exact* same
train/val/test partition STATE uses (multi-dataset, with zeroshot cell types and
fewshot perturbation hold-outs), then materialises the partition as in-memory
:class:`~anndata.AnnData` objects that CellFlow's ``prepare_data`` consumes.

Why go through ``cell_load`` instead of re-reading the TOML ourselves: it
guarantees byte-for-byte identical splits to a STATE run on the same TOML+seed,
so CellFlow is a fair benchmark against STATE.

The module depends only on ``cell_load`` + ``anndata`` (no ``cellflow`` import),
so it is independently testable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np

__all__ = ["StateSplitData", "load_state_toml_adatas"]


@dataclass
class StateSplitData:
    """The three split AnnDatas plus provenance for logging/eval."""

    train: ad.AnnData
    val: ad.AnnData | None
    test: ad.AnnData | None
    embed_key: str | None
    pert_col: str
    cell_type_key: str
    control_pert: str
    n_cells: dict[str, int] = field(default_factory=dict)


def _row_indices_by_split_and_file(dm) -> dict[str, dict[str, set[int]]]:
    """Union the row indices each ``cell_load`` subset covers, per split and h5 file.

    ``subset.indices`` (a torch ``Subset``) already contains perturbed **and**
    control rows (``should_yield_control_cells=True``); ``subset.dataset.h5_path``
    identifies the file. Controls are intentionally shared across splits, so the
    same physical row may land in more than one split — that is correct (each
    split needs its own control/source population for optimal transport).
    """
    split_lists = {
        "train": getattr(dm, "train_datasets", []),
        "val": getattr(dm, "val_datasets", []),
        "test": getattr(dm, "test_datasets", []),
    }
    out: dict[str, dict[str, set[int]]] = {s: defaultdict(set) for s in split_lists}
    for split, subsets in split_lists.items():
        for sub in subsets:
            h5_path = str(sub.dataset.h5_path)
            out[split][h5_path].update(int(i) for i in np.asarray(sub.indices))
    return out


def _slice_file(
    h5_path: str,
    row_idx: set[int],
    split: str,
    pert_col: str,
    cell_type_key: str,
    control_pert: str,
    embed_key: str | None,
    dataset_name: str,
    _cache: dict[str, ad.AnnData],
) -> ad.AnnData:
    """Read (cached) an h5ad and return the sliced rows with CellFlow annotations."""
    if h5_path not in _cache:
        _cache[h5_path] = ad.read_h5ad(h5_path)
    adata = _cache[h5_path]

    rows = np.sort(np.fromiter(row_idx, dtype=np.int64))
    sub = adata[rows].copy()
    # stable, globally-unique cell ids: identical across splits for a shared cell,
    # and collision-free when datasets are concatenated.
    sub.obs_names = [f"{dataset_name}:{int(r)}" for r in rows]

    for col in (pert_col, cell_type_key):
        if col not in sub.obs.columns:
            raise KeyError(f"Column '{col}' not found in obs of {h5_path}. Present: {list(sub.obs.columns)}")
    if embed_key is not None and embed_key not in sub.obsm:
        raise KeyError(f"embed_key '{embed_key}' not found in obsm of {h5_path}. Present: {list(sub.obsm)}")

    sub.obs["is_control"] = (sub.obs[pert_col].astype(str) == control_pert).values
    sub.obs["split"] = split
    sub.obs["dataset"] = dataset_name
    return sub


def load_state_toml_adatas(
    toml_config_path: str,
    embed_key: Literal["X_hvg", "X_state"] | None = "X_hvg",
    *,
    pert_col: str = "gene",
    cell_type_key: str = "cell_type",
    control_pert: str = "non-targeting",
    batch_col: str = "gem_group",
    seed: int = 42,
) -> StateSplitData:
    """Materialise STATE's train/val/test split as CellFlow-ready AnnDatas.

    Parameters
    ----------
    toml_config_path
        STATE-style TOML with ``[datasets]``/``[training]``/``[zeroshot]``/``[fewshot]``.
    embed_key
        obsm key used as the cell representation downstream (``X_hvg`` or
        ``X_state``); :obj:`None` uses ``.X``. Validated against each file.
    pert_col, cell_type_key, control_pert, batch_col
        obs schema, matching the STATE run.
    seed
        Random seed forwarded to ``cell_load`` so control shuffling is identical.

    Returns
    -------
    StateSplitData
        ``.train`` / ``.val`` / ``.test`` AnnDatas (val/test may be :obj:`None` if
        the TOML defines no such split), each with ``obs['is_control']``,
        ``obs['split']``, ``obs['dataset']`` and the chosen ``obsm[embed_key]``.
    """
    from cell_load.data_modules import PerturbationDataModule

    dm = PerturbationDataModule(
        toml_config_path=str(toml_config_path),
        embed_key=embed_key,
        pert_col=pert_col,
        cell_type_key=cell_type_key,
        control_pert=control_pert,
        batch_col=batch_col,
        random_seed=seed,
        should_yield_control_cells=True,  # keep controls in subset.indices
    )
    dm.setup()

    # dataset name per h5 file (for provenance / logging)
    name_by_file: dict[str, str] = {}
    for subsets in (dm.train_datasets, dm.val_datasets, dm.test_datasets):
        for sub in subsets:
            name_by_file[str(sub.dataset.h5_path)] = sub.dataset.name

    idx = _row_indices_by_split_and_file(dm)

    read_cache: dict[str, ad.AnnData] = {}
    split_adatas: dict[str, ad.AnnData | None] = {"train": None, "val": None, "test": None}
    n_cells: dict[str, int] = {}

    for split, per_file in idx.items():
        parts = []
        for h5_path, rows in sorted(per_file.items()):
            if not rows:
                continue
            parts.append(
                _slice_file(
                    h5_path, rows, split, pert_col, cell_type_key, control_pert,
                    embed_key, name_by_file.get(h5_path, Path(h5_path).stem), read_cache,
                )
            )
        if not parts:
            n_cells[split] = 0
            continue
        merged = parts[0] if len(parts) == 1 else ad.concat(parts, join="inner", merge="same")
        split_adatas[split] = merged
        n_cells[split] = merged.n_obs

    if split_adatas["train"] is None:
        raise ValueError(f"No training cells produced from {toml_config_path}. Check [training]/[fewshot] and obs schema.")

    return StateSplitData(
        train=split_adatas["train"],
        val=split_adatas["val"],
        test=split_adatas["test"],
        embed_key=embed_key,
        pert_col=pert_col,
        cell_type_key=cell_type_key,
        control_pert=control_pert,
        n_cells=n_cells,
    )
