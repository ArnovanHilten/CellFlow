"""CellFlow training script for the Replogle K562 CRISPR perturbation dataset.

Uses a STATE toml file to produce an identical train/val/test split to scDFM and STATE,
enabling a fair three-way benchmark.

Usage
-----
python scripts/run_replogle.py \
    --data_path /path/to/data \
    --data_name emb_Replogle \
    --split_toml /path/to/split.toml \
    --result_path /path/to/results \
    --wandb_project my_project \
    --wandb_entity my_entity
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import tomllib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CellFlow on a CRISPR perturbation dataset")

    # Data
    p.add_argument("--data_path",    required=True,  help="Directory containing <data_name>.h5ad")
    p.add_argument("--data_name",    required=True,  help="Dataset name (= h5ad filename without .h5ad)")
    p.add_argument("--condition_col", default="gene", help="obs column with perturbation labels")
    p.add_argument("--control_value", default="non-targeting", help="Value in condition_col that means unperturbed")
    p.add_argument("--preprocessed", action="store_true", help="Data is already log1p-normalised; skip normalisation")
    p.add_argument("--split_toml",   default="",     help="Path to STATE toml; uses val/test gene lists for splitting")
    p.add_argument("--result_path",  required=True,  help="Output directory")
    p.add_argument("--run_id",       default="",     help="Optional human-readable run label")

    # Preprocessing
    p.add_argument("--input_rep", default="counts",
                   help="Input representation: 'counts' uses adata.X (log1p HVG → PCA); "
                        "any other value is treated as an adata.obsm key (e.g. 'X_state') "
                        "and fed directly into PCA without HVG filtering.")
    p.add_argument("--n_top_genes",      type=int, default=5000, help="Number of highly variable genes (counts mode only)")
    p.add_argument("--n_pca_components", type=int, default=50,   help="Number of PCA components")

    # Training
    p.add_argument("--num_iterations",        type=int,   default=200_000)
    p.add_argument("--batch_size",            type=int,   default=1024)
    p.add_argument("--valid_freq",            type=int,   default=5000)
    p.add_argument("--lr",                    type=float, default=1e-4)

    # Architecture
    p.add_argument("--condition_embedding_dim", type=int, default=256)
    p.add_argument("--hidden_dim",              type=int, default=512, help="Hidden layer width (same for all layers)")
    p.add_argument("--n_hidden_layers",         type=int, default=3)

    # W&B
    p.add_argument("--wandb_project", default="", help="W&B project name; leave empty to disable W&B")
    p.add_argument("--wandb_entity",  default="", help="W&B entity (team or username)")
    p.add_argument("--wandb_tags",    default="", help="Comma-separated tags")

    # Resume / eval-only
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; load saved CellFlow.pkl from result_path and run test eval only")

    # Decoder (embedding modes only)
    p.add_argument("--state_checkpoint", default="",
                   help="Path to a trained STATE model checkpoint (.ckpt).  When --input_rep is an "
                        "embedding key (e.g. X_state), the STATE LatentToGeneDecoder extracted from "
                        "this checkpoint is used to map predicted embeddings → gene expression.  "
                        "If omitted, falls back to a Ridge regression fitted on training cells.")

    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_save_path(args):
    """Deterministic output folder name matching scDFM's make_path() convention."""
    key = {
        "data_name":   args.data_name,
        "split_toml":  os.path.basename(args.split_toml),
        "lr":          args.lr,
        "hidden_dim":  args.hidden_dim,
        "n_hidden_layers": args.n_hidden_layers,
        "n_top_genes": args.n_top_genes,
        "n_pca_components": args.n_pca_components,
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    name = f"{args.run_id}_{args.data_name}" if args.run_id else f"{args.data_name}_{h}"
    return os.path.join(args.result_path, name)


def parse_toml_split(split_toml):
    """Read val/test gene lists from a STATE fewshot toml file.

    Returns (val_genes: set, test_genes: set).
    """
    with open(split_toml, "rb") as f:
        toml_data = tomllib.load(f)
    val_genes, test_genes = set(), set()
    for entry in toml_data.get("fewshot", {}).values():
        if isinstance(entry, dict):
            if "val"  in entry:
                val_genes.update(entry["val"])
            if "test" in entry:
                test_genes.update(entry["test"])
    return val_genes, test_genes


# ── Inline STATE decoder (no STATE package required) ─────────────────────────
# LatentToGeneDecoder is copied verbatim from
#   github.com/ArcInstitute/state  src/state/tx/models/base.py
# so that we can load STATE checkpoint weights without installing the full
# `arc-state` package (which would create dependency conflicts with CellFlow).

import torch
import torch.nn as nn
from typing import List as _List


class LatentToGeneDecoder(nn.Module):
    """MLP that maps latent embeddings (output_dim of STATE) to gene expression.

    Copied from ArcInstitute/state src/state/tx/models/base.py.
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: _List[int] = (512, 1024),
        dropout: float = 0.1,
        residual_decoder: bool = False,
    ):
        super().__init__()
        self.residual_decoder = residual_decoder

        if residual_decoder:
            self.blocks = nn.ModuleList()
            in_dim = latent_dim
            for h in hidden_dims:
                self.blocks.append(nn.Sequential(
                    nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)
                ))
                in_dim = h
            self.final_layer = nn.Sequential(nn.Linear(in_dim, gene_dim), nn.ReLU())
        else:
            layers, in_dim = [], latent_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                in_dim = h
            layers += [nn.Linear(in_dim, gene_dim), nn.ReLU()]
            self.decoder = nn.Sequential(*layers)

    def n_output_genes(self):
        if self.residual_decoder:
            return self.final_layer[0].out_features
        for m in reversed(self.decoder):
            if isinstance(m, nn.Linear):
                return m.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_decoder:
            outs, cur = [], x
            for i, block in enumerate(self.blocks):
                out = block(cur)
                if i >= 1 and i % 2 == 1:
                    out = out + outs[i - 1]
                outs.append(out)
                cur = out
            return self.final_layer(cur)
        return self.decoder(x)


def load_state_decoder(checkpoint_path: str):
    """Load only the LatentToGeneDecoder from a STATE .ckpt checkpoint.

    Does NOT require the `arc-state` package — uses plain torch.load.

    Returns
    -------
    decoder : LatentToGeneDecoder (eval mode, CPU)
    gene_names : list[str] | None
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    decoder_cfg = hparams.get("decoder_cfg")
    if not decoder_cfg:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no 'decoder_cfg' in hyper_parameters. "
            "The STATE model was likely trained with gene_decoder_bool=False or "
            "output_space='embedding'. Re-train with output_space='gene'."
        )

    decoder = LatentToGeneDecoder(**decoder_cfg)

    # Extract just the gene_decoder.* keys from the full state_dict
    prefix = "gene_decoder."
    decoder_sd = {
        k[len(prefix):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(prefix)
    }
    if not decoder_sd:
        raise RuntimeError(
            f"No 'gene_decoder.*' keys found in checkpoint state_dict. "
            "Keys present: " + str([k for k in ckpt["state_dict"] if "decoder" in k])
        )
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    gene_names = hparams.get("gene_names")  # list[str] | None
    return decoder, gene_names


def reconstruct_from_pca(X_pca, adata, state_decoder=None, state_decoder_gene_idx=None):
    """Map from PCA space back to gene expression space.

    Three modes depending on how training was configured:

    counts mode  — PCA was fit on HVG expression.
                   Inverse: X_pca @ components + mean, clipped ≥ 0.

    STATE decoder — PCA was fit on X_state embeddings.
                   Step 1: PCA inverse → X_state  (output_dim of STATE = input_dim = 2048)
                   Step 2: STATE LatentToGeneDecoder(X_state) → gene expression
                   `state_decoder` is the nn.Module; `state_decoder_gene_idx` is an
                   optional integer index array to reorder decoder output to match adata.var order.

    Ridge fallback — PCA was fit on X_state (no STATE checkpoint given).
                   Linear map stored in adata.uns["_decoder_coef"]: X_pca → gene expression.
    """
    if adata.uns.get("_pca_input_rep", "counts") == "counts":
        # ── counts mode: PCA inverse → HVG counts ────────────────────────────
        components = adata.uns["_pca_components"]   # (n_pcs, n_genes_hvg)
        mean       = adata.uns["_pca_mean"]         # (n_genes_hvg,)
        X_recon    = np.asarray(X_pca, dtype=np.float32) @ components + mean
        return np.clip(X_recon, 0, None)

    elif state_decoder is not None:
        # ── STATE decoder: PCA inverse → X_state → gene expression ───────────
        components = adata.uns["_pca_components"]   # (n_pcs, embedding_dim)
        mean       = adata.uns["_pca_mean"]         # (embedding_dim,)
        X_emb      = np.asarray(X_pca, dtype=np.float32) @ components + mean  # (n, embedding_dim)
        with torch.no_grad():
            gene_pred = state_decoder(torch.from_numpy(X_emb)).cpu().numpy()   # (n, decoder_gene_dim)
        # Reorder to match adata.var order if needed
        if state_decoder_gene_idx is not None:
            gene_pred = gene_pred[:, state_decoder_gene_idx]
        return gene_pred

    else:
        # ── Ridge fallback: X_pca → gene expression (linear, stored in uns) ──
        coef      = adata.uns["_decoder_coef"]        # (n_genes, n_pcs)
        intercept = adata.uns["_decoder_intercept"]   # (n_genes,)
        return np.asarray(X_pca, dtype=np.float32) @ coef.T + intercept


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    save_path = make_save_path(args)
    os.makedirs(save_path, exist_ok=True)

    # Save config
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    h5ad_path = os.path.join(args.data_path, args.data_name + ".h5ad")
    print(f"Loading {h5ad_path} …")
    adata = sc.read_h5ad(h5ad_path)

    # ── 1b. Cell-line filter — toml split is cell-line-specific ──────────────
    # The STATE toml uses keys like "replogle.k562", meaning the held-out gene
    # lists only apply to K562 cells.  If the h5ad contains multiple cell lines
    # (e.g. jurkat, rpe1, hepg2 in addition to k562), we must subset to the
    # target cell line BEFORE applying the toml split; otherwise non-k562 cells
    # with k562-test genes are wrongly excluded from training.
    if args.split_toml and "cell_line" in adata.obs.columns:
        # Infer the target cell line from the toml key (e.g. "replogle.k562" → "k562")
        import tomllib as _tomllib
        with open(args.split_toml, "rb") as _f:
            _toml = _tomllib.load(_f)
        _fewshot_keys = list(_toml.get("fewshot", {}).keys())  # e.g. ["replogle.k562"]
        _cell_lines_in_toml = {k.split(".")[-1] for k in _fewshot_keys}   # {"k562"}
        _avail = set(adata.obs["cell_line"].unique())
        _matched = _cell_lines_in_toml & _avail
        if _matched:
            _target = sorted(_matched)[0]   # use the first match (typically only one)
            n_before = adata.n_obs
            adata = adata[adata.obs["cell_line"] == _target].copy()
            print(f"  Cell-line filter: kept '{_target}' cells "
                  f"({adata.n_obs:,} / {n_before:,})")
        else:
            print(f"  Warning: toml cell lines {_cell_lines_in_toml} not found in "
                  f"adata.obs['cell_line'] ({_avail}); skipping cell-line filter")

    # ── 2. Normalise if needed (counts mode only) ─────────────────────────────
    if args.input_rep == "counts" and not args.preprocessed:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    # ── 3. Add metadata columns ───────────────────────────────────────────────
    adata.obs["condition"]  = adata.obs[args.condition_col].astype(str)
    adata.obs["is_control"] = (adata.obs["condition"] == args.control_value)

    # ── 4. HVG selection (counts mode only) ──────────────────────────────────
    if args.input_rep == "counts":
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes)
        perturb_genes = set(adata.obs["condition"].unique()) - {args.control_value}
        for g in perturb_genes:
            if g in adata.var_names:
                adata.var.loc[g, "highly_variable"] = True
            else:
                print(f"  Warning: perturbation gene '{g}' not in var_names")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"  HVG matrix: {adata.shape}")

    # ── 5. Parse toml split (BEFORE PCA — determines training cells) ──────────
    toml_stem   = Path(args.split_toml).stem if args.split_toml else "random"
    split_cache = os.path.join(args.data_path, args.data_name, f"split_results_cellflow_{toml_stem}.pkl")
    os.makedirs(os.path.dirname(split_cache), exist_ok=True)

    if args.split_toml and os.path.exists(split_cache):
        with open(split_cache, "rb") as f:
            sr = pickle.load(f)
        val_conditions   = sr["val"]
        test_conditions  = sr["test"]
        train_conditions = sr["train"]
    elif args.split_toml:
        val_genes, test_genes = parse_toml_split(args.split_toml)
        all_conds    = set(adata.obs["condition"].unique())
        non_control  = [c for c in all_conds if c != args.control_value]
        val_conditions  = [c for c in non_control if c in val_genes]
        test_set        = set(c for c in non_control if c in test_genes)
        val_set         = set(val_conditions)
        # genes in both → keep in test only
        val_conditions  = [c for c in val_conditions if c not in test_set]
        test_conditions = [c for c in non_control if c in test_set]
        val_set         = set(val_conditions)
        held_out        = val_set | test_set
        train_conditions = [c for c in non_control if c not in held_out]
        print(f"STATE toml split: {len(train_conditions)} train / "
              f"{len(val_conditions)} val / {len(test_conditions)} test perturbations")
        sr = {"train": train_conditions, "val": val_conditions, "test": test_conditions}
        with open(split_cache, "wb") as f:
            pickle.dump(sr, f)
    else:
        # No toml: random 70/30 split for sanity testing
        non_control = [c for c in adata.obs["condition"].unique() if c != args.control_value]
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(non_control)
        split_idx = int(len(shuffled) * 0.3)
        test_conditions  = shuffled[:split_idx].tolist()
        val_conditions   = []
        train_conditions = shuffled[split_idx:].tolist()
        print(f"Random split: {len(train_conditions)} train / {len(test_conditions)} test")

    # ── 6. PCA fitted on TRAINING CELLS ONLY ─────────────────────────────────
    # Training cells = control cells + cells with a training perturbation.
    # Val/test perturbation cells must not influence the PCA axes.
    from sklearn.decomposition import PCA as SklearnPCA

    train_cell_mask = (
        adata.obs["is_control"] |
        adata.obs["condition"].isin(train_conditions)
    ).values

    if args.input_rep == "counts":
        X_all = adata.X
        if hasattr(X_all, "toarray"):
            X_all = X_all.toarray()
        X_all = np.asarray(X_all, dtype=np.float32)
    else:
        if args.input_rep not in adata.obsm:
            raise ValueError(f"--input_rep '{args.input_rep}' not found in adata.obsm. "
                             f"Available keys: {list(adata.obsm.keys())}")
        X_all = np.asarray(adata.obsm[args.input_rep], dtype=np.float32)
        print(f"  Using obsm['{args.input_rep}'] as input: shape {X_all.shape}")

    # ── NaN/Inf audit & fix ──────────────────────────────────────────────────
    # Apply unconditionally: NaN or Inf in the input representation propagate
    # through PCA → gene_emb → condition encoder → loss as NaN.
    bad_mask = ~np.isfinite(X_all)
    n_bad_cells = bad_mask.any(axis=1).sum()
    if n_bad_cells > 0:
        print(f"  WARNING: {n_bad_cells} cells have NaN/Inf in '{args.input_rep}' "
              f"({bad_mask.sum()} total bad values) — replacing with 0.")
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    if args.input_rep != "counts":
        adata.obsm[args.input_rep] = X_all
    # counts mode: X_all was already densified; adata.X not updated
    # (PCA operates on X_all directly, adata.X not needed after this)

    X_train_cells = X_all[train_cell_mask]
    pca = SklearnPCA(n_components=args.n_pca_components, random_state=0)
    pca.fit(X_train_cells)
    print(f"  PCA fitted on {train_cell_mask.sum()} training cells "
          f"({args.n_pca_components} components, "
          f"var explained: {pca.explained_variance_ratio_.sum():.1%})")

    X_pca = pca.transform(X_all).astype(np.float32)
    bad_in_pca = (~np.isfinite(X_pca)).sum()
    if bad_in_pca > 0:
        print(f"  WARNING: {bad_in_pca} NaN/Inf values in X_pca after PCA — replacing with 0.")
    X_pca = np.nan_to_num(X_pca, nan=0.0, posinf=0.0, neginf=0.0)
    adata.obsm["X_pca"] = X_pca

    # Store PCA params for inverse-transform (used by reconstruct_from_pca)
    adata.uns["_pca_components"] = pca.components_.astype(np.float32)  # (n_pcs, n_features)
    adata.uns["_pca_mean"]       = pca.mean_.astype(np.float32)        # (n_features,)
    adata.uns["_pca_input_rep"]  = args.input_rep

    # ── 6b. Linear decoder: X_pca → gene expression (embedding modes only) ───
    # For counts mode, gene expression is recovered by PCA inverse.
    # For embedding modes (X_state, …), the PCA space is over ~2048-dim
    # embeddings — not gene expression.  We prefer the trained STATE
    # LatentToGeneDecoder (an MLP already trained on the same corpus).  If no
    # STATE checkpoint is supplied, we fall back to a Ridge regression fitted on
    # training cells as a linear approximation.
    state_decoder          = None   # nn.Module or None
    state_decoder_gene_idx = None   # index array to reorder decoder output → adata.var order

    if args.input_rep != "counts":
        if args.state_checkpoint:
            # ── STATE decoder (preferred) ─────────────────────────────────────
            # Loaded with plain torch.load — no `arc-state` package needed.
            try:
                print(f"  Loading STATE decoder from: {args.state_checkpoint} …")
                state_decoder, decoder_gene_names = load_state_decoder(args.state_checkpoint)

                # Align decoder output genes to adata.var order.
                # The decoder predicts a fixed gene set (gene_dim=2000); adata may have
                # far more genes.  We take the intersection and subset adata so that
                # adata.X (ground truth) and decoder output are on the same gene set.
                if decoder_gene_names is not None:
                    decoder_gene_set  = set(decoder_gene_names)
                    adata_gene_set    = set(adata.var_names)
                    n_missing_in_adata = len(decoder_gene_set - adata_gene_set)
                    if n_missing_in_adata:
                        print(f"  {n_missing_in_adata} decoder genes absent from adata.var (will be ignored)")

                    # Keep intersection, in adata.var order for consistent slicing
                    genes_in_adata_order = [g for g in adata.var_names if g in decoder_gene_set]
                    print(f"  Evaluating on {len(genes_in_adata_order)} overlapping genes "
                          f"(decoder has {len(decoder_gene_names)}, adata had {adata.n_vars})")

                    # For each overlapping gene (in adata.var order), its column in decoder output
                    gene_to_decoder_idx = {g: i for i, g in enumerate(decoder_gene_names)}
                    state_decoder_gene_idx = np.array(
                        [gene_to_decoder_idx[g] for g in genes_in_adata_order], dtype=np.int64
                    )

                    # Subset adata to decoder genes so adata.X aligns with predictions.
                    # Steps 7 & 8 run after this block, so all slices inherit the subset.
                    adata = adata[:, genes_in_adata_order].copy()

                n_genes = state_decoder.n_output_genes()
                print(f"  STATE LatentToGeneDecoder ready: → {n_genes} genes")

            except Exception as e:
                print(f"  Warning: STATE decoder loading failed ({e}). "
                      f"Falling back to Ridge regression.")
                state_decoder = None

        if state_decoder is None:
            # ── Ridge fallback ────────────────────────────────────────────────
            from sklearn.linear_model import Ridge

            X_pca_tr  = adata.obsm["X_pca"][train_cell_mask]   # (n_tr, n_pcs)
            Y_gene_tr = adata.X[train_cell_mask]
            if hasattr(Y_gene_tr, "toarray"):
                Y_gene_tr = Y_gene_tr.toarray()
            Y_gene_tr = np.asarray(Y_gene_tr, dtype=np.float32)

            print(f"  Fitting Ridge decoder: "
                  f"X_pca({args.n_pca_components}) → genes({Y_gene_tr.shape[1]}) "
                  f"on {train_cell_mask.sum():,} training cells …")
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_pca_tr, Y_gene_tr)

            adata.uns["_decoder_coef"]      = ridge.coef_.astype(np.float32)
            adata.uns["_decoder_intercept"] = ridge.intercept_.astype(np.float32)

            Y_hat  = X_pca_tr @ ridge.coef_.T + ridge.intercept_
            ss_res = float(((Y_gene_tr - Y_hat) ** 2).sum())
            ss_tot = float(((Y_gene_tr - Y_gene_tr.mean(0)) ** 2).sum())
            print(f"  Ridge in-sample R²: {1 - ss_res / ss_tot:.4f}")

    # ── 7. Pre-compute perturbation embeddings for ALL genes ──────────────────
    # Gene embedding = mean PCA profile of that gene's TRAINING knockdown cells.
    # Val/test genes are also included in the dict so CellFlow can look them up,
    # but their embeddings are derived from held-out cells' PCA projections —
    # this is acceptable because PCA was fit on training cells, so the
    # val/test PCA values are out-of-sample projections, not training signals.
    all_perturb_genes = list(set(train_conditions) | set(val_conditions) | set(test_conditions))
    X_pca_all = adata.obsm["X_pca"]
    cond_vals  = adata.obs["condition"].values
    gene_emb   = {}
    for g in all_perturb_genes:
        mask = cond_vals == g
        if mask.any():
            gene_emb[g] = X_pca_all[mask].mean(axis=0).astype(np.float32)
        else:
            gene_emb[g] = np.zeros(args.n_pca_components, dtype=np.float32)
    # Store in adata.uns before slicing so all subsets inherit it
    adata.uns["gene_emb"] = gene_emb
    print(f"  Gene embeddings: {len(gene_emb)} genes × {args.n_pca_components} dims")

    # ── 8. Slice adatas ───────────────────────────────────────────────────────
    ctrl_mask = adata.obs["is_control"]
    adata_train = adata[ctrl_mask | adata.obs["condition"].isin(train_conditions)].copy()
    adata_val   = adata[ctrl_mask | adata.obs["condition"].isin(val_conditions)].copy()
    adata_test  = adata[ctrl_mask | adata.obs["condition"].isin(test_conditions)].copy()
    print(f"  adata_train: {adata_train.shape[0]} cells  "
          f"adata_val: {adata_val.shape[0]} cells  adata_test: {adata_test.shape[0]} cells")

    # ── 9. CellFlow training ──────────────────────────────────────────────────
    import optax
    import cellflow
    from cellflow.model import CellFlow
    from cellflow.training import Metrics, WandbLogger

    cf = CellFlow(adata_train, solver="otfm")

    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"gene": ["condition"]},
        perturbation_covariate_reps={"gene": "gene_emb"},  # dict lookup → works for val/test genes
        max_combination_length=1,
    )

    if val_conditions:
        cf.prepare_validation_data(
            adata_val,
            name="val",
            n_conditions_on_log_iteration=min(50, len(val_conditions)),
            n_conditions_on_train_end=None,
        )

    dims = tuple([args.hidden_dim] * args.n_hidden_layers)
    cf.prepare_model(
        condition_embedding_dim=args.condition_embedding_dim,
        hidden_dims=dims,
        decoder_dims=dims,
        time_encoder_dims=dims,
        pooling="attention_token",
        optimizer=optax.adam(args.lr),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    from cellflow.training._callbacks import BaseCallback

    class NaNStopCallback(BaseCallback):
        """Stops training when loss becomes NaN or Inf.

        Checked every valid_freq iterations (on_log_iteration).
        Raises RuntimeError so the job fails fast with a clear message.
        """

        def on_train_begin(self):
            pass

        def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver):
            # training_logs lives on cf.trainer, not the solver
            trainer = getattr(cf, "trainer", None)
            if trainer is None:
                return {}
            loss_history = trainer.training_logs.get("loss", [])
            if not loss_history:
                return {}
            recent = loss_history[-args.valid_freq:]
            nan_count = sum(1 for v in recent if v != v or abs(v) == float("inf"))
            if nan_count > 0:
                msg = (
                    f"[NaNStopCallback] {nan_count}/{len(recent)} loss values are NaN/Inf "
                    f"in the last {args.valid_freq} iterations. "
                    f"Last 5 losses: {recent[-5:]}. Stopping training."
                )
                print(f"\n[FATAL] {msg}")
                raise RuntimeError(msg)
            return {}

        def on_train_end(self, *args, **kwargs):
            return {}

    callbacks = []
    callbacks.append(NaNStopCallback())

    # Metrics computed in PCA space (fast, used for monitoring)
    callbacks.append(Metrics(metrics=["r_squared", "e_distance", "mmd"]))

    wandb_cb = None
    if args.wandb_project:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run_name = args.run_id or os.path.basename(save_path)
        wandb_cb = WandbLogger(
            project=args.wandb_project,
            out_dir=save_path,
            config=vars(args),
            entity=args.wandb_entity or None,
            name=run_name,
            tags=tags,
        )
        callbacks.append(wandb_cb)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.eval_only:
        print(f"\nSkipping training — loading saved model from {save_path} …")
        cf = CellFlow.load(save_path)
    else:
        print(f"\nTraining for {args.num_iterations} iterations (valid_freq={args.valid_freq}) …")
        try:
            cf.train(
                num_iterations=args.num_iterations,
                batch_size=args.batch_size,
                valid_freq=args.valid_freq,
                callbacks=callbacks,
                monitor_metrics=["val_r_squared_mean"] if val_conditions else [],
            )
        except RuntimeError as e:
            if "NaNStopCallback" in str(e):
                print(f"\n[FATAL] Training stopped due to NaN loss: {e}")
                sys.exit(1)
            raise
        cf.save(save_path, overwrite=True)
        print(f"Model saved to {save_path}/CellFlow.pkl")

    # ── 10. Final evaluation on held-out TEST set ─────────────────────────────
    print("\n=== Running final evaluation on held-out TEST set ===")
    final_path = os.path.join(save_path, "final_test")
    os.makedirs(final_path, exist_ok=True)

    # Predict for each test perturbation starting from control cells
    control_cells = adata_test[adata_test.obs["is_control"]].copy()
    covariate_df  = pd.DataFrame({
        "condition":     test_conditions,
        "is_control":    False,
        # condition_name is a separate ID column — must differ from the perturbation
        # covariate key ("condition") so that _get_perturb_covar_df does not consume
        # it as the index before CellFlow can use it as a condition identifier.
        # With condition_id_key="condition_name", predictions is keyed by the gene
        # name strings (e.g. "BRCA1") rather than by tuples (e.g. ("BRCA1",)).
        "condition_name": test_conditions,
    })

    predictions = cf.predict(
        adata=control_cells,
        covariate_data=covariate_df,
        sample_rep="X_pca",
        condition_id_key="condition_name",
    )
    # predictions: dict {gene_name_str → np.ndarray (n_cells, n_pca)}

    # Reconstruct from PCA → gene expression space
    all_pred_expr, obs_pred_names = [], []
    all_real_expr, obs_real_names = [], []

    # Shorthand so we don't repeat kwargs at every call site
    def _decode(X_pca):
        return reconstruct_from_pca(
            X_pca, adata,
            state_decoder=state_decoder,
            state_decoder_gene_idx=state_decoder_gene_idx,
        )

    # Control baseline
    # pred: decoded from the PCA coordinates that CellFlow uses as source
    # real: actual gene expression from adata.X (ground truth)
    ctrl_pred_expr = _decode(control_cells.obsm["X_pca"])
    ctrl_real_expr = np.asarray(
        control_cells.X.toarray() if hasattr(control_cells.X, "toarray")
        else control_cells.X
    )
    all_pred_expr.append(ctrl_pred_expr)
    all_real_expr.append(ctrl_real_expr)
    obs_pred_names.extend(["control"] * ctrl_pred_expr.shape[0])
    obs_real_names.extend(["control"] * ctrl_real_expr.shape[0])

    skipped = []
    for cond in test_conditions:
        if cond not in predictions:
            skipped.append(cond)
            continue
        pred_pca  = predictions[cond]      # (n_cells, n_pca)
        pred_gene = _decode(pred_pca)      # (n_cells, n_genes)

        # Ground-truth perturbed cells
        real_cells = adata_test[adata_test.obs["condition"] == cond]
        real_gene  = np.asarray(real_cells.X.todense()
                                if hasattr(real_cells.X, "todense") else real_cells.X)

        all_pred_expr.append(pred_gene)
        all_real_expr.append(real_gene)
        obs_pred_names.extend([cond] * pred_gene.shape[0])
        obs_real_names.extend([cond] * real_gene.shape[0])

    all_pred_expr = np.concatenate(all_pred_expr, axis=0)
    all_real_expr = np.concatenate(all_real_expr, axis=0)

    pred_adata = ad.AnnData(X=all_pred_expr,
                            obs=pd.DataFrame({"perturbation": obs_pred_names}))
    real_adata = ad.AnnData(X=all_real_expr,
                            obs=pd.DataFrame({"perturbation": obs_real_names}))
    pred_adata.write_h5ad(os.path.join(final_path, "pred.h5ad"))
    real_adata.write_h5ad(os.path.join(final_path, "real.h5ad"))

    # cell-eval metrics — run in a subprocess so pdex can fork freely.
    # JAX is multithreaded; os.fork() after JAX starts causes a deadlock in
    # pdex's worker pool.  A fresh subprocess has no JAX threads → no deadlock.
    try:
        import subprocess, sys, json as _json

        pred_path    = os.path.join(final_path, "pred.h5ad")
        real_path    = os.path.join(final_path, "real.h5ad")
        results_csv  = os.path.join(final_path, "results.csv")
        agg_csv      = os.path.join(final_path, "agg_results.csv")

        eval_script = f"""
import anndata as ad, json, pandas as pd
from cell_eval import MetricsEvaluator

pred = ad.read_h5ad({_json.dumps(pred_path)})
real = ad.read_h5ad({_json.dumps(real_path)})
evaluator = MetricsEvaluator(
    adata_pred=pred, adata_real=real,
    control_pert="control", pert_col="perturbation", num_threads=8,
)
results, agg = evaluator.compute()
results.write_csv({_json.dumps(results_csv)})
agg.write_csv({_json.dumps(agg_csv)})
agg_df = agg.to_pandas()
mean_row = agg_df[agg_df["statistic"] == "mean"].iloc[0].to_dict()
print(json.dumps({{k: v for k, v in mean_row.items() if isinstance(v, float)}}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", eval_script],
            capture_output=False,   # stream stdout/stderr directly to the log
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"cell-eval subprocess exited with code {proc.returncode}")

        # Read back mean metrics from the written CSV
        import pandas as _pd
        agg_df   = _pd.read_csv(agg_csv)
        mean_row = agg_df[agg_df["statistic"] == "mean"].iloc[0].to_dict()
        print("Test set metrics (mean across perturbations):")
        for k, v in mean_row.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # Log to W&B
        if wandb_cb is not None:
            import wandb
            if wandb.run is not None:
                wandb.log({f"test/{k}": v for k, v in mean_row.items()
                           if isinstance(v, (int, float))})
    except Exception as e:
        print(f"Warning: cell-eval failed ({e}). Skipping final metrics computation.")

    if skipped:
        print(f"\n{len(skipped)} test perturbations were not returned by predict() and were skipped:")
        for s in sorted(skipped):
            print(f"  {s}")

    print(f"\nDone. Results written to: {save_path}")


if __name__ == "__main__":
    main()
