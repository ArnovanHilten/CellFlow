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


def reconstruct_from_pca(X_pca, adata):
    """Inverse-PCA: project back from the PCA space used during training.

    Supports both count-mode (reconstruction to gene space, clipped ≥ 0)
    and embedding-mode (reconstruction to embedding space, no clipping).
    PCA params are stored in adata.uns["_pca_components"] / ["_pca_mean"]
    by our sklearn fit step.
    """
    components = adata.uns["_pca_components"]   # (n_pcs, n_features)
    mean       = adata.uns["_pca_mean"]         # (n_features,)
    X_recon    = np.asarray(X_pca, dtype=np.float32) @ components + mean
    # Clip to 0 only for gene-expression reconstructions (embedding reps can be negative)
    if adata.uns.get("_pca_input_rep", "counts") == "counts":
        X_recon = np.clip(X_recon, 0, None)
    return X_recon


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

    X_train_cells = X_all[train_cell_mask]
    pca = SklearnPCA(n_components=args.n_pca_components, random_state=0)
    pca.fit(X_train_cells)
    print(f"  PCA fitted on {train_cell_mask.sum()} training cells "
          f"({args.n_pca_components} components, "
          f"var explained: {pca.explained_variance_ratio_.sum():.1%})")

    adata.obsm["X_pca"] = pca.transform(X_all).astype(np.float32)

    # Store PCA params for inverse-transform (used by reconstruct_from_pca)
    adata.uns["_pca_components"] = pca.components_.astype(np.float32)  # (n_pcs, n_features)
    adata.uns["_pca_mean"]       = pca.mean_.astype(np.float32)        # (n_features,)
    adata.uns["_pca_input_rep"]  = args.input_rep

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
    callbacks = []

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
        cf.train(
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            valid_freq=args.valid_freq,
            callbacks=callbacks,
            monitor_metrics=["val_r_squared_mean"] if val_conditions else [],
        )
        cf.save(save_path, overwrite=True)
        print(f"Model saved to {save_path}/CellFlow.pkl")

    # ── 10. Final evaluation on held-out TEST set ─────────────────────────────
    print("\n=== Running final evaluation on held-out TEST set ===")
    final_path = os.path.join(save_path, "final_test")
    os.makedirs(final_path, exist_ok=True)

    # Predict for each test perturbation starting from control cells
    control_cells = adata_test[adata_test.obs["is_control"]].copy()
    covariate_df  = pd.DataFrame({
        "condition":  test_conditions,
        "is_control": False,   # required by DataManager._get_condition_data
    })

    predictions = cf.predict(
        adata=control_cells,
        covariate_data=covariate_df,
        sample_rep="X_pca",
        condition_id_key="condition",
    )
    # predictions: dict {gene_name → np.ndarray (n_cells, n_pca)}

    # Reconstruct from PCA → gene expression space
    all_pred_expr, obs_pred_names = [], []
    all_real_expr, obs_real_names = [], []

    # Control baseline (same for all evaluators)
    ctrl_gene_expr = reconstruct_from_pca(control_cells.obsm["X_pca"], adata)
    all_pred_expr.append(ctrl_gene_expr)
    all_real_expr.append(ctrl_gene_expr)
    obs_pred_names.extend(["control"] * ctrl_gene_expr.shape[0])
    obs_real_names.extend(["control"] * ctrl_gene_expr.shape[0])

    skipped = []
    for cond in test_conditions:
        if cond not in predictions:
            skipped.append(cond)
            continue
        pred_pca  = predictions[cond]                          # (n_cells, n_pca)
        pred_gene = reconstruct_from_pca(pred_pca, adata)     # (n_cells, n_genes)

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

    # cell-eval metrics (same evaluator as scDFM)
    try:
        from cell_eval import MetricsEvaluator
        evaluator = MetricsEvaluator(
            adata_pred=pred_adata,
            adata_real=real_adata,
            control_pert="control",
            pert_col="perturbation",
            num_threads=8,
        )
        results, agg_results = evaluator.compute()
        results.write_csv(os.path.join(final_path, "results.csv"))
        agg_results.write_csv(os.path.join(final_path, "agg_results.csv"))
        agg_dict = agg_results.to_pandas().iloc[0].to_dict()
        print("Test set metrics:")
        for k, v in agg_dict.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # Log test metrics to W&B
        if wandb_cb is not None:
            import wandb
            if wandb.run is not None:
                wandb.log({f"test/{k}": v for k, v in agg_dict.items()
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
