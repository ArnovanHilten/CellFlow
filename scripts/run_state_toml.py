"""CellFlow training on STATE-style multi-dataset TOMLs, with functional embeddings.

Front-end: ``state_toml_loader.load_state_toml_adatas`` reuses STATE's ``cell_load``
to build the *identical* train/val/test split (multi-dataset, zeroshot/fewshot).
Downstream mirrors ``run_replogle.py`` (PCA → functional gene embeddings →
CellFlow flow-matching → decode → eval), generalised to ``cell_type`` as the
split covariate. Gene-space decoding reuses run_replogle's helpers.

Usage
-----
python scripts/run_state_toml.py \
    --toml /path/Tian1921.toml \
    --embed_key X_hvg \
    --embeddings_dir /path/per_source --embedding_sources all \
    --embedding_gene_id_map /path/gencode.v50.gene_map.tsv \
    --result_path /path/results --wandb_project my_project
"""

import argparse
import hashlib
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse run_replogle's gene-space decoder helpers
from run_replogle import load_state_decoder, reconstruct_from_pca
from state_toml_loader import load_state_toml_adatas


def make_save_path(args):
    """Deterministic output folder from the run's identifying config."""
    key = {
        "data_name": args.data_name,
        "toml": os.path.basename(args.toml),
        "embed_key": args.embed_key,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "n_hidden_layers": args.n_hidden_layers,
        "n_pca_components": args.n_pca_components,
        "sources": args.embedding_sources,
        "fusion": args.embedding_fusion,
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    # run_id is used verbatim as the folder name (it already encodes the config);
    # only fall back to data_name + hash when no run_id is given. Avoids the previous
    # double "<run_id>_<data_name>" when run_id already contained data_name.
    name = args.run_id if args.run_id else f"{args.data_name}_{h}"
    return os.path.join(args.result_path, name)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train CellFlow on a STATE-style TOML with functional embeddings")

    # Data / split
    p.add_argument("--toml", required=True, help="STATE-style TOML ([datasets]/[training]/[zeroshot]/[fewshot])")
    p.add_argument("--embed_key", default="X_hvg", choices=["X_hvg", "X_state"],
                   help="obsm key used as the cell representation (OT space via PCA).")
    p.add_argument("--pert_col", default="gene")
    p.add_argument("--cell_type_key", default="cell_type")
    p.add_argument("--control_pert", default="non-targeting")
    p.add_argument("--batch_col", default="gem_group")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--result_path", required=True)
    p.add_argument("--run_id", default="")
    p.add_argument("--data_name", default="state_toml")

    # PCA / decode
    p.add_argument("--n_pca_components", type=int, default=50)
    p.add_argument("--state_checkpoint", default="",
                   help="STATE .ckpt for decoding X_state → genes (embedding mode); Ridge fallback otherwise.")

    # Functional gene embeddings (same surface as run_replogle.py)
    p.add_argument("--embeddings_dir", default="")
    p.add_argument("--embedding_sources", default="all")
    p.add_argument("--embedding_fusion", default="concat", choices=["concat", "multi_stream"])
    p.add_argument("--embedding_anchor", default="ESM-2")
    p.add_argument("--embedding_gene_id_map", default="")

    # Training / architecture
    p.add_argument("--num_iterations", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--valid_freq", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--condition_embedding_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_hidden_layers", type=int, default=3)

    # W&B
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_tags", default="")
    p.add_argument("--eval_num_threads", type=int, default=32)

    return p.parse_args()


def _feature_matrix(adata, embed_key):
    """Return the (n_cells, d) matrix to PCA, from obsm[embed_key] or .X."""
    if embed_key is not None:
        if embed_key not in adata.obsm:
            raise ValueError(f"embed_key '{embed_key}' not in obsm (have {list(adata.obsm)}).")
        return np.asarray(adata.obsm[embed_key], dtype=np.float32)
    return np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)


def _load_gene_id_map(path):
    if not path:
        return None
    if path.endswith(".parquet"):
        m = pd.read_parquet(path)
    else:
        m = pd.read_csv(path, sep=None, engine="python")
    m.columns = [str(c) for c in m.columns]
    cols = list(m.columns)
    if "gene_name" in cols and "gene_id_base" in cols:
        key_col, val_col = "gene_name", "gene_id_base"
    elif "gene_name" in cols and "gene_id" in cols:
        key_col, val_col = "gene_name", "gene_id"
    else:
        key_col, val_col = cols[0], cols[1]
    if "gene_type" in cols:
        m = m.sort_values("gene_type", key=lambda s: s.eq("protein_coding"))
    vals = m[val_col].astype(str).str.split(".").str[0]
    return dict(zip(m[key_col].astype(str), vals, strict=False))


def main():
    """Load the STATE-TOML split, train CellFlow with functional embeddings, and evaluate."""
    args = parse_args()
    save_path = make_save_path(args)
    os.makedirs(save_path, exist_ok=True)

    # ── 1. Build the STATE-identical split as AnnDatas ────────────────────────
    print(f"Loading split from {args.toml} (embed_key={args.embed_key}) …")
    data = load_state_toml_adatas(
        args.toml,
        embed_key=args.embed_key,
        pert_col=args.pert_col,
        cell_type_key=args.cell_type_key,
        control_pert=args.control_pert,
        batch_col=args.batch_col,
        seed=args.seed,
    )
    adata_train, adata_val, adata_test = data.train, data.val, data.test
    print(f"  splits (cells): {data.n_cells}")
    for a in (adata_train, adata_val, adata_test):
        if a is not None:
            a.obs["condition"] = a.obs[args.pert_col].astype(str)

    # ── 2. PCA on the train representation, project all splits ────────────────
    from sklearn.decomposition import PCA

    X_train = _feature_matrix(adata_train, args.embed_key)
    pca = PCA(n_components=args.n_pca_components, svd_solver="arpack").fit(X_train)
    # embedding-mode decode needs to know the source rep; X_hvg is gene-space-like
    # (PCA-invertible), X_state needs a decoder.
    is_embedding_mode = args.embed_key == "X_state"
    for a in (adata_train, adata_val, adata_test):
        if a is None:
            continue
        a.obsm["X_pca"] = pca.transform(_feature_matrix(a, args.embed_key)).astype(np.float32)
        a.uns["_pca_components"] = pca.components_.astype(np.float32)
        a.uns["_pca_mean"] = pca.mean_.astype(np.float32)
        a.uns["_pca_input_rep"] = args.embed_key if is_embedding_mode else "counts"
    print(f"  PCA: {args.n_pca_components} comps, var explained {pca.explained_variance_ratio_.sum():.3f}")

    # ── 3. Decoder for gene-space eval (embedding mode only) ──────────────────
    state_decoder = state_decoder_gene_idx = None
    if is_embedding_mode:
        if args.state_checkpoint:
            state_decoder, decoder_gene_names = load_state_decoder(args.state_checkpoint)
            # Align adata.X to the decoder's fixed gene set so decoded predictions and
            # ground-truth .X share a gene axis (mirrors run_replogle).
            if decoder_gene_names is not None:
                dg = set(decoder_gene_names)
                genes = [g for g in adata_train.var_names if g in dg]
                g2i = {g: i for i, g in enumerate(decoder_gene_names)}
                state_decoder_gene_idx = np.array([g2i[g] for g in genes], dtype=np.int64)
                adata_train = adata_train[:, genes].copy()
                if adata_val is not None:
                    adata_val = adata_val[:, genes].copy()
                if adata_test is not None:
                    adata_test = adata_test[:, genes].copy()
                print(f"  STATE decoder: evaluating on {len(genes)} overlapping genes")
        else:
            from sklearn.linear_model import Ridge

            Xtr = adata_train.obsm["X_pca"]
            Ytr = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else np.asarray(adata_train.X)
            ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)
            for a in (adata_train, adata_val, adata_test):
                if a is not None:
                    a.uns["_decoder_coef"] = ridge.coef_.astype(np.float32)
                    a.uns["_decoder_intercept"] = ridge.intercept_.astype(np.float32)

    # ── 4. Functional gene embeddings (replace / augment the perturbation rep) ─
    func_cfg = None
    if args.embeddings_dir:
        from cellflow.preprocessing import (
            FUNCTIONAL_EMBEDDING_SOURCES,
            load_functional_gene_embeddings,
        )

        sources = (
            list(FUNCTIONAL_EMBEDDING_SOURCES)
            if args.embedding_sources.strip().lower() == "all"
            else [s.strip() for s in args.embedding_sources.split(",") if s.strip()]
        )
        gene_id_map = _load_gene_id_map(args.embedding_gene_id_map)
        # Apply to every split so uns reps + (multi_stream) obs columns exist for
        # prepare_data / prepare_validation_data / predict.
        func_cfg = None
        for split_name, a in (("train", adata_train), ("val", adata_val), ("test", adata_test)):
            if a is None:
                continue
            cfg = load_functional_gene_embeddings(
                a, args.embeddings_dir, sources=sources, gene_cols=["condition"],
                base_group="gene", fusion=args.embedding_fusion,
                anchor=(args.embedding_anchor or None), gene_id_map=gene_id_map,
                on_missing="mean", ignore_values=[args.control_pert],
            )
            # coverage is per-split: held-out val/test genes are what drive generalization.
            print(f"  [{split_name}] functional-emb coverage {cfg.per_source_coverage} "
                  f"| fully-unmapped genes: {cfg.n_fully_unmapped}")
            if split_name == "train":
                func_cfg = cfg  # train cfg defines the covariate wiring used below
        print(f"  Functional embeddings: sources={func_cfg.sources} fusion={func_cfg.fusion}")

    # ── 5. CellFlow ───────────────────────────────────────────────────────────
    import jax
    import optax

    from cellflow.model import CellFlow
    from cellflow.training import Metrics

    if func_cfg is not None:
        pert_kwargs = func_cfg.prepare_data_kwargs()
    else:
        # fall back to a mean-PCA-profile gene embedding (like run_replogle default)
        genes = sorted(set(adata_train.obs["condition"]) - {args.control_pert})
        Xp = adata_train.obsm["X_pca"]
        cv = adata_train.obs["condition"].values
        gene_emb = {g: Xp[cv == g].mean(0).astype(np.float32) for g in genes if (cv == g).any()}
        for a in (adata_train, adata_val, adata_test):
            if a is not None:
                a.uns["gene_emb"] = gene_emb
        pert_kwargs = {"perturbation_covariates": {"gene": ["condition"]},
                       "perturbation_covariate_reps": {"gene": "gene_emb"}}

    cf = CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        split_covariates=[args.cell_type_key],
        max_combination_length=1,
        **pert_kwargs,
    )
    if adata_val is not None and (~adata_val.obs["is_control"]).any():
        val_conds = sorted(set(adata_val.obs.loc[~adata_val.obs["is_control"], "condition"]))
        cf.prepare_validation_data(adata_val, name="val",
                                   n_conditions_on_log_iteration=min(50, len(val_conds)))

    dims = tuple([args.hidden_dim] * args.n_hidden_layers)
    model_kwargs = {}
    if func_cfg is not None and func_cfg.fusion == "multi_stream":
        model_kwargs["layers_before_pool"] = func_cfg.layers_before_pool()
    cf.prepare_model(
        condition_embedding_dim=args.condition_embedding_dim,
        hidden_dims=dims, decoder_dims=dims, time_encoder_dims=dims,
        pooling="attention_token", optimizer=optax.adam(args.lr),
        **model_kwargs,
    )

    callbacks = [Metrics(metrics=["r_squared", "e_distance", "mmd"])]

    has_val = adata_val is not None and (~adata_val.obs["is_control"]).any()

    # Best-checkpoint on the validation set: training loss → 0 while held-out R²
    # degrades (overfitting), so we keep the inference params from the iteration
    # with the highest val_r_squared_mean instead of the last ones. Subclasses
    # Metrics so it computes the same value itself (order-independent) and snapshots
    # solver.vf_state_inference.params (what predict uses) to host memory.
    best_ckpt = None
    if has_val:
        class BestValCheckpoint(Metrics):
            def __init__(self):
                super().__init__(metrics=["r_squared"])
                self.best = None
                self.best_params = None
                self.best_iter = -1
                self._n = 0

            def _maybe_update(self, vs, vt, vp, solver):
                out = super().on_log_iteration(vs, vt, vp, solver)
                self._n += 1
                v = out.get("val_r_squared_mean")
                if v is None or v != v:  # missing / NaN
                    return
                if self.best is None or v > self.best:
                    self.best = float(v)
                    self.best_iter = self._n
                    self.best_params = jax.device_get(solver.vf_state_inference.params)

            def on_log_iteration(self, vs, vt, vp, solver):
                self._maybe_update(vs, vt, vp, solver)
                return {}

            def on_train_end(self, vs, vt, vp, solver):
                self._maybe_update(vs, vt, vp, solver)
                return {}

        best_ckpt = BestValCheckpoint()
        callbacks.append(best_ckpt)

    if args.wandb_project:
        from cellflow.training import WandbLogger

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        callbacks.append(WandbLogger(project=args.wandb_project, out_dir=save_path,
                                     config=vars(args), entity=args.wandb_entity or None,
                                     name=args.run_id or os.path.basename(save_path), tags=tags))

    cf.train(num_iterations=args.num_iterations, batch_size=args.batch_size,
             valid_freq=args.valid_freq, callbacks=callbacks,
             monitor_metrics=["val_r_squared_mean"] if has_val else [])

    # Restore the best-validation inference params (predict/eval + save use them).
    if best_ckpt is not None and best_ckpt.best_params is not None:
        import jax.numpy as jnp

        cf.solver.vf_state_inference = cf.solver.vf_state_inference.replace(
            params=jax.tree_util.tree_map(jnp.asarray, best_ckpt.best_params)
        )
        print(f"Restored best val_r_squared_mean={best_ckpt.best:.4f} "
              f"(validation #{best_ckpt.best_iter}) for eval/save.")
    cf.save(save_path, overwrite=True)
    print(f"Model saved to {save_path}/CellFlow.pkl")

    # ── 6. Predict + decode on TEST, write pred/real h5ads ────────────────────
    if adata_test is None or not (~adata_test.obs["is_control"]).any():
        print("No test perturbations; skipping final eval.")
        return
    final_path = os.path.join(save_path, "final_test")
    os.makedirs(final_path, exist_ok=True)

    # unique (condition, cell_type) pairs among perturbed test cells
    pert_test = adata_test.obs.loc[~adata_test.obs["is_control"], ["condition", args.cell_type_key]].drop_duplicates()
    covariate_df = pd.DataFrame({
        "condition": pert_test["condition"].values,
        args.cell_type_key: pert_test[args.cell_type_key].values,
        "is_control": False,
        "condition_name": [f"{c}|{ct}" for c, ct in zip(pert_test["condition"], pert_test[args.cell_type_key], strict=False)],
    })
    # multi_stream fusion duplicates the gene column per source (condition__<src>);
    # the prediction covariate_df must carry those columns too (same gene value).
    if func_cfg is not None:
        for cols in func_cfg.perturbation_covariates.values():
            for col in cols:
                if col not in covariate_df.columns:
                    covariate_df[col] = covariate_df["condition"].values
    control_cells = adata_test[adata_test.obs["is_control"]].copy()
    predictions = cf.predict(adata=control_cells, covariate_data=covariate_df,
                             sample_rep="X_pca", condition_id_key="condition_name")

    def _decode(X_pca):
        return reconstruct_from_pca(X_pca, adata_test, state_decoder=state_decoder,
                                    state_decoder_gene_idx=state_decoder_gene_idx)

    def _real(cells):
        # ground truth in the SAME space _decode() outputs: obsm[embed_key] for the
        # X_hvg (PCA-inverse) path, gene-space .X for the X_state (decoder) path.
        if is_embedding_mode:
            X = cells.X
            return np.asarray(X.toarray() if hasattr(X, "toarray") else X)
        return np.asarray(cells.obsm[args.embed_key])

    pred_expr, pred_names, real_expr, real_names = [], [], [], []
    # control baseline (needed by cell-eval, which computes effects relative to control)
    pred_expr.append(_decode(control_cells.obsm["X_pca"]))
    pred_names += ["control"] * pred_expr[-1].shape[0]
    real_expr.append(_real(control_cells))
    real_names += ["control"] * real_expr[-1].shape[0]

    skipped = []
    for _, row in covariate_df.iterrows():
        key = row["condition_name"]
        if key not in predictions:
            skipped.append(key)
            continue
        pred_expr.append(_decode(predictions[key]))
        pred_names += [row["condition"]] * pred_expr[-1].shape[0]
        rc = adata_test[(adata_test.obs["condition"] == row["condition"]) &
                        (adata_test.obs[args.cell_type_key] == row[args.cell_type_key])]
        real_expr.append(_real(rc))
        real_names += [row["condition"]] * real_expr[-1].shape[0]

    pred_path = os.path.join(final_path, "pred.h5ad")
    real_path = os.path.join(final_path, "real.h5ad")
    ad.AnnData(X=np.concatenate(pred_expr).astype(np.float32),
               obs=pd.DataFrame({"perturbation": pred_names})).write_h5ad(pred_path)
    ad.AnnData(X=np.concatenate(real_expr).astype(np.float32),
               obs=pd.DataFrame({"perturbation": real_names})).write_h5ad(real_path)
    if skipped:
        print(f"  {len(skipped)} test conditions not returned by predict() were skipped.")
    print(f"Wrote pred/real to {final_path}")

    # ── 7. cell-eval metrics (subprocess: pdex forks, which deadlocks with JAX) ──
    import json as _json
    import subprocess

    results_csv = os.path.join(final_path, "results.csv")
    agg_csv = os.path.join(final_path, "agg_results.csv")
    eval_script = f"""
import anndata as ad, json
from cell_eval import MetricsEvaluator
pred = ad.read_h5ad({_json.dumps(pred_path)})
real = ad.read_h5ad({_json.dumps(real_path)})
ev = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert="control",
                      pert_col="perturbation", num_threads={args.eval_num_threads})
results, agg = ev.compute()
results.write_csv({_json.dumps(results_csv)})
agg.write_csv({_json.dumps(agg_csv)})
"""
    try:
        proc = subprocess.run([sys.executable, "-c", eval_script], text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"cell-eval exited {proc.returncode}")
        agg_df = pd.read_csv(agg_csv)
        mean_row = agg_df[agg_df["statistic"] == "mean"].iloc[0].to_dict()
        print("Test metrics (mean across perturbations):")
        for k, v in mean_row.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: cell-eval failed ({e}); pred/real h5ads are written for manual eval.")


if __name__ == "__main__":
    main()
