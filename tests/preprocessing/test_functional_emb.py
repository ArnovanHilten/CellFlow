import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellflow.preprocessing import (
    FunctionalEmbeddingConfig,
    load_functional_gene_embeddings,
)

# Synthetic gene universe (Ensembl-like IDs) and sources with distinct dims.
GENES = [f"ENSG{i:011d}" for i in range(8)]
# ESM-2 is the anchor; give it full coverage like the real file.
SOURCE_DIMS = {"ESM-2": 5, "consensus": 4, "Reactome": 6, "DepMap": 3}


def _write_npz(path, gene_ids, dim, absent=()):
    rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
    emb = rng.standard_normal((len(gene_ids), dim)).astype(np.float32)
    mask = np.array([g in set(absent) for g in gene_ids], dtype=bool)  # True == absent
    np.savez(
        path,
        embedding=emb,
        mask=mask,
        gene_ids=np.array(gene_ids, dtype="<U15"),
        source=np.array(str(path.stem)),
        dim=np.array(dim, dtype=np.int64),
    )


@pytest.fixture()
def emb_dir(tmp_path):
    # `consensus` deliberately misses the last gene to exercise the mask path.
    for source, dim in SOURCE_DIMS.items():
        absent = [GENES[-1]] if source == "consensus" else []
        _write_npz(tmp_path / f"{source}.npz", GENES, dim, absent=absent)
    return tmp_path


def _make_genetic_adata(gene_values):
    n_obs = 200
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "gene_target_1": rng.choice(gene_values, n_obs),
            "gene_target_2": rng.choice(gene_values, n_obs),
        }
    )
    adata = ad.AnnData(X=rng.random((n_obs, 20)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.random((n_obs, 10)).astype(np.float32)
    # designate a control population sharing a gene token
    control_idcs = rng.choice(n_obs, n_obs // 10, replace=False)
    for col in ["gene_target_1", "gene_target_2"]:
        adata.obs.loc[adata.obs.index[control_idcs], col] = "control"
    adata.obs["control"] = adata.obs["gene_target_1"] == "control"
    for col in ["gene_target_1", "gene_target_2"]:
        adata.obs[col] = adata.obs[col].astype("category")
    return adata


class TestLoadFunctionalGeneEmbeddings:
    """Source/fusion logic, isolated with anchor=None."""

    def test_concat_fusion(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        sources = ["consensus", "Reactome", "DepMap"]
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=sources, gene_cols="gene_target_", fusion="concat", anchor=None
        )
        assert isinstance(cfg, FunctionalEmbeddingConfig)
        assert list(cfg.perturbation_covariates) == ["gene"]
        assert cfg.perturbation_covariates["gene"] == ("gene_target_1", "gene_target_2")
        assert cfg.perturbation_covariate_reps == {"gene": "func_emb"}
        total = sum(SOURCE_DIMS[s] for s in sources)
        assert cfg.embedding_dims["gene"] == total
        rep = adata.uns["func_emb"]
        assert set(rep) == {str(v) for c in ("gene_target_1", "gene_target_2") for v in adata.obs[c].unique()}
        assert all(v.shape == (total,) for v in rep.values())

    def test_multi_stream_fusion(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        sources = ["consensus", "Reactome", "DepMap"]
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=sources, gene_cols="gene_target_", fusion="multi_stream", anchor=None
        )
        assert list(cfg.perturbation_covariates) == [f"gene_{s}" for s in sources]
        for s in sources:
            assert cfg.perturbation_covariates[f"gene_{s}"] == (f"gene_target_1__{s}", f"gene_target_2__{s}")
            assert f"gene_target_1__{s}" in adata.obs.columns
            assert cfg.perturbation_covariate_reps[f"gene_{s}"] == f"func_emb_{s}"
            assert cfg.embedding_dims[f"gene_{s}"] == SOURCE_DIMS[s]
            assert all(v.shape == (SOURCE_DIMS[s],) for v in adata.uns[f"func_emb_{s}"].values())
        lbp = cfg.layers_before_pool(dims=(8, 8))
        assert isinstance(lbp, dict)
        assert set(lbp) == set(cfg.perturbation_covariates)

    def test_l2_normalization(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata,
            emb_dir,
            sources=["Reactome"],
            fusion="multi_stream",
            l2_normalize=True,
            anchor=None,
            ignore_values=["control"],  # only real (present) genes remain
        )
        rep = adata.uns["func_emb_Reactome"]
        norms = [np.linalg.norm(v) for v in rep.values()]
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)
        assert cfg.fusion == "multi_stream"

    def test_gene_id_map_symbols(self, emb_dir):
        symbol_to_ensg = {f"SYM{i}": GENES[i] for i in range(6)}
        adata = _make_genetic_adata(list(symbol_to_ensg))
        cfg = load_functional_gene_embeddings(
            adata,
            emb_dir,
            sources=["DepMap"],
            fusion="concat",
            gene_id_map=symbol_to_ensg,
            anchor=None,
            ignore_values=["control"],
            on_missing="raise",
        )
        assert cfg.n_fully_unmapped == 0
        assert "SYM0" in adata.uns["func_emb"]

    def test_ignore_values(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["DepMap"], fusion="concat", anchor=None, ignore_values=["control"]
        )
        assert "control" not in adata.uns["func_emb"]
        assert cfg.n_fully_unmapped == 0

    def test_copy_does_not_mutate_input(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        load_functional_gene_embeddings(
            adata, emb_dir, sources=["DepMap"], fusion="concat", anchor=None, copy=True
        )
        assert "func_emb" not in adata.uns


class TestCoverageAndFill:
    def test_partial_coverage_keeps_other_sources(self, emb_dir):
        # GENES[-1] is masked only in `consensus`; DepMap still covers it. It must NOT
        # count as fully unmapped, and its DepMap segment must be real (not filled).
        adata = _make_genetic_adata([GENES[0], GENES[-1]])
        cfg = load_functional_gene_embeddings(
            adata,
            emb_dir,
            sources=["consensus", "DepMap"],
            fusion="concat",
            anchor=None,
            ignore_values=["control"],
            on_missing="raise",  # would raise if partial were treated as unmapped
        )
        assert cfg.n_fully_unmapped == 0
        assert cfg.per_source_coverage["DepMap"] == len(adata.uns["func_emb"])
        assert cfg.per_source_coverage["consensus"] < len(adata.uns["func_emb"])

    def test_mean_fill_is_nonzero(self, emb_dir):
        # A gene missing from consensus is mean-filled (nonzero) by default, so its
        # condition element is never mistaken for a padded/empty slot.
        adata = _make_genetic_adata([GENES[0], GENES[-1]])
        load_functional_gene_embeddings(
            adata,
            emb_dir,
            sources=["consensus"],
            fusion="multi_stream",
            anchor=None,
            ignore_values=["control"],
            on_missing="mean",
        )
        filled = adata.uns["func_emb_consensus"][GENES[-1]]
        assert not np.allclose(filled, 0.0)

    def test_zero_fill_legacy(self, emb_dir):
        adata = _make_genetic_adata([GENES[0], GENES[-1]])
        load_functional_gene_embeddings(
            adata,
            emb_dir,
            sources=["consensus"],
            fusion="multi_stream",
            anchor=None,
            ignore_values=["control"],
            on_missing="zero",
        )
        assert np.allclose(adata.uns["func_emb_consensus"][GENES[-1]], 0.0)

    def test_fully_unmapped_raise(self, emb_dir):
        # "control" maps to nothing in any source -> fully unmapped.
        adata = _make_genetic_adata([GENES[0], GENES[-1]])
        with pytest.raises(ValueError, match="no embedding"):
            load_functional_gene_embeddings(
                adata, emb_dir, sources=["consensus"], fusion="concat", anchor=None, on_missing="raise"
            )


class TestAnchor:
    def test_anchor_prepended(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["consensus"], fusion="multi_stream", anchor="ESM-2"
        )
        assert cfg.anchor == "ESM-2"
        assert cfg.sources[0] == "ESM-2"  # anchor first (primary group)
        assert set(cfg.sources) == {"ESM-2", "consensus"}
        assert "gene_ESM-2" in cfg.perturbation_covariates

    def test_anchor_not_duplicated(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["ESM-2", "consensus"], fusion="concat", anchor="ESM-2"
        )
        assert cfg.sources.count("ESM-2") == 1

    def test_gwas_signed_alias(self, emb_dir):
        # On some environments GWAS ships as GWASAtlas_signed.npz; "GWASAtlas" must
        # still resolve to it.
        _write_npz(emb_dir / "GWASAtlas_signed.npz", GENES, 7)
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["GWASAtlas"], fusion="multi_stream", anchor=None,
            ignore_values=["control"],
        )
        assert cfg.embedding_dims["gene_GWASAtlas"] == 7
        assert all(v.shape == (7,) for v in adata.uns["func_emb_GWASAtlas"].values())

    def test_missing_source_lists_available(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        with pytest.raises(FileNotFoundError, match="Available:"):
            load_functional_gene_embeddings(
                adata, emb_dir, sources=["NoSuchSource"], fusion="concat", anchor=None
            )

    def test_combined_shortcut(self, emb_dir):
        # 'combined' resolves to gene_embeddings_combined.npz and is NOT in the
        # curated 'all' set.
        _write_npz(emb_dir / "gene_embeddings_combined.npz", GENES, 12)
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["combined"], fusion="concat", anchor=None,
            ignore_values=["control"],
        )
        assert cfg.sources == ["combined"]
        assert cfg.embedding_dims["gene"] == 12

    def test_bad_schema_raises_clearly(self, emb_dir):
        import numpy as _np

        # a file without the required 'gene_ids' key
        _np.savez(emb_dir / "Broken.npz", embedding=_np.zeros((8, 4), dtype=_np.float32))
        adata = _make_genetic_adata(GENES[:6])
        with pytest.raises(KeyError, match="missing required key"):
            load_functional_gene_embeddings(
                adata, emb_dir, sources=["Broken"], fusion="concat", anchor=None
            )

    def test_anchor_none(self, emb_dir):
        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["consensus"], fusion="concat", anchor=None
        )
        assert cfg.anchor is None
        assert cfg.sources == ["consensus"]

    def test_anchor_guarantees_no_null(self, emb_dir):
        # Even a gene missing from the functional source keeps a real ESM-2 segment,
        # so its concat vector is never all-zeros (would otherwise be masked out).
        adata = _make_genetic_adata([GENES[0], GENES[-1]])
        load_functional_gene_embeddings(
            adata,
            emb_dir,
            sources=["consensus"],
            fusion="concat",
            anchor="ESM-2",
            ignore_values=["control"],
            on_missing="zero",  # consensus segment zeroed, ESM-2 segment still real
        )
        vec = adata.uns["func_emb"][GENES[-1]]
        assert not np.allclose(vec, 0.0)


class TestIntegrationWithCellFlow:
    """The generated config must drive prepare_data -> prepare_model -> train."""

    @pytest.mark.parametrize("fusion", ["concat", "multi_stream"])
    def test_end_to_end(self, emb_dir, fusion):
        import cellflow

        adata = _make_genetic_adata(GENES[:6])
        cfg = load_functional_gene_embeddings(
            adata, emb_dir, sources=["consensus", "DepMap"], fusion=fusion  # anchor ESM-2 default
        )
        cf = cellflow.model.CellFlow(adata, solver="otfm")
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            **cfg.prepare_data_kwargs(),
        )
        cf.prepare_model(
            condition_embedding_dim=2,
            hidden_dims=(2, 2),
            decoder_dims=(2, 2),
            layers_before_pool=cfg.layers_before_pool(dims=(4,)),
        )
        cf.train(num_iterations=2)
        assert cf.solver.is_trained
