from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd

from cellflow._logging import logger
from cellflow._types import Layers_separate_input_t, Layers_t, PathLike

__all__ = [
    "FunctionalEmbeddingConfig",
    "load_functional_gene_embeddings",
]


# Default per-source biological embedding files shipped by QuantumCell_embeddings.
# Keys are the ``sources`` names accepted by :func:`load_functional_gene_embeddings`,
# values are the ``.npz`` file stems (without extension).
FUNCTIONAL_EMBEDDING_SOURCES: dict[str, str] = {
    # PPI / network
    "consensus": "consensus",
    "STRING": "STRING",
    # pathway / ontology
    "Reactome": "Reactome",
    "GeneOntology": "GeneOntology",
    "MSigDB": "MSigDB",
    "WikiPathways": "WikiPathways",
    # functional genomics / assays
    "DepMap": "DepMap",
    "GTEx": "GTEx",
    "GWASAtlas": "GWASAtlas",
    "CellPainting": "CellPainting",
    # protein language model (sequence)
    "ESM-2": "ESM-2",
}

# Some sources ship under different filenames across environments (e.g. the signed
# vs unsigned GWAS Atlas build). Resolution tries these candidate stems in order.
# ``combined`` is the pre-fused "everything" vector (state_integration/combine.py);
# it is intentionally NOT in FUNCTIONAL_EMBEDDING_SOURCES so the curated "all" set
# excludes it — request it explicitly (typically with anchor=None).
_SOURCE_FILE_ALIASES: dict[str, list[str]] = {
    "GWASAtlas": ["GWASAtlas", "GWASAtlas_signed"],
    "combined": ["gene_embeddings_combined"],
}


@dataclass
class FunctionalEmbeddingConfig:
    """Configuration returned by :func:`load_functional_gene_embeddings`.

    Bundles the arguments that wire the loaded embeddings into
    :meth:`cellflow.model.CellFlow.prepare_data` and
    :meth:`cellflow.model.CellFlow.prepare_model`. This is the single object an
    ablation run varies: change ``sources`` (and ``fusion``) and the resulting
    covariate groups / representations follow automatically.

    Attributes
    ----------
    perturbation_covariates
        Maps each generated covariate group to the gene columns in ``adata.obs``.
        In ``'concat'`` mode there is a single group; in ``'multi_stream'`` mode
        there is one group per source. The first item is the primary group.
    perturbation_covariate_reps
        Maps each generated covariate group to the ``adata.uns`` key holding its
        ``{gene_name: vector}`` representation.
    sources
        The embedding sources included, in order (the anchor first, if any).
    anchor
        The always-on anchor source (e.g. ``"ESM-2"``) guaranteeing near-universal
        coverage so no gene is ever fully null, or :obj:`None` if disabled.
    fusion
        Fusion mode used (``'concat'`` or ``'multi_stream'``).
    embedding_dims
        Per-group embedding dimensionality.
    per_source_coverage
        For each source, the number of distinct gene identifiers that have a real
        (non-zero-filled) embedding. Genes missing from a source are zero-filled for
        that source only and still use every other source's vectors.
    n_fully_unmapped
        Number of distinct gene identifiers missing from *every* selected source
        (the only genes that carry no functional signal at all).
    """

    perturbation_covariates: dict[str, tuple[str, ...]]
    perturbation_covariate_reps: dict[str, str]
    sources: list[str]
    fusion: Literal["concat", "multi_stream"]
    anchor: str | None = None
    embedding_dims: dict[str, int] = field(default_factory=dict)
    per_source_coverage: dict[str, int] = field(default_factory=dict)
    n_fully_unmapped: int = 0

    @property
    def primary_group(self) -> str:
        """Name of the primary perturbation covariate group."""
        return next(iter(self.perturbation_covariates))

    def prepare_data_kwargs(
        self,
        extra_perturbation_covariates: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """Kwargs for :meth:`cellflow.model.CellFlow.prepare_data`.

        Parameters
        ----------
        extra_perturbation_covariates
            Additional perturbation covariate groups to merge in (e.g. dose or
            timing). Their column tuples must have the same length as the gene
            covariate groups.
        """
        pert_cov = dict(self.perturbation_covariates)
        if extra_perturbation_covariates:
            pert_cov.update({k: tuple(v) for k, v in extra_perturbation_covariates.items()})
        return {
            "perturbation_covariates": pert_cov,
            "perturbation_covariate_reps": dict(self.perturbation_covariate_reps),
        }

    def layers_before_pool(
        self,
        dims: Sequence[int] = (256, 256),
        dropout_rate: float = 0.0,
    ) -> Layers_t | Layers_separate_input_t:
        """Suggested ``layers_before_pool`` for :meth:`~cellflow.model.CellFlow.prepare_model`.

        In ``'multi_stream'`` mode returns a per-group ``dict`` (one MLP sub-network
        per source, using the encoder's ``separate_inputs`` path). In ``'concat'``
        mode returns a single shared MLP spec.
        """
        mlp: dict[str, Any] = {"layer_type": "mlp", "dims": list(dims), "dropout_rate": dropout_rate}
        if self.fusion == "multi_stream":
            return {group: [dict(mlp)] for group in self.perturbation_covariates}
        return [dict(mlp)]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; leaves zero rows untouched."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _resolve_source_file(embeddings_dir: PathLike, source: str) -> str:
    """Find the .npz file for a source, trying known filename aliases."""
    stem = FUNCTIONAL_EMBEDDING_SOURCES.get(source, source)
    candidates = _SOURCE_FILE_ALIASES.get(source, [stem])
    if stem not in candidates:
        candidates = [stem, *candidates]
    for cand in candidates:
        path = os.path.join(str(embeddings_dir), f"{cand}.npz")
        if os.path.exists(path):
            return path
    available = (
        sorted(f[:-4] for f in os.listdir(embeddings_dir) if f.endswith(".npz"))
        if os.path.isdir(embeddings_dir)
        else []
    )
    raise FileNotFoundError(
        f"No embedding file for source '{source}' in {embeddings_dir} "
        f"(tried {[f'{c}.npz' for c in candidates]}). Available: {available}"
    )


def _load_source_npz(
    path: str,
    l2_normalize: bool,
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    """Load one QuantumCell ``.npz`` into ``{ENSG: vector}``, its dim, and its mean.

    Genes flagged absent by the ``mask`` (zero-imputed) are dropped so they fall
    back to ``on_missing`` handling rather than contributing spurious zeros. The
    returned mean is over the *present* rows and is used as the nonzero fill for
    genes missing from this source.
    """
    data = np.load(path, allow_pickle=True)
    for key in ("embedding", "gene_ids"):
        if key not in data.files:
            raise KeyError(
                f"Embedding file {path} is missing required key '{key}'. "
                f"Found keys: {list(data.files)}. Expected the standardised "
                f"'embedding'/'gene_ids'/'mask' schema."
            )
    emb = np.asarray(data["embedding"], dtype=np.float32)
    gene_ids = np.asarray(data["gene_ids"]).astype(str)
    if emb.ndim != 2 or emb.shape[0] != len(gene_ids):
        raise ValueError(
            f"Embedding file {path}: 'embedding' shape {emb.shape} is incompatible with "
            f"{len(gene_ids)} gene_ids (expected (n_genes, dim))."
        )
    # ``mask`` True == gene absent / zero-imputed in the source.
    present = ~np.asarray(data["mask"], dtype=bool) if "mask" in data.files else np.ones(len(gene_ids), dtype=bool)
    if l2_normalize:
        emb = _l2_normalize(emb)
    dim = int(emb.shape[1])
    mapping = {gid: emb[i] for i, gid in enumerate(gene_ids) if present[i]}
    mean_vec = emb[present].mean(axis=0) if present.any() else np.zeros(dim, dtype=np.float32)
    return mapping, dim, mean_vec.astype(np.float32)


def _resolve_gene_cols(adata: ad.AnnData, gene_cols: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve ``gene_cols`` as a prefix (str) or an explicit list of obs columns."""
    if isinstance(gene_cols, str):
        cols = [c for c in adata.obs.columns if c.startswith(gene_cols)]
        if not cols:
            raise ValueError(f"No columns in `adata.obs` start with prefix '{gene_cols}'.")
        return tuple(sorted(cols))
    missing = [c for c in gene_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Gene columns not found in `adata.obs`: {missing}.")
    return tuple(sorted(gene_cols))


def _to_ensembl_mapper(gene_id_map: Mapping[str, str] | pd.Series | Callable[[str], str] | None) -> Callable[[str], str | None]:
    """Normalize ``gene_id_map`` into a callable obs_value -> ENSG (or None)."""
    if gene_id_map is None:
        return lambda g: g  # assume obs values are already Ensembl gene IDs
    if callable(gene_id_map):
        return gene_id_map  # type: ignore[return-value]
    if isinstance(gene_id_map, pd.Series):
        d = gene_id_map.to_dict()
        return lambda g: d.get(g)
    d = dict(gene_id_map)
    return lambda g: d.get(g)


def load_functional_gene_embeddings(
    adata: ad.AnnData,
    embeddings_dir: PathLike,
    sources: Sequence[str],
    gene_cols: str | Sequence[str] = "gene_target_",
    *,
    anchor: str | None = "ESM-2",
    base_group: str = "gene",
    fusion: Literal["concat", "multi_stream"] = "multi_stream",
    gene_id_map: Mapping[str, str] | pd.Series | Callable[[str], str] | None = None,
    l2_normalize: bool = True,
    on_missing: Literal["mean", "zero", "raise"] = "mean",
    null_value: float = 0.0,
    uns_prefix: str = "func_emb",
    ignore_values: Sequence[str] | None = None,
    copy: bool = False,
) -> FunctionalEmbeddingConfig:
    """Load QuantumCell functional gene embeddings and wire them as CellFlow conditions.

    Each perturbed gene (knockout / overexpression) is represented by the selected
    functional embeddings (PPI, pathway, GTEx, DepMap, GWAS, CellPainting, ESM-2),
    which are fed into CellFlow's :class:`~cellflow.networks.ConditionEncoder` in
    place of / alongside the default ESM-2 gene embedding. This targets the
    documented bottleneck for genetic perturbations, where a sequence-only ESM-2
    embedding is insufficient to capture functional effects.

    Two fusion modes:

    - ``'concat'``: concatenate all selected sources into one per-gene vector stored
      under ``adata.uns[uns_prefix]``; a single covariate group is produced.
    - ``'multi_stream'``: store one ``{gene: vector}`` dict per source under
      ``adata.uns[f"{uns_prefix}_{source}"]`` and produce one covariate group per
      source (all pointing at the same gene columns). Combined with a per-group
      ``layers_before_pool``, each source gets its own projection sub-network.

    Parameters
    ----------
    adata
        Annotated data matrix. Modified in place unless ``copy`` is :obj:`True`.
    embeddings_dir
        Directory containing the QuantumCell ``*.npz`` embedding files.
    sources
        Embedding sources to include, e.g. ``["consensus", "Reactome", "DepMap"]``.
        This is the ablation knob. Names are matched against
        :data:`FUNCTIONAL_EMBEDDING_SOURCES` (falling back to ``"{source}.npz"``).
    anchor
        An always-on source (default ``"ESM-2"``) prepended to ``sources`` if not
        already present. ESM-2 has near-universal gene coverage, so anchoring every
        configuration on it guarantees no gene is ever fully null and keeps the gene
        cohort identical across ablation runs (the ablation then measures what each
        functional view adds *on top of* the sequence baseline). Pass :obj:`None` to
        disable (e.g. to test functional views in isolation).
    gene_cols
        Either an ``adata.obs`` column-name prefix (default ``"gene_target_"``, as in
        :func:`~cellflow.preprocessing.get_esm_embedding`) or an explicit list of
        columns holding the perturbed gene identities.
    base_group
        Base name for the generated covariate groups.
    fusion
        ``'concat'`` or ``'multi_stream'`` (see above).
    gene_id_map
        Optional mapping from the gene identifier used in ``adata.obs`` (e.g. a gene
        symbol) to an Ensembl gene ID (``ENSG...``) matching the embedding files.
        A ``dict``, :class:`~pandas.Series`, or callable. If :obj:`None`, obs values
        are assumed to already be Ensembl gene IDs.
    l2_normalize
        Per-source row-wise L2 normalization so sources of different dimensionality
        (e.g. GTEx 268 vs PPI 64) contribute on comparable scales.
    on_missing
        How to fill a gene absent from a given source (filling is always per-source;
        the other sources still contribute their real vectors):

        - ``'mean'`` (default): fill with that source's mean embedding. Nonzero, so the
          condition element is never mistaken for a padded/empty slot (all-``null_value``
          elements are masked out by the encoder), and provides a sane "average gene"
          prior.
        - ``'zero'``: fill with ``null_value``. Legacy; risks a gene fully missing from
          all sources being masked out entirely (treated as no perturbation).
        - ``'raise'``: mean-fill per-source gaps but raise if any gene is missing from
          *every* source (no real signal anywhere).
    null_value
        Fill value used by ``on_missing='zero'`` (should match the ``null_value``
        passed to :meth:`~cellflow.model.CellFlow.prepare_data`).
    uns_prefix
        Base key under which representations are stored in ``adata.uns``.
    ignore_values
        Gene-column values to skip entirely (e.g. a control / non-targeting token).
        These are not added to the representation dicts.
    copy
        Operate on and return a copy of ``adata`` instead of modifying in place.

    Returns
    -------
    FunctionalEmbeddingConfig
        Configuration wiring the embeddings into ``prepare_data`` / ``prepare_model``.
        When ``copy`` is :obj:`True`, the modified copy is available as the ``adata``
        passed in (returned config references the same ``uns`` keys).
    """
    if copy:
        adata = adata.copy()
    if fusion not in ("concat", "multi_stream"):
        raise ValueError(f"`fusion` must be 'concat' or 'multi_stream', got {fusion!r}.")
    if not sources:
        raise ValueError("`sources` must be a non-empty sequence of embedding names.")

    # Prepend the anchor (e.g. ESM-2) if not already selected, so every configuration
    # shares a near-universally-covered source and no gene is ever fully null.
    sources = list(sources)
    if anchor is not None and anchor not in sources:
        sources = [anchor, *sources]

    resolved_cols = _resolve_gene_cols(adata, gene_cols)
    mapper = _to_ensembl_mapper(gene_id_map)
    ignore = set(ignore_values or [])

    # Collect the distinct gene identifiers actually present in the data.
    obs_values: set[str] = set()
    for col in resolved_cols:
        obs_values.update(str(v) for v in adata.obs[col].unique())
    obs_values -= ignore
    obs_values -= {"nan", "None"}

    # Load each requested source.
    per_source_maps: dict[str, dict[str, np.ndarray]] = {}
    per_source_dims: dict[str, int] = {}
    per_source_mean: dict[str, np.ndarray] = {}
    for source in sources:
        path = _resolve_source_file(embeddings_dir, source)
        mapping, dim, mean_vec = _load_source_npz(path, l2_normalize)
        per_source_maps[source] = mapping
        per_source_dims[source] = dim
        per_source_mean[source] = mean_vec

    def _vector(source: str, gene_value: str) -> np.ndarray | None:
        ensg = mapper(gene_value)
        if ensg is None:
            return None
        return per_source_maps[source].get(ensg)

    # Coverage is per-source and independent: a gene missing from one source is
    # zero-filled for *that source only* and still uses the real vectors from every
    # other source. A gene is only truly uninformative when it is missing from
    # *every* selected source (its whole representation becomes null).
    per_source_missing: dict[str, set[str]] = {
        source: {g for g in obs_values if _vector(source, g) is None} for source in sources
    }
    per_source_coverage = {s: len(obs_values) - len(m) for s, m in per_source_missing.items()}
    fully_unmapped = set.intersection(*per_source_missing.values()) if per_source_missing else set()

    fill_desc = "mean-filled" if on_missing != "zero" else f"filled with null_value={null_value}"
    for source, missing in per_source_missing.items():
        if missing:
            logger.info(
                f"[{source}] {len(missing)}/{len(obs_values)} genes have no embedding; "
                f"{fill_desc} for this source only (other sources still contribute)."
            )
    if fully_unmapped:
        msg = (
            f"{len(fully_unmapped)} of {len(obs_values)} gene identifiers have no embedding in "
            f"ANY selected source (e.g. {sorted(fully_unmapped)[:5]}); these carry no functional "
            f"signal and are indistinguishable from a null perturbation."
        )
        if on_missing == "raise":
            raise ValueError(msg)
        logger.warning(msg)

    def _filled(source: str, gene_value: str) -> np.ndarray:
        vec = _vector(source, gene_value)
        if vec is not None:
            return np.asarray(vec, dtype=np.float32)
        # nonzero mean-fill by default so the element is not masked out as padding;
        # 'zero' keeps the legacy null_value fill.
        if on_missing == "zero":
            return np.full((per_source_dims[source],), null_value, dtype=np.float32)
        return np.asarray(per_source_mean[source], dtype=np.float32)

    perturbation_covariates: dict[str, tuple[str, ...]] = {}
    perturbation_covariate_reps: dict[str, str] = {}
    embedding_dims: dict[str, int] = {}

    if fusion == "concat":
        rep: dict[str, np.ndarray] = {
            gene_value: np.concatenate([_filled(source, gene_value) for source in sources])
            for gene_value in obs_values
        }
        adata.uns[uns_prefix] = rep
        perturbation_covariates[base_group] = resolved_cols
        perturbation_covariate_reps[base_group] = uns_prefix
        embedding_dims[base_group] = int(sum(per_source_dims[s] for s in sources))
    else:  # multi_stream: one group + uns key per source
        # Each source becomes its own perturbation covariate group. CellFlow's
        # DataManager flattens all group columns into one key list, so groups may
        # not share obs columns (duplicates would collapse a condition row from a
        # Series into a DataFrame). We therefore give each source its own copies of
        # the gene columns, holding identical gene identities.
        for source in sources:
            uns_key = f"{uns_prefix}_{source}"
            group = f"{base_group}_{source}"
            source_cols = tuple(f"{col}__{source}" for col in resolved_cols)
            for src_col, orig_col in zip(source_cols, resolved_cols, strict=True):
                adata.obs[src_col] = adata.obs[orig_col].values
            adata.uns[uns_key] = {gene_value: _filled(source, gene_value) for gene_value in obs_values}
            perturbation_covariates[group] = source_cols
            perturbation_covariate_reps[group] = uns_key
            embedding_dims[group] = per_source_dims[source]

    return FunctionalEmbeddingConfig(
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_covariate_reps,
        sources=list(sources),
        anchor=anchor,
        fusion=fusion,
        embedding_dims=embedding_dims,
        per_source_coverage=per_source_coverage,
        n_fully_unmapped=len(fully_unmapped),
    )
