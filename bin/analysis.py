#!/usr/bin/env python3
"""
FAST replicate-aware Welch's t-test on pseudo_EPR per (Delivery, Cell_Type, UTR_Reference),
where Delivery/Cell_Type/Dox_Treatment/Replicate are PARSED from Sample_Stem.

Also COMPUTES TPM and RPKM (no intermediate long file is ever written):
  - Requires per-row counts (e.g., Total_Reads)
  - Requires UTR lengths from --fasta or --lengths-csv
  - Computes per-sample library size as sum(counts) over UTRs within each Sample_Stem
  - Adds TPM/RPKM to the Welch summary outputs (replicate padded to 3 + means)

Key speedups:
  1) Vectorized Sample_Stem parsing via .str.extract (no Python apply)
  2) TPM/RPKM via groupby.transform (no merges)
  3) COLLAPSE to ONE ROW per (Delivery,Cell,UTR,Dox,Replicate) before testing
     (avoids huge group DataFrames)
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

try:
    from statsmodels.stats.multitest import fdrcorrection
except ImportError as e:
    raise ImportError("statsmodels is required (pip install statsmodels)") from e

_bio_import_error = None
try:
    from Bio import SeqIO
except ImportError as _e:
    SeqIO = None
    _bio_import_error = str(_e)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ----------------------------- argparse --------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute TPM/RPKM then run Welch's t-test on pseudo_EPR (Dox vs NoDox) after parsing Sample_Stem"
    )

    p.add_argument("--input", required=True, help="Input long CSV")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--outfile", default="welches_t_test_EPR.csv", help="Output Welch results CSV name")

    # Column mapping
    p.add_argument("--sample-col", default="Sample_Stem", help="Sample stem column (default: Sample_Stem)")
    p.add_argument("--utr-col", default="UTR_Reference", help="UTR column (default: UTR_Reference)")
    p.add_argument("--epr-col", default="EPR", help="EPR column (default: EPR)")
    p.add_argument("--reads-col", default="Total_Reads",
                   help="Counts column used for TPM/RPKM and optional filtering (default: Total_Reads)")

    # Lengths input (one of these required)
    p.add_argument("--fasta", default=None,
                   help="FASTA to compute Length_bp (record IDs must match UTR_Reference).")
    p.add_argument("--fasta-id-split", default=None,
                   help="Optional delimiter to split FASTA record.id before matching to UTR_Reference.")
    p.add_argument("--fasta-id-field", type=int, default=0,
                   help="Field index to keep after split (default: 0).")
    p.add_argument("--lengths-csv", default=None,
                   help="CSV with columns: UTR_Reference,Length_bp (alternative to --fasta).")

    # Dox labels
    p.add_argument("--dox-label", default="Dox", help='Dox label (default: "Dox")')
    p.add_argument("--nodox-label", default="NoDox", help='NoDox label (default: "NoDox")')

    # pseudo_EPR pseudocount
    p.add_argument("--pseudo", type=float, default=1e-5, help="Pseudocount added to EPR (default: 1e-5)")

    # Sample_Stem parsing
    p.add_argument(
        "--sample-regex",
        default=r"^(?P<Delivery>[^_]+)_(?P<Cell_Type>[^_]+)_(?P<Dox_Treatment>Dox|NoDox)(?:_(?P<Timepoint>[^_]+))?_(?P<Replicate>R\d+)$",
        help=("Regex with named groups Delivery, Cell_Type, Dox_Treatment, Replicate.\n"
              "Optional Timepoint group (e.g. 4h, 24h) is parsed when present.\n"
              "Default matches: Delivery_CellType_(Dox|NoDox)[_Timepoint]_R#"),
    )

    # Filtering (optional)
    p.add_argument("--min-reads", type=int, default=50,
                   help="Min reads per (Sample_Stem, UTR) replicate row before stats (default: 100). Set to 0 to disable.")

    return p.parse_args()


# ----------------------------- lengths ---------------------------------------

def compute_lengths_from_fasta(fasta_path: str, id_split: str | None, id_field: int) -> pd.DataFrame:
    if SeqIO is None:
        detail = f" Import failed with: {_bio_import_error}" if _bio_import_error else ""
        raise ImportError(f"Biopython is required for --fasta (pip install biopython).{detail}")

    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        key = rec.id
        if id_split is not None:
            parts = key.split(id_split)
            if id_field >= len(parts):
                raise ValueError(
                    f"--fasta-id-field {id_field} out of range for FASTA id '{rec.id}' split by '{id_split}'"
                )
            key = parts[id_field]
        rows.append({"UTR_Reference": key, "Length_bp": len(rec.seq)})

    return pd.DataFrame(rows)


# ----------------------------- sample parsing --------------------------------

def parse_sample_stem(df: pd.DataFrame, sample_col: str, sample_regex: str) -> pd.DataFrame:
    parsed = df[sample_col].astype(str).str.extract(sample_regex)
    # Timepoint is optional — only verify required columns parsed successfully
    required = [c for c in ["Delivery", "Cell_Type", "Dox_Treatment", "Replicate"] if c in parsed.columns]
    if parsed[required].isna().any(axis=1).any():
        bad = parsed[required].isna().any(axis=1)
        examples = df.loc[bad, sample_col].astype(str).unique()[:10].tolist()
        raise ValueError(
            f"Failed to parse {bad.sum()} {sample_col} values using --sample-regex.\n"
            f"Examples: {examples}\n"
            f"Regex: {sample_regex}"
        )
    return pd.concat([df, parsed], axis=1)


# ----------------------------- TPM/RPKM --------------------------------------

def add_tpm_rpkm(df: pd.DataFrame, sample_col: str, reads_col: str) -> pd.DataFrame:
    """
    Requires: Length_bp, reads_col, sample_col
    Adds: LibSizeCounts, RPKM, TPM
    Uses transform (no merges) for speed + lower memory.
    """
    out = df.copy()

    out["LibSizeCounts"] = out.groupby(sample_col)[reads_col].transform("sum")

    # RPKM: counts * 1e9 / (lib_size_counts * length_bp)
    denom = (out["Length_bp"] * out["LibSizeCounts"]).replace(0, np.nan)
    out["RPKM"] = (out[reads_col] * 1e9) / denom

    # TPM: rate = counts / length_kb; TPM = rate / sum(rate) * 1e6 (within sample)
    length_kb = (out["Length_bp"] / 1000.0).replace(0, np.nan)
    out["_rate"] = out[reads_col] / length_kb
    out["_rate_sum"] = out.groupby(sample_col)["_rate"].transform("sum")
    out["TPM"] = (out["_rate"] / out["_rate_sum"].replace(0, np.nan)) * 1e6

    return out.drop(columns=["_rate", "_rate_sum"])


# ----------------------------- stats helpers ---------------------------------

def _tp_sort_key(tp):
    """Numeric sort key for timepoint strings (e.g. '24h' → 24)."""
    import re
    m = re.match(r"(\d+)", str(tp))
    return int(m.group(1)) if m else 0


def _pad(vals, pad_value, n=3):
    vals = list(vals)
    if len(vals) >= n:
        return vals[:n]
    return vals + [pad_value] * (n - len(vals))


def welch_from_arrays(
    dox_epr: np.ndarray,
    nodox_epr: np.ndarray,
    dox_reads: np.ndarray,
    nodox_reads: np.ndarray,
    dox_tpm: np.ndarray,
    nodox_tpm: np.ndarray,
    dox_rpkm: np.ndarray,
    nodox_rpkm: np.ndarray,
    pseudo: float,
):
    n_dox = int(len(dox_epr))
    n_nodox = int(len(nodox_epr))

    # Missing condition: return NA stats but still report padded values
    if n_dox == 0 or n_nodox == 0:
        return {
            "n_dox": n_dox,
            "n_nodox": n_nodox,

            "Dox_EPR_Rep1": _pad(dox_epr, pseudo)[0],
            "Dox_EPR_Rep2": _pad(dox_epr, pseudo)[1],
            "Dox_EPR_Rep3": _pad(dox_epr, pseudo)[2],
            "NoDox_EPR_Rep1": _pad(nodox_epr, pseudo)[0],
            "NoDox_EPR_Rep2": _pad(nodox_epr, pseudo)[1],
            "NoDox_EPR_Rep3": _pad(nodox_epr, pseudo)[2],

            "mean_dox": np.nan,
            "mean_nodox": np.nan,

            "Dox_Reads_Rep1": _pad(dox_reads, 0)[0],
            "Dox_Reads_Rep2": _pad(dox_reads, 0)[1],
            "Dox_Reads_Rep3": _pad(dox_reads, 0)[2],
            "NoDox_Reads_Rep1": _pad(nodox_reads, 0)[0],
            "NoDox_Reads_Rep2": _pad(nodox_reads, 0)[1],
            "NoDox_Reads_Rep3": _pad(nodox_reads, 0)[2],

            "mean_reads_dox": np.nan,
            "mean_reads_nodox": np.nan,

            "Dox_TPM_Rep1": _pad(dox_tpm, 0.0)[0],
            "Dox_TPM_Rep2": _pad(dox_tpm, 0.0)[1],
            "Dox_TPM_Rep3": _pad(dox_tpm, 0.0)[2],
            "NoDox_TPM_Rep1": _pad(nodox_tpm, 0.0)[0],
            "NoDox_TPM_Rep2": _pad(nodox_tpm, 0.0)[1],
            "NoDox_TPM_Rep3": _pad(nodox_tpm, 0.0)[2],
            "mean_TPM_dox": np.nan,
            "mean_TPM_nodox": np.nan,

            "Dox_RPKM_Rep1": _pad(dox_rpkm, 0.0)[0],
            "Dox_RPKM_Rep2": _pad(dox_rpkm, 0.0)[1],
            "Dox_RPKM_Rep3": _pad(dox_rpkm, 0.0)[2],
            "NoDox_RPKM_Rep1": _pad(nodox_rpkm, 0.0)[0],
            "NoDox_RPKM_Rep2": _pad(nodox_rpkm, 0.0)[1],
            "NoDox_RPKM_Rep3": _pad(nodox_rpkm, 0.0)[2],
            "mean_RPKM_dox": np.nan,
            "mean_RPKM_nodox": np.nan,

            "log2fc": np.nan,
            "t_stat": np.nan,
            "p_value": 1.0
        }

    # Welch t-test on pseudo_EPR — one-sided (Dox > NoDox)
    t_stat, p_val = ttest_ind(dox_epr, nodox_epr, equal_var=False, alternative='greater')

    mean_dox = float(np.mean(dox_epr))
    mean_nodox = float(np.mean(nodox_epr))
    log2fc = float(np.log2(mean_dox / mean_nodox))  # pseudo already included

    dox_epr_reps = _pad(dox_epr, pseudo)
    nodox_epr_reps = _pad(nodox_epr, pseudo)

    dox_reads_reps = _pad(dox_reads, 0)
    nodox_reads_reps = _pad(nodox_reads, 0)

    dox_tpm_reps = _pad(dox_tpm, 0.0)
    nodox_tpm_reps = _pad(nodox_tpm, 0.0)

    dox_rpkm_reps = _pad(dox_rpkm, 0.0)
    nodox_rpkm_reps = _pad(nodox_rpkm, 0.0)

    return {
        "n_dox": n_dox,
        "n_nodox": n_nodox,

        "Dox_EPR_Rep1": dox_epr_reps[0],
        "Dox_EPR_Rep2": dox_epr_reps[1],
        "Dox_EPR_Rep3": dox_epr_reps[2],
        "NoDox_EPR_Rep1": nodox_epr_reps[0],
        "NoDox_EPR_Rep2": nodox_epr_reps[1],
        "NoDox_EPR_Rep3": nodox_epr_reps[2],

        "mean_dox": mean_dox,
        "mean_nodox": mean_nodox,

        "Dox_Reads_Rep1": dox_reads_reps[0],
        "Dox_Reads_Rep2": dox_reads_reps[1],
        "Dox_Reads_Rep3": dox_reads_reps[2],
        "NoDox_Reads_Rep1": nodox_reads_reps[0],
        "NoDox_Reads_Rep2": nodox_reads_reps[1],
        "NoDox_Reads_Rep3": nodox_reads_reps[2],

        "mean_reads_dox": float(np.mean(dox_reads)) if len(dox_reads) else np.nan,
        "mean_reads_nodox": float(np.mean(nodox_reads)) if len(nodox_reads) else np.nan,

        "Dox_TPM_Rep1": dox_tpm_reps[0],
        "Dox_TPM_Rep2": dox_tpm_reps[1],
        "Dox_TPM_Rep3": dox_tpm_reps[2],
        "NoDox_TPM_Rep1": nodox_tpm_reps[0],
        "NoDox_TPM_Rep2": nodox_tpm_reps[1],
        "NoDox_TPM_Rep3": nodox_tpm_reps[2],
        "mean_TPM_dox": float(np.mean(dox_tpm)) if len(dox_tpm) else np.nan,
        "mean_TPM_nodox": float(np.mean(nodox_tpm)) if len(nodox_tpm) else np.nan,

        "Dox_RPKM_Rep1": dox_rpkm_reps[0],
        "Dox_RPKM_Rep2": dox_rpkm_reps[1],
        "Dox_RPKM_Rep3": dox_rpkm_reps[2],
        "NoDox_RPKM_Rep1": nodox_rpkm_reps[0],
        "NoDox_RPKM_Rep2": nodox_rpkm_reps[1],
        "NoDox_RPKM_Rep3": nodox_rpkm_reps[2],
        "mean_RPKM_dox": float(np.mean(dox_rpkm)) if len(dox_rpkm) else np.nan,
        "mean_RPKM_nodox": float(np.mean(nodox_rpkm)) if len(nodox_rpkm) else np.nan,

        "log2fc": log2fc,
        "t_stat": float(t_stat),
        "p_value": float(p_val)
    }


# ----------------------------- QC plots --------------------------------------

def _plot_volcano(res_tp, outdir, tp_label, utr_col):
    data = res_tp.dropna(subset=["log2fc", "p_value"]).copy()
    if data.empty:
        return
    data["neg_log10_p"] = -np.log10(data["p_value"].clip(lower=1e-300))
    sig = data["fdr"] < 0.05

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data.loc[~sig, "log2fc"], data.loc[~sig, "neg_log10_p"],
               c="gray", alpha=0.4, s=6, rasterized=True, label="FDR ≥ 0.05")
    ax.scatter(data.loc[sig, "log2fc"], data.loc[sig, "neg_log10_p"],
               c="#d62728", alpha=0.6, s=6, rasterized=True, label="FDR < 0.05")
    ax.axhline(-np.log10(0.05), color="#d62728", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log2(Dox / NoDox)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"Volcano — {tp_label}")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"volcano_{tp_label}.png"), dpi=150)
    plt.close(fig)


def _plot_pca(rep_tp, outdir, tp_label):
    if len(rep_tp) < 2:
        return

    rep = rep_tp.copy()
    rep["_sample"] = (
        rep["Delivery"] + "_" + rep["Cell_Type"] + "_" +
        rep["Dox_Treatment"] + "_" + rep["Replicate"].astype(str)
    )

    pivot = rep.pivot_table(index="_sample", columns="_UTR", values="TPM", aggfunc="mean")
    mat = np.log2(pivot.fillna(0).values + 1)

    if mat.shape[0] < 2:
        return

    try:
        from sklearn.decomposition import PCA as _PCA
        mat_c = mat - mat.mean(axis=0)
        pca = _PCA(n_components=min(2, mat.shape[0]))
        coords = pca.fit_transform(mat_c)
        var_exp = pca.explained_variance_ratio_ * 100
    except ImportError:
        mat_c = mat - mat.mean(axis=0)
        U, S, _ = np.linalg.svd(mat_c, full_matrices=False)
        coords = U[:, :2] * S[:2]
        total = float((S ** 2).sum())
        var_exp = (S[:2] ** 2 / total * 100) if total > 0 else np.zeros(2)

    sample_meta = (
        rep[["_sample", "Delivery", "Cell_Type", "Dox_Treatment"]]
        .drop_duplicates("_sample")
        .set_index("_sample")
    )
    sample_order = list(pivot.index)

    dox_palette = {"Dox": "#d62728", "NoDox": "#1f77b4"}
    marker_cycle = ["o", "s", "^", "D", "v", "<", ">"]
    cell_markers = {
        ct: marker_cycle[i % len(marker_cycle)]
        for i, ct in enumerate(sample_meta["Cell_Type"].unique())
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for samp, row in sample_meta.iterrows():
        idx = sample_order.index(samp)
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=dox_palette.get(row["Dox_Treatment"], "gray"),
            marker=cell_markers.get(row["Cell_Type"], "o"),
            s=90, alpha=0.85,
            label=f"{row['Delivery']}_{row['Cell_Type']}_{row['Dox_Treatment']}"
        )

    handles, labs = ax.get_legend_handles_labels()
    by_label = dict(zip(labs, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)" if len(var_exp) > 0 else "PC1")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)" if len(var_exp) > 1 else "PC2")
    ax.set_title(f"PCA (log2 TPM+1) — {tp_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pca_{tp_label}.png"), dpi=150)
    plt.close(fig)


def _plot_cdf(res_tp, outdir, tp_label):
    data = res_tp.dropna(subset=["log2fc"])
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for (delivery, cell_type), grp in data.groupby(["Delivery", "Cell_Type"]):
        vals = np.sort(grp["log2fc"].values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=f"{delivery}_{cell_type}")

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log2(Dox / NoDox)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title(f"CDF of log2FC — {tp_label}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cdf_{tp_label}.png"), dpi=150)
    plt.close(fig)


def _plot_total_reads_barplot(rep_tp, outdir, tp_label, reads_col="Total_Reads"):
    if rep_tp.empty:
        return

    rep = rep_tp.copy()
    rep["_sample"] = (
        rep["Delivery"] + "_" + rep["Cell_Type"] + "_" +
        rep["Dox_Treatment"] + "_" + rep["Replicate"].astype(str)
    )
    sample_reads = (
        rep.groupby(["_sample", "Dox_Treatment"])["Total_Reads"]
        .sum()
        .reset_index()
        .sort_values("_sample")
    )

    dox_palette = {"Dox": "#d62728", "NoDox": "#1f77b4"}
    colors = [dox_palette.get(d, "gray") for d in sample_reads["Dox_Treatment"]]

    fig, ax = plt.subplots(figsize=(max(6, len(sample_reads) * 0.5), 5))
    ax.bar(sample_reads["_sample"], sample_reads["Total_Reads"], color=colors)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Total Reads")
    ax.set_title(f"Total Reads per Library — {tp_label}")
    ax.tick_params(axis="x", rotation=45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in dox_palette.items()], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"total_reads_barplot_{tp_label}.png"), dpi=150)
    plt.close(fig)


def _plot_mean_epr_barplot(rep_tp, outdir, tp_label):
    if rep_tp.empty:
        return

    rep = rep_tp.copy()
    rep["_sample"] = (
        rep["Delivery"] + "_" + rep["Cell_Type"] + "_" +
        rep["Dox_Treatment"] + "_" + rep["Replicate"].astype(str)
    )
    sample_epr = (
        rep.groupby(["_sample", "Dox_Treatment"])["_pseudo_EPR"]
        .mean()
        .reset_index()
        .sort_values("_sample")
    )

    dox_palette = {"Dox": "#d62728", "NoDox": "#1f77b4"}
    colors = [dox_palette.get(d, "gray") for d in sample_epr["Dox_Treatment"]]

    fig, ax = plt.subplots(figsize=(max(6, len(sample_epr) * 0.5), 5))
    ax.bar(sample_epr["_sample"], sample_epr["_pseudo_EPR"], color=colors)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Mean pseudo-EPR")
    ax.set_title(f"Mean EPR per Library — {tp_label}")
    ax.tick_params(axis="x", rotation=45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in dox_palette.items()], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"mean_epr_barplot_{tp_label}.png"), dpi=150)
    plt.close(fig)


def plot_qc_per_timepoint(result, df_rep, outdir, has_timecourse, utr_col):
    if not _HAS_MPL:
        print("  [skip] matplotlib not available — install it to generate QC plots")
        return

    if has_timecourse:
        timepoints = sorted(result["Timepoint"].dropna().unique(), key=_tp_sort_key)
    else:
        timepoints = [None]

    for tp in timepoints:
        tp_label = str(tp) if tp is not None else "all"
        if tp is not None:
            res_tp = result[result["Timepoint"] == tp]
            rep_tp = df_rep[df_rep["Timepoint"] == tp]
        else:
            res_tp = result
            rep_tp = df_rep

        print(f"  Plotting QC for timepoint: {tp_label}")
        _plot_volcano(res_tp, outdir, tp_label, utr_col)
        _plot_pca(rep_tp, outdir, tp_label)
        _plot_cdf(res_tp, outdir, tp_label)
        _plot_total_reads_barplot(rep_tp, outdir, tp_label)
        _plot_mean_epr_barplot(rep_tp, outdir, tp_label)


# ----------------------------- main ------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Reading input CSV...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} rows")

    # Validate required columns
    required = {args.sample_col, args.utr_col, args.epr_col, args.reads_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    # Numeric coercion
    df[args.epr_col] = pd.to_numeric(df[args.epr_col], errors="coerce")
    df[args.reads_col] = pd.to_numeric(df[args.reads_col], errors="coerce").fillna(0)

    # Lengths (must provide fasta or lengths-csv)
    print("Loading UTR lengths...")
    if args.lengths_csv is not None:
        lengths = pd.read_csv(args.lengths_csv)
        if not {"UTR_Reference", "Length_bp"}.issubset(lengths.columns):
            raise ValueError("--lengths-csv must contain columns: UTR_Reference,Length_bp")
        if args.utr_col != "UTR_Reference":
            lengths = lengths.rename(columns={"UTR_Reference": args.utr_col})
    elif args.fasta is not None:
        lengths = compute_lengths_from_fasta(args.fasta, args.fasta_id_split, args.fasta_id_field)
        if args.utr_col != "UTR_Reference":
            lengths = lengths.rename(columns={"UTR_Reference": args.utr_col})
    else:
        raise ValueError("Provide UTR lengths via --fasta or --lengths-csv")

    # Merge lengths
    print("Merging lengths...")
    df = df.merge(lengths[[args.utr_col, "Length_bp"]].drop_duplicates(), on=args.utr_col, how="left")
    if df["Length_bp"].isna().any():
        bad = df.loc[df["Length_bp"].isna(), args.utr_col].dropna().unique()[:10].tolist()
        raise ValueError(
            f"Missing Length_bp for some UTRs after merge (examples: {bad}). "
            "Your FASTA IDs (or lengths CSV) must match UTR_Reference exactly."
        )

    # Compute TPM/RPKM in memory (fast)
    print("Computing TPM/RPKM...")
    df = add_tpm_rpkm(df, sample_col=args.sample_col, reads_col=args.reads_col)

    # Parse Sample_Stem -> Delivery, Cell_Type, Dox_Treatment, [Timepoint], Replicate (fast)
    print("Parsing Sample_Stem...")
    df = parse_sample_stem(df, sample_col=args.sample_col, sample_regex=args.sample_regex)

    has_timecourse = "Timepoint" in df.columns and df["Timepoint"].notna().any()
    if has_timecourse:
        tps = sorted(df["Timepoint"].dropna().unique().tolist())
        print(f"  Timecourse detected — {len(tps)} timepoints: {tps}")

    # Validate Dox labels exist
    dox_vals = set(df["Dox_Treatment"].astype(str).unique())
    expected = {args.dox_label, args.nodox_label}
    if not expected.issubset(dox_vals):
        raise ValueError(
            f"Dox_Treatment values found: {sorted(dox_vals)}; expected at least {sorted(expected)}.\n"
            f"If your labels differ, set --dox-label/--nodox-label."
        )

    # pseudo_EPR
    df["_pseudo_EPR"] = df[args.epr_col] + args.pseudo

    # Key
    df["_UTR"] = df[args.utr_col]

    # ------------------------------------------------------------
    # CRITICAL SPEEDUP: collapse to ONE row per replicate per UTR
    # ------------------------------------------------------------
    print("Collapsing to replicate-level data...")
    rep_cols = ["Delivery", "Cell_Type", "_UTR", "Dox_Treatment", "Replicate"]
    if has_timecourse:
        rep_cols = ["Delivery", "Cell_Type", "Timepoint", "_UTR", "Dox_Treatment", "Replicate"]
    df_rep = (
        df.groupby(rep_cols, as_index=False)
          .agg(
              _pseudo_EPR=("_pseudo_EPR", "mean"),
              Total_Reads=(args.reads_col, "sum"),
              TPM=("TPM", "mean"),
              RPKM=("RPKM", "mean"),
          )
    )
    print(f"  {len(df_rep)} replicate-level rows")

    # Optional min-reads filter (per replicate row *after* collapsing)
    if args.min_reads > 0:
        before = len(df_rep)
        df_rep = df_rep[df_rep["Total_Reads"] >= args.min_reads].copy()
        print(f"  Filtered to {len(df_rep)} rows (>= {args.min_reads} reads), removed {before - len(df_rep)}")

    # ------------------------------------------------------------
    # Run Welch's t-test per (Delivery, Cell_Type, [Timepoint], UTR)
    # ------------------------------------------------------------
    print("Running Welch's t-tests...")
    rows = []
    group_cols = ["Delivery", "Cell_Type", "_UTR"]
    if has_timecourse:
        group_cols = ["Delivery", "Cell_Type", "Timepoint", "_UTR"]

    groups = list(df_rep.groupby(group_cols, dropna=False, sort=False))
    print(f"  Processing {len(groups)} groups...")

    for i, (keys, g) in enumerate(groups):
        if (i + 1) % 10000 == 0:
            print(f"    Processed {i + 1}/{len(groups)} groups...")

        if has_timecourse:
            delivery, cell_type, timepoint, utr = keys
        else:
            delivery, cell_type, utr = keys
            timepoint = None

        dox = g[g["Dox_Treatment"] == args.dox_label]
        nodox = g[g["Dox_Treatment"] == args.nodox_label]

        stats = welch_from_arrays(
            dox_epr=dox["_pseudo_EPR"].to_numpy(),
            nodox_epr=nodox["_pseudo_EPR"].to_numpy(),
            dox_reads=dox["Total_Reads"].to_numpy(),
            nodox_reads=nodox["Total_Reads"].to_numpy(),
            dox_tpm=dox["TPM"].to_numpy(),
            nodox_tpm=nodox["TPM"].to_numpy(),
            dox_rpkm=dox["RPKM"].to_numpy(),
            nodox_rpkm=nodox["RPKM"].to_numpy(),
            pseudo=args.pseudo
        )

        stats["Delivery"] = delivery
        stats["Cell_Type"] = cell_type
        if has_timecourse:
            stats["Timepoint"] = timepoint
        stats["_UTR"] = utr
        rows.append(stats)

    print("Building results dataframe...")
    result = pd.DataFrame(rows).rename(columns={"_UTR": args.utr_col})

    # FDR — only on rows with a valid p-value; NaN rows (untestable) are left as NaN.
    print("Computing FDR...")
    result["fdr"] = np.nan
    testable = result["p_value"].notna()
    if testable.any():
        _, fdr_vals = fdrcorrection(result.loc[testable, "p_value"].to_numpy())
        result.loc[testable, "fdr"] = fdr_vals

    # Relative_Abundance: mean_RPKM at this timepoint / mean_RPKM at earliest timepoint
    # Computed separately for Dox and NoDox; NaN for non-timecourse runs.
    if has_timecourse:
        print("Computing Relative_Abundance (RPKM / earliest-timepoint RPKM)...")
        tps = sorted(result["Timepoint"].dropna().unique(), key=_tp_sort_key)
        earliest = tps[0]
        print(f"  Baseline timepoint: {earliest}")

        base_cols = ["Delivery", "Cell_Type", args.utr_col]
        baseline = (
            result[result["Timepoint"] == earliest]
            [base_cols + ["mean_RPKM_dox", "mean_RPKM_nodox"]]
            .rename(columns={"mean_RPKM_dox": "_base_dox", "mean_RPKM_nodox": "_base_nodox"})
        )
        result = result.merge(baseline, on=base_cols, how="left")
        result["RA_Dox"]   = result["mean_RPKM_dox"]   / result["_base_dox"].replace(0, np.nan)
        result["RA_NoDox"] = result["mean_RPKM_nodox"]  / result["_base_nodox"].replace(0, np.nan)
        result = result.drop(columns=["_base_dox", "_base_nodox"])

    # Reorder
    front = ["Delivery", "Cell_Type"]
    if has_timecourse and "Timepoint" in result.columns:
        front.append("Timepoint")
    front.append(args.utr_col)
    other = [c for c in result.columns if c not in front]
    result = result[front + other]

    outpath = os.path.join(args.outdir, args.outfile)
    result.to_csv(outpath, index=False)
    print(f"Wrote Welch test results: {outpath}")
    print(f"  {len(result)} tests completed")

    print("Generating QC plots...")
    plot_qc_per_timepoint(result, df_rep, args.outdir, has_timecourse, args.utr_col)
    print("QC plots written.")


if __name__ == "__main__":
    main()