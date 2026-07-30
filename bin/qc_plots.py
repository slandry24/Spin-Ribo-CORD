#!/usr/bin/env python3
"""
Generate QC plots for RNA editing analysis:
1. Average EPR per library (from Welch test results)
2. Average reads per library member (from raw data)
3. Total reads per sample (summed across all UTRs)
4. CDF of pool composition (one plot per cell type)
5. Volcano plots showing log2FC vs -log10(FDR) (one plot per cell type)
6. Spearman correlation heatmaps between samples (one plot per cell type)
7. PCA plots showing sample clustering (one plot per cell type)

All plots show Dox vs NoDox conditions across cell types and delivery methods.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate QC plots: EPR per library and reads per contig"
    )

    # Input files
    p.add_argument("--welch-results", required=True,
                   help="Welch test results CSV (output from welch t-test script)")
    p.add_argument("--raw-data", required=True,
                   help="Raw input CSV with per-sample counts")

    # Output
    p.add_argument("--outdir", required=True,
                   help="Output directory for plots")

    # Column names
    p.add_argument("--sample-col", default="Sample_Stem",
                   help="Sample stem column (default: Sample_Stem)")
    p.add_argument("--utr-col", default="UTR_Reference",
                   help="UTR column (default: UTR_Reference)")
    p.add_argument("--reads-col", default="Total_Reads",
                   help="Reads column (default: Total_Reads)")
    p.add_argument("--epr-col", default="EPR",
                   help="EPR column (default: EPR)")

    # Sample parsing
    p.add_argument(
        "--sample-regex",
        default=r"^(?P<Delivery>[^_]+)_(?P<Cell_Type>[^_]+)_(?P<Dox_Treatment>Dox|NoDox)(?:_(?P<Timepoint>[^_]+))?_(?P<Replicate>R\d+)$",
        help="Regex to parse Sample_Stem into Delivery, Cell_Type, Dox_Treatment, [Timepoint], Replicate"
    )

    # Plot customization
    p.add_argument("--delivery1", default="Lenti",
                   help="First delivery method name (default: Lenti)")
    p.add_argument("--delivery2", default="IVTMods",
                   help="Second delivery method name (default: IVTMods)")
    p.add_argument("--color1", default="#4747d1",
                   help="Color for first delivery (default: #4747d1)")
    p.add_argument("--color2", default="#f13636",
                   help="Color for second delivery (default: #f13636)")

    return p.parse_args()


def safe_sorted_unique_str(s: pd.Series) -> list[str]:
    """Return sorted unique values, safely coerced to strings (handles mixed int/str)."""
    return sorted(s.dropna().astype(str).unique().tolist())


def normalize_dox_treatment(series):
    """Normalize Dox_Treatment labels consistently."""
    return (series
            .astype(str)
            .str.replace("\u2212", "-", regex=False)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace("NoDox", "No Dox", regex=False)
            .str.replace("No_Dox", "No Dox", regex=False)
            .str.replace("mean_nodox", "No Dox", regex=False)
            .str.replace("mean_dox", "Dox", regex=False)
            .replace({"+Dox": "Dox", "-Dox": "No Dox"}))


def parse_sample_stem(df, sample_col, sample_regex):
    """Parse Sample_Stem into Delivery, Cell_Type, Dox_Treatment, [Timepoint], Replicate."""
    parsed = df[sample_col].astype(str).str.extract(sample_regex)
    # Timepoint is optional — only check required columns for failures
    required = [c for c in ["Delivery", "Cell_Type", "Dox_Treatment", "Replicate"] if c in parsed.columns]
    if parsed[required].isna().any(axis=1).any():
        bad = parsed[required].isna().any(axis=1)
        examples = df.loc[bad, sample_col].astype(str).unique()[:10].tolist()
        raise ValueError(
            f"Failed to parse {bad.sum()} {sample_col} values.\n"
            f"Examples: {examples}\n"
            f"Regex: {sample_regex}"
        )
    return pd.concat([df, parsed], axis=1)


def plot_bars_with_upper_error(ax, data, title, palette, hue_order):
    """Plot grouped bar chart with upward-only error bars."""
    data = data.copy()

    # FIX: mixed int/str Cell_Type breaks sorting in Python3
    data["Cell_Type"] = data["Cell_Type"].astype(str)
    data["Delivery"] = data["Delivery"].astype(str)

    cell_types = safe_sorted_unique_str(data["Cell_Type"])
    x_pos = np.arange(len(cell_types))
    width = 0.35

    for i, delivery in enumerate(hue_order):
        delivery = str(delivery)
        delivery_data = data[data["Delivery"] == delivery]
        means = []
        stds = []

        for cell_type in cell_types:
            cell_data = delivery_data[delivery_data["Cell_Type"] == cell_type]
            if not cell_data.empty:
                means.append(cell_data["mean"].values[0])
                stds.append(cell_data["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        # Plot bars with asymmetric error bars (only upward)
        offset = width * (i - 0.5)
        ax.bar(
            x_pos + offset, means, width,
            label=delivery, color=palette.get(delivery, "#999999"),
            edgecolor='black', linewidth=2
        )

        ax.errorbar(
            x_pos + offset, means,
            yerr=[np.zeros(len(means)), stds],
            fmt='none', color='black', capsize=5,
            capthick=2, elinewidth=2
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_title(title, fontsize=18, fontweight="bold")


def plot_epr_per_library(welch_results, outdir, palette, hue_order):
    """Generate EPR per library plot from Welch test results."""
    print("\n=== Generating EPR per library plot ===")

    print(f"Plotting {len(welch_results)} UTRs")

    # Prepare data
    df = welch_results[['Delivery', 'Cell_Type', 'UTR_Reference', 'mean_nodox', 'mean_dox']].copy()

    # Ensure consistent types (prevents mixed int/str issues)
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"] = df["Delivery"].astype(str)

    # Reshape to long format
    df_long = pd.melt(
        df,
        id_vars=["Cell_Type", "Delivery", "UTR_Reference"],
        value_vars=["mean_nodox", "mean_dox"],
        var_name="Dox_Treatment",
        value_name="Mean_EPR"
    )

    # Normalize Dox_Treatment
    df_long["Dox_Treatment"] = normalize_dox_treatment(df_long["Dox_Treatment"])

    print("Counts by Dox_Treatment:\n", df_long["Dox_Treatment"].value_counts(dropna=False))

    # Calculate statistics
    stats_df = (df_long.groupby(["Cell_Type", "Delivery", "Dox_Treatment"])["Mean_EPR"]
                .agg(['mean', 'std', 'count'])
                .reset_index())

    print(f"\nStats summary:\n{stats_df}")

    # Create plot
    sns.set_theme(context="talk", style="white")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Dox plot
    dox_data = stats_df[stats_df["Dox_Treatment"] == "Dox"]
    plot_bars_with_upper_error(axes[0], dox_data, "Mean EPR + Dox", palette, hue_order)
    axes[0].set_xlabel("Cell Type", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Mean EPR", fontsize=14, fontweight="bold")
    axes[0].set_ylim(bottom=0)

    # NoDox plot
    nodox_data = stats_df[stats_df["Dox_Treatment"] == "No Dox"]
    plot_bars_with_upper_error(axes[1], nodox_data, "Mean EPR – Dox", palette, hue_order)
    axes[1].set_xlabel("Cell Type", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].set_ylim(bottom=0)

    # Legend
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="Delivery", frameon=False,
                   fontsize=12, title_fontsize=12, loc="center left",
                   bbox_to_anchor=(1.02, 0.5))

    # Style
    for ax in axes:
        ax.tick_params(axis='y', which='both', left=True, right=False, labelsize=12, width=1.5)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelsize=12, width=1.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    outpath = os.path.join(outdir, "mean_epr_per_library_barplot.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def plot_reads_per_contig(raw_data, sample_col, reads_col, sample_regex,
                          outdir, palette, hue_order):
    """Generate reads per contig per sample plot from raw data."""
    print("\n=== Generating reads per contig plot ===")

    df = raw_data.copy()

    # Parse Sample_Stem if columns don't exist
    required = ['Delivery', 'Cell_Type', 'Dox_Treatment', 'Replicate']
    if not all(col in df.columns for col in required):
        print("Parsing Sample_Stem...")
        df = parse_sample_stem(df, sample_col, sample_regex)

    # Ensure consistent types
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"] = df["Delivery"].astype(str)

    # Normalize Dox_Treatment
    df["Dox_Treatment"] = normalize_dox_treatment(df["Dox_Treatment"])

    print("Counts by Dox_Treatment:\n", df["Dox_Treatment"].value_counts(dropna=False))

    tp_iter = [(None, "")]
    if "Timepoint" in df.columns:
        tps = sorted(df["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
        tp_iter += [(tp, f"_{tp}") for tp in tps]

    for tp, tp_suffix in tp_iter:
        plot_df = df if tp is None else df[df["Timepoint"] == tp]
        tp_label = "" if tp is None else f" ({tp})"

        stats_df = (plot_df.groupby(["Cell_Type", "Delivery", "Dox_Treatment"])[reads_col]
                    .agg(['mean', 'std', 'count'])
                    .reset_index())

        print(f"\nReads stats{tp_label}:\n{stats_df}")

        sns.set_theme(context="talk", style="white")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        dox_data = stats_df[stats_df["Dox_Treatment"] == "Dox"]
        plot_bars_with_upper_error(axes[0], dox_data, f"Reads Distribution + Dox{tp_label}", palette, hue_order)
        axes[0].set_xlabel("Cell Type", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Average Reads Per Library Member", fontsize=14, fontweight="bold")
        dox_max = (dox_data['mean'] + dox_data['std']).max()

        nodox_data = stats_df[stats_df["Dox_Treatment"] == "No Dox"]
        plot_bars_with_upper_error(axes[1], nodox_data, f"Reads Distribution – Dox{tp_label}", palette, hue_order)
        axes[1].set_xlabel("Cell Type", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("")
        nodox_max = (nodox_data['mean'] + nodox_data['std']).max()

        max_val = max(dox_max, nodox_max)
        for ax in axes:
            ax.set_ylim(0, max_val * 1.1)

        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels, title="Delivery", frameon=False,
                       fontsize=12, title_fontsize=12, loc="center left",
                       bbox_to_anchor=(1.02, 0.5))

        for ax in axes:
            ax.tick_params(axis='y', which='both', left=True, right=False, labelsize=12, width=1.5)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelsize=12, width=1.5)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("bold")
            sns.despine(ax=ax, top=True, right=True)

        plt.tight_layout()
        outpath = os.path.join(outdir, f"read_distribution_per_contig_barplot{tp_suffix}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved: {outpath}")
        plt.close()


def plot_total_reads_per_sample(raw_data, sample_col, reads_col, sample_regex,
                                outdir, palette, hue_order):
    """Generate total reads per sample plot (summed across all UTRs)."""
    print("\n=== Generating total reads per sample plot ===")

    df = raw_data.copy()

    # Parse Sample_Stem if columns don't exist
    required = ['Delivery', 'Cell_Type', 'Dox_Treatment', 'Replicate']
    if not all(col in df.columns for col in required):
        print("Parsing Sample_Stem...")
        df = parse_sample_stem(df, sample_col, sample_regex)

    # Ensure consistent types
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"] = df["Delivery"].astype(str)

    # Normalize Dox_Treatment
    df["Dox_Treatment"] = normalize_dox_treatment(df["Dox_Treatment"])

    def upper_sd_errorbar(x):
        mean = np.mean(x)
        std = np.std(x)
        return (mean, mean + std)

    tp_iter = [(None, "")]
    if "Timepoint" in df.columns:
        tps = sorted(df["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
        tp_iter += [(tp, f"_{tp}") for tp in tps]

    for tp, tp_suffix in tp_iter:
        plot_df = df if tp is None else df[df["Timepoint"] == tp]
        tp_label = "" if tp is None else f" ({tp})"

        summed_reads = (
            plot_df.groupby(["Cell_Type", "Delivery", "Dox_Treatment", "Replicate"])[reads_col]
              .sum()
              .reset_index()
              .rename(columns={reads_col: "Total_Reads"})
        )

        print(f"\nTotal reads per sample{tp_label}:\n{summed_reads.head()}")

        sns.set_theme(context="talk", style="white")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        sns.barplot(
            data=summed_reads[summed_reads["Dox_Treatment"] == "Dox"],
            x="Cell_Type", y="Total_Reads", hue="Delivery",
            palette=palette, hue_order=hue_order,
            edgecolor="black", linewidth=2,
            ax=axes[0], errorbar=upper_sd_errorbar, capsize=0.1,
            saturation=1.0
        )
        axes[0].set_title(f"Total Reads + Dox{tp_label}", fontsize=18, fontweight="bold")
        axes[0].set_xlabel("Cell Type", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Average Reads Per Sample", fontsize=14, fontweight="bold")
        axes[0].get_legend().remove()

        sns.barplot(
            data=summed_reads[summed_reads["Dox_Treatment"] == "No Dox"],
            x="Cell_Type", y="Total_Reads", hue="Delivery",
            palette=palette, hue_order=hue_order,
            edgecolor="black", linewidth=2,
            ax=axes[1], errorbar=upper_sd_errorbar, capsize=0.1,
            saturation=1.0
        )
        axes[1].set_title(f"Total Reads – Dox{tp_label}", fontsize=18, fontweight="bold")
        axes[1].set_xlabel("Cell Type", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("")

        for ax in axes:
            all_lines = [child for child in ax.get_children()
                         if isinstance(child, matplotlib.lines.Line2D)]
            for line in all_lines:
                line.set_color('black')
                line.set_linewidth(2)
                line.set_markeredgecolor('black')
                line.set_markerfacecolor('black')

        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(
            handles, labels, title="Delivery", frameon=False,
            fontsize=12, title_fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.5)
        )

        for ax in axes:
            ax.tick_params(axis='y', which='both', left=True, right=False, labelsize=12, width=1.5)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelsize=12, width=1.5)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("bold")
            sns.despine(ax=ax, top=True, right=True)

        plt.tight_layout()
        outpath = os.path.join(outdir, f"total_reads_barplot{tp_suffix}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved: {outpath}")
        plt.close()


def plot_cdf_pool_composition(raw_data, sample_col, utr_col, reads_col, sample_regex,
                              outdir, palette, hue_order):
    """Generate CDF plots showing pool composition per cell type."""
    print("\n=== Generating CDF pool composition plots ===")

    df = raw_data.copy()

    # Parse Sample_Stem if columns don't exist
    required = ['Delivery', 'Cell_Type', 'Dox_Treatment', 'Replicate']
    if not all(col in df.columns for col in required):
        print("Parsing Sample_Stem...")
        df = parse_sample_stem(df, sample_col, sample_regex)

    # Ensure consistent types
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"] = df["Delivery"].astype(str)

    # Normalize Dox_Treatment
    df["Dox_Treatment"] = normalize_dox_treatment(df["Dox_Treatment"])

    # Per-sample normalization
    df["Total_Reads_Sample"] = df.groupby(sample_col)[reads_col].transform("sum")
    df["Percent_of_Library"] = (df[reads_col] / df["Total_Reads_Sample"]) * 100

    tp_iter = [(None, "")]
    if "Timepoint" in df.columns:
        tps = sorted(df["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
        tp_iter += [(tp, f"_{tp}") for tp in tps]

    sns.set_theme(context="talk", style="white")

    for tp, tp_suffix in tp_iter:
        plot_df = df if tp is None else df[df["Timepoint"] == tp]
        tp_label = "" if tp is None else f" — {tp}"

        avg_df = (
            plot_df.groupby(["Delivery", "Cell_Type", "Dox_Treatment", utr_col], as_index=False)
              ["Percent_of_Library"].mean()
        )

        cdf_list = []
        for (delivery, cell, dox), sub in avg_df.groupby(["Delivery", "Cell_Type", "Dox_Treatment"]):
            sub = sub.sort_values("Percent_of_Library", ascending=False).reset_index(drop=True)
            sub["Variant_Number"] = np.arange(1, len(sub) + 1)
            sub["CDF"] = np.cumsum(sub["Percent_of_Library"]) / np.sum(sub["Percent_of_Library"])
            sub["Delivery"] = delivery
            sub["Cell_Type"] = cell
            sub["Dox_Treatment"] = dox
            cdf_list.append(sub)

        if not cdf_list:
            continue
        cdf_df = pd.concat(cdf_list, ignore_index=True)

        for cell_type in safe_sorted_unique_str(cdf_df["Cell_Type"]):
            sub_df = cdf_df[cdf_df["Cell_Type"].astype(str) == str(cell_type)].copy()

            plt.figure(figsize=(9, 6))

            sns.lineplot(
                data=sub_df,
                x="Variant_Number",
                y="CDF",
                hue="Delivery",
                style="Dox_Treatment",
                palette=palette,
                linewidth=2
            )

            plt.title(f"CDF of Pool Composition — {cell_type}{tp_label}", fontsize=18, weight="bold", pad=16)
            plt.xlabel("Member Rank", fontsize=16, weight="bold")
            plt.ylabel("Cumulative Fraction of Library", fontsize=16, weight="bold")
            plt.ylim(0, 1)

            plt.xticks(fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.tick_params(axis='y', which='both', left=True, right=False)
            plt.tick_params(axis='x', which='both', bottom=True, top=False)
            sns.despine(top=True, right=True)

            plt.legend(title=None, loc="lower right", frameon=False)
            plt.tight_layout()

            safe_name = str(cell_type).replace(" ", "_")
            outpath = os.path.join(outdir, f"CDF_Avg_byCondition_{safe_name}{tp_suffix}.png")
            plt.savefig(outpath, dpi=300)
            print(f"Saved: {outpath}")
            plt.close()


def plot_volcano(welch_results, outdir, palette, hue_order):
    """Generate volcano plots (log2FC vs -log10(FDR)) per cell type."""
    print("\n=== Generating volcano plots ===")

    result = welch_results.copy()

    # Validate required columns
    required = ['Cell_Type', 'Delivery', 'log2fc', 'fdr']
    missing = set(required) - set(result.columns)
    if missing:
        print(f"WARNING: Missing columns for volcano plot: {sorted(missing)}")
        return

    # Ensure consistent types
    result["Cell_Type"] = result["Cell_Type"].astype(str)
    result["Delivery"] = result["Delivery"].astype(str)

    tp_iter = [(None, "")]
    if "Timepoint" in result.columns:
        tps = sorted(result["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
        tp_iter += [(tp, f"_{tp}") for tp in tps]

    sns.set_theme(context='talk', style="white")
    cells = safe_sorted_unique_str(result['Cell_Type'])

    if len(cells) == 0:
        print("No Cell_Type values found; nothing to plot.")
        return

    for tp, tp_suffix in tp_iter:
        tp_data = result if tp is None else result[result["Timepoint"] == tp]
        tp_label = "" if tp is None else f" | {tp}"

        for cell in cells:
            print(f"Generating volcano plot for: {cell}{tp_label}")
            sub = tp_data[tp_data['Cell_Type'] == str(cell)].copy()

            if sub.empty:
                print(f"  Skipping {cell}: no rows.")
                continue

            sub['log2fc'] = pd.to_numeric(sub['log2fc'], errors='coerce')
            sub['fdr'] = pd.to_numeric(sub['fdr'], errors='coerce')
            before = len(sub)
            sub = sub.dropna(subset=['log2fc', 'fdr'])
            after = len(sub)
            if after == 0:
                print(f"  Skipping {cell}: all {before} rows dropped by NaNs in log2fc/fdr.")
                continue

            pass_mask = (sub['log2fc'] > 0.5) & (sub['fdr'] < 0.05)
            total_pass = int(pass_mask.sum())
            per_group = sub.loc[pass_mask].groupby('Delivery').size()

            count_dict = {}
            for delivery in hue_order:
                delivery = str(delivery)
                count_dict[delivery] = int(per_group.get(delivery, 0))

            count_str = ", ".join([f"{k}={v}" for k, v in count_dict.items()])
            print(f"  Passing (log2fc>0.5 & FDR<0.05): {total_pass}  [{count_str}]")

            fdr_floor = 1e-10
            yvals = -np.log10(sub['fdr'].clip(lower=fdr_floor))

            fig = plt.figure(figsize=(10, 7))
            ax = sns.scatterplot(
                data=sub,
                x='log2fc',
                y=yvals,
                hue='Delivery',
                palette=palette,
                s=15,
                edgecolor='black',
                alpha=0.6,
                linewidth=0.3,
                legend=True
            )

            ax.axhline(y=-np.log10(0.05), color='black', linestyle='--')
            ax.axvline(x=0.5, color='black', linestyle='--')
            ax.axvline(x=-0.5, color='black', linestyle='--')

            ax.set_xlim(-10, 10)
            ax.set_ylim(bottom=0, top=8)
            ax.set_xlabel('Log2FC(EPR Dox/EPR NoDox)', fontweight='bold', color='black', fontsize=18)
            ax.set_ylabel('-log10(FDR)', fontweight='bold', color='black', fontsize=18)
            ax.set_title(f"{cell}{tp_label}", fontweight='bold', fontsize=20, color='black')

            sns.despine(top=True, right=True)
            plt.xticks(fontsize=16, fontweight='bold', color='black')
            plt.yticks(fontsize=16, fontweight='bold', color='black')
            plt.tick_params(axis='y', which='both', left=True, right=False)
            plt.tick_params(axis='x', which='both', bottom=True, top=False)

            plt.legend(prop={'weight': 'bold', 'size': 13}, frameon=False)

            annot_lines = [f"Pass log2FC>0.5 & FDR<0.05:", f"Total = {total_pass}"]
            for delivery in hue_order:
                delivery = str(delivery)
                annot_lines.append(f"{delivery} = {count_dict[delivery]}")
            annot_text = "\n".join(annot_lines)

            ax.text(
                0.02, 0.98, annot_text,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=12, fontweight='bold',
                color="black"
            )

            safe_name = str(cell).replace(" ", "_")
            outpath = os.path.join(outdir, f"{safe_name}{tp_suffix}_volcano.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {outpath}")
            plt.close()


def plot_spearman_correlation(raw_data, sample_col, utr_col, epr_col, sample_regex,
                              outdir, palette, hue_order):
    """Generate single Spearman correlation heatmap across all samples."""
    print("\n=== Generating Spearman correlation heatmap ===")

    df = raw_data.copy()

    # Parse Sample_Stem if columns don't exist
    required = ['Delivery', 'Cell_Type', 'Dox_Treatment']
    if not all(col in df.columns for col in required):
        print("Parsing Sample_Stem...")
        df = parse_sample_stem(df, sample_col, sample_regex)

    # Ensure consistent types
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"] = df["Delivery"].astype(str)

    # Normalize Dox_Treatment
    df["Dox_Treatment"] = normalize_dox_treatment(df["Dox_Treatment"])

    # Check for EPR column
    if epr_col not in df.columns:
        print(f"WARNING: EPR column '{epr_col}' not found. Skipping correlation plot.")
        return

    df[epr_col] = pd.to_numeric(df[epr_col], errors='coerce')

    # Pivot: UTRs as rows, samples as columns, values = EPR
    epr_wide = df.pivot_table(index=utr_col, columns=sample_col, values=epr_col, aggfunc='mean')

    # Drop contigs with too many NAs (optional - adjust threshold as needed)
    epr_wide = epr_wide.dropna(thresh=3)

    if epr_wide.shape[1] < 2:
        print(f"  Skipping: need ≥2 samples. Got {epr_wide.shape[1]}.")
        return

    # Fill remaining NaN with 0
    epr_wide = epr_wide.fillna(0)

    # Compute sample-to-sample Spearman correlation matrix
    cor_matrix = epr_wide.corr(method='spearman')

    # Set theme
    sns.set_theme(
        context='talk',
        style='white',
        font_scale=0.5
    )

    # Clustered heatmap using hierarchical clustering
    g = sns.clustermap(
        cor_matrix,
        method='average',       # linkage method
        metric='correlation',   # distance metric
        cmap='plasma',
        annot=False,
        figsize=(10, 10),
        vmax=1,
        vmin=0,
        dendrogram_ratio=(0.07, 0.07),
        cbar_pos=(1.02, 0.3, 0.03, 0.4)
    )

    # Make sample names bold
    for label in g.ax_heatmap.get_xticklabels():
        label.set_fontweight('bold')
    for label in g.ax_heatmap.get_yticklabels():
        label.set_fontweight('bold')

    # Remove axis labels and adjust tick positions
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.xaxis.set_ticks_position('bottom')
    g.ax_heatmap.tick_params(axis='y', which='major', left=False, right=True, length=6, width=1)
    g.ax_heatmap.tick_params(axis='x', which='major', bottom=True, top=False, length=6, width=1)

    # Save
    outpath = os.path.join(outdir, "spearman.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def plot_pca(raw_data, sample_col, utr_col, epr_col, sample_regex, outdir, hue_order):
    """Generate PCA plots per cell type showing sample clustering."""
    print("\n=== Generating PCA plots ===")

    df = raw_data.copy()

    # Parse Sample_Stem if columns don't exist
    required = ['Delivery', 'Cell_Type', 'Dox_Treatment']
    if not all(col in df.columns for col in required):
        print("Parsing Sample_Stem...")
        df = parse_sample_stem(df, sample_col, sample_regex)

    # Ensure consistent types
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"] = df["Delivery"].astype(str)

    # Normalize Dox_Treatment
    df["Dox_Treatment"] = normalize_dox_treatment(df["Dox_Treatment"])

    # Check for EPR column
    if epr_col not in df.columns:
        print(f"WARNING: EPR column '{epr_col}' not found. Skipping PCA plots.")
        return

    df[epr_col] = pd.to_numeric(df[epr_col], errors='coerce')

    # Palette for platform + condition
    group_palette = {
        f"{hue_order[0]}_Dox": "#4747d1",
        f"{hue_order[0]}_No Dox": "#9797f7",
        f"{hue_order[1]}_Dox": "#f13636",
        f"{hue_order[1]}_No Dox": "#ff9999",
        "UnknownPlatform_UnknownCondition": "#cccccc",
    }

    sns.set_theme(context='talk', style="white")

    tp_iter = [(None, "")]
    if "Timepoint" in df.columns:
        tps = sorted(df["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
        tp_iter += [(tp, f"_{tp}") for tp in tps]

    for tp, tp_suffix in tp_iter:
        tp_df = df if tp is None else df[df["Timepoint"] == tp]
        tp_label = "" if tp is None else f" | {tp}"

        for cell_type in safe_sorted_unique_str(tp_df["Cell_Type"]):
            print(f"Generating PCA for: {cell_type}{tp_label}")
            sub = tp_df[tp_df["Cell_Type"] == str(cell_type)].copy()

            wide = sub.pivot_table(index=sample_col, columns=utr_col, values=epr_col, aggfunc='mean')
            wide = wide.fillna(0)

            keep_cols = wide.columns[wide.std(axis=0) > 0]
            wide = wide[keep_cols]

            if wide.shape[0] < 2 or wide.shape[1] < 2:
                print(f"  Skipping {cell_type}{tp_label}: need ≥2 samples and ≥2 variable contigs. Got {wide.shape}.")
                continue

            scaler = StandardScaler()
            scaled = scaler.fit_transform(wide)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled)

            pca_df = pd.DataFrame(
                pca_result,
                columns=["PC1", "PC2"],
                index=wide.index
            ).reset_index().rename(columns={sample_col: "Sample"})

            meta = (
                sub[[sample_col, "Delivery", "Dox_Treatment"]]
                .drop_duplicates(subset=[sample_col])
                .set_index(sample_col)
                .reindex(pca_df["Sample"])
            )

            if meta["Delivery"].isna().any() or meta["Dox_Treatment"].isna().any():
                for idx, row in meta.iterrows():
                    if pd.isna(row["Delivery"]):
                        for delivery in hue_order:
                            if str(delivery) in str(idx):
                                meta.at[idx, "Delivery"] = str(delivery)
                                break
                    if pd.isna(row["Dox_Treatment"]):
                        if "NoDox" in str(idx) or "No_Dox" in str(idx):
                            meta.at[idx, "Dox_Treatment"] = "No Dox"
                        elif "Dox" in str(idx):
                            meta.at[idx, "Dox_Treatment"] = "Dox"

            meta["Delivery"] = meta["Delivery"].fillna("UnknownPlatform").astype(str)
            meta["Dox_Treatment"] = meta["Dox_Treatment"].fillna("UnknownCondition").astype(str)
            meta["group"] = meta["Delivery"] + "_" + meta["Dox_Treatment"]

            pca_df = pca_df.join(meta[["Delivery", "Dox_Treatment", "group"]], on="Sample")

            plt.figure(figsize=(8, 6))
            ax = sns.scatterplot(
                data=pca_df,
                x="PC1", y="PC2",
                hue="group",
                palette=group_palette,
                s=150, edgecolor="black", alpha=1.0
            )
            sns.despine(top=True, right=True)
            plt.xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)",
                fontsize=18, fontweight='bold', color='black'
            )
            plt.ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)",
                fontsize=18, fontweight='bold', color='black'
            )
            plt.xticks(fontsize=16, fontweight='bold', color='black')
            plt.yticks(fontsize=16, fontweight='bold', color='black')

            leg = plt.legend(
                title="Group",
                prop={'weight': 'bold', 'size': 13},
                frameon=False,
                title_fontproperties={'weight': 'bold', 'size': 14}
            )
            if leg is not None:
                leg.set_draggable(True)

            plt.tick_params(axis='y', which='both', left=True, right=False)
            plt.tick_params(axis='x', which='both', bottom=True, top=False)

            plt.suptitle(f"{cell_type}{tp_label}", fontsize=26, fontweight='bold', color='black', y=1.05)

            safe_cell = re.sub(r'[^A-Za-z0-9._-]+', '_', str(cell_type))
            outpath = os.path.join(outdir, f"{safe_cell}{tp_suffix}_pca.png")

            plt.tight_layout()
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {outpath}")
            plt.close()


def _trajectory_panel(ax, sub, utr_col, metric_col, x_ticks, x_labels, color, ylabel):
    """Draw individual UTR lines (transparent) + bold mean line onto ax."""
    for utr in sub[utr_col].unique():
        utr_sub = sub[sub[utr_col] == utr].sort_values("_x")
        if utr_sub[metric_col].isna().all():
            continue
        ax.plot(utr_sub["_x"], utr_sub[metric_col],
                color=color, alpha=0.12, linewidth=0.8)

    mean_line = (
        sub.groupby("_x")[metric_col].mean()
        .reset_index().sort_values("_x")
    )
    ax.plot(mean_line["_x"], mean_line[metric_col],
            color=color, alpha=1.0, linewidth=2.5, label="Mean")

    ax.set_xlabel("Time (hrs)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.legend(
        handles=[matplotlib.lines.Line2D([0], [0], color=color, linewidth=2.5)],
        labels=["Mean"], frameon=False, fontsize=11,
    )
    sns.despine(ax=ax, top=True, right=True)


def plot_epr_trajectories(welch_results, utr_col, outdir, palette, hue_order):
    """
    Per-construct EPR trajectory plots over time.
    One PNG per (Delivery × Cell_Type): 1×2 panels [+Dox | −Dox].
    Individual UTRs in transparent color; bold line = mean.
    """
    print("\n=== Generating per-construct EPR trajectory plots ===")

    result = welch_results.copy()
    if "Timepoint" not in result.columns or result["Timepoint"].isna().all():
        print("  No Timepoint data — skipping.")
        return

    result = result[result["Timepoint"].notna()].copy()
    result["Cell_Type"] = result["Cell_Type"].astype(str)
    result["Delivery"]  = result["Delivery"].astype(str)

    timepoints = sorted(result["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
    tp_num = {tp: _timepoint_sort_key(tp) for tp in timepoints}
    result["_x"] = result["Timepoint"].map(tp_num)
    x_ticks  = sorted(tp_num.values())
    x_labels = [str(v) for v in x_ticks]

    sns.set_theme(context="talk", style="white")

    for delivery in hue_order:
        delivery = str(delivery)
        color = palette.get(delivery, "#999999")
        for cell_type in safe_sorted_unique_str(result["Cell_Type"]):
            sub = result[
                (result["Delivery"] == delivery) &
                (result["Cell_Type"] == cell_type)
            ].copy()
            if sub.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            fig.suptitle(f"Per-construct EPR trajectories — {delivery} | {cell_type}",
                         fontsize=16, fontweight="bold")

            for ax, (col, title) in zip(axes, [("mean_dox", "+Dox"), ("mean_nodox", "−Dox")]):
                ax.set_title(title, fontsize=14, fontweight="bold")
                _trajectory_panel(ax, sub, utr_col, col, x_ticks, x_labels, color, "EPR")

            axes[1].set_ylabel("")
            plt.tight_layout()
            safe_del  = re.sub(r"[^A-Za-z0-9._-]+", "_", delivery)
            safe_cell = re.sub(r"[^A-Za-z0-9._-]+", "_", cell_type)
            outpath = os.path.join(outdir, f"epr_trajectories_{safe_del}_{safe_cell}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {outpath}")
            plt.close()


def plot_ra_trajectories(welch_results, utr_col, outdir, palette, hue_order):
    """
    Per-construct RA trajectory plots over time (RPKM ÷ earliest timepoint).
    One PNG per (Delivery × Cell_Type): 1×2 panels [+Dox | −Dox].
    Individual UTRs in transparent color; bold line = mean.
    Skipped if RA_Dox / RA_NoDox columns are absent.
    """
    print("\n=== Generating per-construct RA trajectory plots ===")

    result = welch_results.copy()
    if "Timepoint" not in result.columns or result["Timepoint"].isna().all():
        print("  No Timepoint data — skipping.")
        return
    if "RA_Dox" not in result.columns or "RA_NoDox" not in result.columns:
        print("  RA_Dox / RA_NoDox columns not found — skipping.")
        return

    result = result[result["Timepoint"].notna()].copy()
    result["Cell_Type"] = result["Cell_Type"].astype(str)
    result["Delivery"]  = result["Delivery"].astype(str)

    timepoints = sorted(result["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
    earliest = timepoints[0]
    tp_num = {tp: _timepoint_sort_key(tp) for tp in timepoints}
    result["_x"] = result["Timepoint"].map(tp_num)
    x_ticks  = sorted(tp_num.values())
    x_labels = [str(v) for v in x_ticks]
    ylabel = f"RA (RPKM ÷ {earliest})"

    sns.set_theme(context="talk", style="white")

    for delivery in hue_order:
        delivery = str(delivery)
        color = palette.get(delivery, "#999999")
        for cell_type in safe_sorted_unique_str(result["Cell_Type"]):
            sub = result[
                (result["Delivery"] == delivery) &
                (result["Cell_Type"] == cell_type)
            ].copy()
            if sub.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            fig.suptitle(f"Per-construct RA trajectories — {delivery} | {cell_type}",
                         fontsize=16, fontweight="bold")

            for ax, (col, title) in zip(axes, [("RA_Dox", "+Dox"), ("RA_NoDox", "−Dox")]):
                ax.set_title(title, fontsize=14, fontweight="bold")
                _trajectory_panel(ax, sub, utr_col, col, x_ticks, x_labels, color, ylabel)
                ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)

            axes[1].set_ylabel("")
            plt.tight_layout()
            safe_del  = re.sub(r"[^A-Za-z0-9._-]+", "_", delivery)
            safe_cell = re.sub(r"[^A-Za-z0-9._-]+", "_", cell_type)
            outpath = os.path.join(outdir, f"ra_trajectories_{safe_del}_{safe_cell}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {outpath}")
            plt.close()


def _timepoint_sort_key(tp):
    """Extract leading integer from a timepoint string for numeric sort (e.g. '24h' → 24)."""
    import re as _re
    m = _re.match(r"(\d+)", str(tp))
    return int(m.group(1)) if m else 0


def plot_timecourse_stability(raw_data, sample_col, utr_col, reads_col, sample_regex, outdir,
                              palette, hue_order):
    """
    Timecourse stability plot: per-cell-type line plot of mean normalized relative abundance
    over time (relative to the earliest timepoint), averaged across all UTRs.

    Relative abundance is computed from ALL UTRs per sample (regardless of EPR filter),
    then normalized so the earliest timepoint equals 1.0.
    """
    print("\n=== Generating timecourse stability plot ===")

    df = raw_data.copy()

    required_cols = ['Delivery', 'Cell_Type', 'Dox_Treatment', 'Timepoint', 'Replicate']
    if not all(c in df.columns for c in required_cols):
        print("Parsing Sample_Stem...")
        df = parse_sample_stem(df, sample_col, sample_regex)

    if "Timepoint" not in df.columns or df["Timepoint"].isna().all():
        print("  No Timepoint data found — skipping.")
        return

    df = df[df["Timepoint"].notna()].copy()
    df["Cell_Type"] = df["Cell_Type"].astype(str)
    df["Delivery"]  = df["Delivery"].astype(str)
    df["Dox_Treatment"] = normalize_dox_treatment(df["Dox_Treatment"])

    # Relative abundance uses ALL UTRs as the denominator (pre-filter, per sample)
    df["LibSize"] = df.groupby(sample_col)[reads_col].transform("sum")
    df["Relative_Abundance"] = df[reads_col] / df["LibSize"].replace(0, np.nan)

    timepoints = sorted(df["Timepoint"].dropna().unique(), key=_timepoint_sort_key)
    if not timepoints:
        return
    earliest = timepoints[0]
    print(f"  Timepoints: {timepoints}  |  baseline = {earliest}")

    sns.set_theme(context="talk", style="white")

    for cell_type in safe_sorted_unique_str(df["Cell_Type"]):
        sub = df[df["Cell_Type"] == cell_type].copy()

        # Average relative abundance across replicates per (Delivery, Dox, Timepoint, UTR)
        avg_per_utr = (
            sub.groupby(["Delivery", "Dox_Treatment", "Timepoint", utr_col], as_index=False)
               ["Relative_Abundance"].mean()
        )

        # Normalize each UTR's trajectory to its value at the earliest timepoint
        baseline = (
            avg_per_utr[avg_per_utr["Timepoint"] == earliest]
            .set_index(["Delivery", "Dox_Treatment", utr_col])["Relative_Abundance"]
            .rename("Baseline_RA")
        )
        avg_per_utr = avg_per_utr.join(baseline, on=["Delivery", "Dox_Treatment", utr_col])
        avg_per_utr["Norm_RA"] = (
            avg_per_utr["Relative_Abundance"] / avg_per_utr["Baseline_RA"].replace(0, np.nan)
        )
        avg_per_utr = avg_per_utr.dropna(subset=["Norm_RA"])

        if avg_per_utr.empty:
            print(f"  Skipping {cell_type}: no data after normalization.")
            continue

        # Pool across all UTRs → mean ± std per (Delivery, Dox_Treatment, Timepoint)
        pool = (
            avg_per_utr.groupby(["Delivery", "Dox_Treatment", "Timepoint"])["Norm_RA"]
            .agg(["mean", "std"])
            .reset_index()
        )
        pool["_tp_num"] = pool["Timepoint"].apply(_timepoint_sort_key)
        pool = pool.sort_values("_tp_num")

        fig, ax = plt.subplots(figsize=(10, 6))

        for delivery in hue_order:
            delivery = str(delivery)
            color = palette.get(delivery, "#999999")
            for dox_treat in ["Dox", "No Dox"]:
                mask = (pool["Delivery"] == delivery) & (pool["Dox_Treatment"] == dox_treat)
                grp = pool[mask]
                if grp.empty:
                    continue
                linestyle = "-" if dox_treat == "Dox" else "--"
                ax.errorbar(
                    grp["Timepoint"], grp["mean"],
                    yerr=grp["std"].fillna(0),
                    label=f"{delivery} {dox_treat}",
                    color=color, linestyle=linestyle,
                    marker="o", linewidth=2, capsize=4,
                )

        ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(f"Library Stability over Time — {cell_type}", fontsize=18, fontweight="bold")
        ax.set_xlabel(f"Timepoint (normalized to {earliest})", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"Norm. Relative Abundance (÷ {earliest})", fontsize=14, fontweight="bold")
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(frameon=False, fontsize=12)
        sns.despine(ax=ax, top=True, right=True)
        plt.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cell_type))
        outpath = os.path.join(outdir, f"timecourse_stability_{safe_name}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {outpath}")
        plt.close()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Set up color palette
    palette = {args.delivery1: args.color1, args.delivery2: args.color2}
    hue_order = [args.delivery1, args.delivery2]

    # Load data
    print(f"Loading Welch test results: {args.welch_results}")
    welch_results = pd.read_csv(args.welch_results)
    print(f"  Loaded {len(welch_results)} rows")

    print(f"\nLoading raw data: {args.raw_data}")
    raw_data = pd.read_csv(args.raw_data)
    print(f"  Loaded {len(raw_data)} rows")

    # Validate columns
    welch_required = ['Delivery', 'Cell_Type', 'mean_nodox', 'mean_dox', 'log2fc', 'fdr']
    missing = set(welch_required) - set(welch_results.columns)
    if missing:
        raise ValueError(f"Welch results missing columns: {sorted(missing)}")

    raw_required = {args.sample_col, args.reads_col}
    missing = raw_required - set(raw_data.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {sorted(missing)}")

    # Normalize common types EARLY (prevents mixed int/str sorting bugs)
    welch_results["Cell_Type"] = welch_results["Cell_Type"].astype(str)
    welch_results["Delivery"] = welch_results["Delivery"].astype(str)
    if "Cell_Type" in raw_data.columns:
        raw_data["Cell_Type"] = raw_data["Cell_Type"].astype(str)
    if "Delivery" in raw_data.columns:
        raw_data["Delivery"] = raw_data["Delivery"].astype(str)

    # Generate plots
    plot_epr_per_library(
        welch_results=welch_results,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_reads_per_contig(
        raw_data=raw_data,
        sample_col=args.sample_col,
        reads_col=args.reads_col,
        sample_regex=args.sample_regex,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_total_reads_per_sample(
        raw_data=raw_data,
        sample_col=args.sample_col,
        reads_col=args.reads_col,
        sample_regex=args.sample_regex,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_cdf_pool_composition(
        raw_data=raw_data,
        sample_col=args.sample_col,
        utr_col=args.utr_col,
        reads_col=args.reads_col,
        sample_regex=args.sample_regex,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_volcano(
        welch_results=welch_results,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_spearman_correlation(
        raw_data=raw_data,
        sample_col=args.sample_col,
        utr_col=args.utr_col,
        epr_col=args.epr_col,
        sample_regex=args.sample_regex,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_pca(
        raw_data=raw_data,
        sample_col=args.sample_col,
        utr_col=args.utr_col,
        epr_col=args.epr_col,
        sample_regex=args.sample_regex,
        outdir=args.outdir,
        hue_order=hue_order
    )

    plot_timecourse_stability(
        raw_data=raw_data,
        sample_col=args.sample_col,
        utr_col=args.utr_col,
        reads_col=args.reads_col,
        sample_regex=args.sample_regex,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order
    )

    plot_epr_trajectories(
        welch_results=welch_results,
        utr_col=args.utr_col,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order,
    )

    plot_ra_trajectories(
        welch_results=welch_results,
        utr_col=args.utr_col,
        outdir=args.outdir,
        palette=palette,
        hue_order=hue_order,
    )

    print("\n=== QC plots complete ===")


if __name__ == "__main__":
    main()