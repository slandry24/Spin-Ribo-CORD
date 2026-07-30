#!/usr/bin/env python3
"""
Interactive dashboard for UTR translation efficiency results.

Usage:
    python dashboard.py --results /path/to/results
    # Then open http://localhost:8050 in your browser
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path,
                   help="Path to pipeline results directory")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()

# ── Data loading ───────────────────────────────────────────────────────────────

STEM_RE = re.compile(
    r"^(?P<Delivery>[^_]+)_(?P<Cell_Type>[^_]+)_(?P<Dox_Treatment>Dox|NoDox)(?:_(?P<Timepoint>[^_]+))?_(?P<Replicate>R\d+)$"
)

def _tp_sort_key(tp):
    """Numeric sort key for timepoint strings (e.g. '24h' → 24)."""
    m = re.match(r"(\d+)", str(tp))
    return int(m.group(1)) if m else 0

def parse_stem(s):
    m = STEM_RE.match(s)
    if m:
        return (m.group("Delivery"), m.group("Cell_Type"), m.group("Dox_Treatment"),
                m.group("Replicate"), m.group("Timepoint"))
    return ("Unknown", "Unknown", "Unknown", "Unknown", None)

def load_data(results_dir: Path):
    summary_path = results_dir / "05_mapping_counting" / "ALL_SAMPLES_c2t_per_utr_reads_edits.csv"
    detail_path  = results_dir / "05_mapping_counting" / "ALL_SAMPLES_c2t_per_read_per_position.csv"
    welch_path   = results_dir / "06_analysis" / "welches_t_test_EPR.csv"
    cache_path   = results_dir / "05_mapping_counting" / "ALL_SAMPLES_position_agg.pkl"

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = pd.read_csv(summary_path)
    parsed = summary["Sample_Stem"].apply(parse_stem)
    summary[["Delivery", "Cell_Type", "Dox_Treatment", "Replicate", "Timepoint"]] = pd.DataFrame(
        parsed.tolist(), index=summary.index
    )

    # ── Welch results — pre-compute columns used in every callback ─────────────
    welch = pd.read_csv(welch_path, low_memory=False)
    dox_cols = [c for c in welch.columns if c.startswith("Dox_EPR_Rep")]
    welch["Mean_Dox_EPR"] = welch[dox_cols].mean(axis=1)
    welch["log2fc"] = pd.to_numeric(welch.get("log2fc"), errors="coerce")
    welch["fdr"]    = pd.to_numeric(welch.get("fdr"),    errors="coerce")
    FDR_FLOOR = 1e-10
    welch["neg_log10_fdr"] = (
        -np.log10(welch["fdr"].clip(lower=FDR_FLOOR))
    ).clip(upper=8)

    # ── Position detail — use parquet cache if available ──────────────────────
    if cache_path.exists():
        print("Loading position data from cache...", flush=True)
        detail = pd.read_pickle(cache_path)
    else:
        print("Loading position data (streaming — will cache for next run)...", flush=True)
        agg_chunks = []
        for chunk in pd.read_csv(detail_path, chunksize=500_000, low_memory=False):
            agg = (
                chunk.groupby(["Sample_Stem", "UTR_Reference", "Ref_Pos_1based"], as_index=False)
                .agg(C2T_Count=("C2T_Count", "sum"), Position_Coverage=("Position_Coverage", "max"))
            )
            agg_chunks.append(agg)

        detail = (
            pd.concat(agg_chunks, ignore_index=True)
            .groupby(["Sample_Stem", "UTR_Reference", "Ref_Pos_1based"], as_index=False)
            .agg(C2T_Count=("C2T_Count", "sum"), Position_Coverage=("Position_Coverage", "max"))
        )
        parsed2 = detail["Sample_Stem"].apply(parse_stem)
        detail[["Delivery", "Cell_Type", "Dox_Treatment", "Replicate", "Timepoint"]] = pd.DataFrame(
            parsed2.tolist(), index=detail.index
        )
        detail.to_pickle(cache_path)
        print(f"Cached to {cache_path}", flush=True)

    print(f"Position data ready: {len(detail):,} rows", flush=True)

    has_timecourse = summary["Timepoint"].notna().any()
    if has_timecourse:
        tps = sorted(summary["Timepoint"].dropna().unique().tolist(), key=_tp_sort_key)
        print(f"Timecourse detected — timepoints: {tps}", flush=True)

    return summary, detail, welch, has_timecourse

# ── App layout ─────────────────────────────────────────────────────────────────

DELIVERY_COLORS = {"Lenti": "#4747d1", "IVTMods": "#f13636"}
RANKING_TOP_N   = 500   # show only top-N UTRs in ranking bars (performance)
BG_SAMPLE_FRAC  = 0.25  # fraction of non-significant volcano points to render

def build_layout(summary, welch, has_timecourse=False):
    utrs       = sorted(summary["UTR_Reference"].unique())
    cell_lines = sorted(summary["Cell_Type"].unique())
    deliveries = sorted(summary["Delivery"].unique())

    loading_style = {"color": "#888", "fontSize": "13px", "padding": "4px 0"}

    # Only send first value on initial load; options updated server-side as user types
    first_utr = utrs[0] if utrs else None
    initial_options = [{"label": first_utr, "value": first_utr}] if first_utr else []

    return html.Div([
        html.H2("UTR Translation Efficiency Dashboard",
                style={"fontFamily": "Arial", "marginBottom": "8px"}),

        # ── Controls ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("UTR of Interest"),
                dcc.Dropdown(
                    id="utr-select",
                    options=initial_options,
                    value=first_utr,
                    clearable=False,
                    style={"width": "400px"},
                ),
            ], style={"marginRight": "30px"}),

            html.Div([
                html.Label("Cell Line(s)"),
                dcc.Dropdown(
                    id="cell-select",
                    options=[{"label": c, "value": c} for c in cell_lines],
                    value=cell_lines,
                    multi=True,
                    style={"width": "300px"},
                ),
            ], style={"marginRight": "30px"}),

            html.Div([
                html.Label("Delivery"),
                dcc.Dropdown(
                    id="delivery-select",
                    options=[{"label": d, "value": d} for d in deliveries],
                    value=deliveries,
                    multi=True,
                    style={"width": "250px"},
                ),
            ]),
        ], style={"display": "flex", "alignItems": "flex-end",
                  "marginBottom": "20px", "fontFamily": "Arial"}),

        # ── Plots (each wrapped in a Loading spinner) ──────────────────────────
        dcc.Loading(dcc.Graph(id="ranking-plot",   style={"height": "auto", "minHeight": "400px"}),
                    style=loading_style),
        dcc.Loading(dcc.Graph(id="volcano-plot",   style={"height": "500px"}),
                    style=loading_style),
        dcc.Loading(dcc.Graph(id="condition-plot", style={"height": "500px"}),
                    style=loading_style),
        dcc.Loading(dcc.Graph(id="position-plot",  style={"height": "400px"}),
                    style=loading_style),

        *([
            html.Hr(),
            html.H3("Timecourse — Library Stability",
                    style={"fontFamily": "Arial", "marginTop": "20px"}),
            html.P("Relative abundance per sample (reads / library depth), normalized to the "
                   "earliest timepoint. Computed from all UTRs regardless of EPR significance.",
                   style={"fontFamily": "Arial", "color": "#555", "fontSize": "13px"}),
            dcc.Loading(dcc.Graph(id="timecourse-plot", style={"height": "500px"}),
                        style=loading_style),
        ] if has_timecourse else []),

    ], style={"padding": "20px", "maxWidth": "1400px"})

# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_callbacks(app, summary, detail, welch, has_timecourse=False):
    # UTR list kept in Python memory — never sent to browser in full
    all_utrs = sorted(summary["UTR_Reference"].unique().tolist())

    # ── 0. UTR search — server-side filtering so browser never holds all options
    @app.callback(
        Output("utr-select", "options"),
        Input("utr-select", "search_value"),
        State("utr-select", "value"),
    )
    def filter_utr_options(search_value, current_value):
        if not search_value:
            matches = all_utrs[:150]
        else:
            q = search_value.lower()
            matches = [u for u in all_utrs if q in u.lower()][:150]

        # Always keep the currently selected UTR visible in the list
        if current_value and current_value not in matches:
            matches = [current_value] + matches

        return [{"label": u, "value": u} for u in matches]

    # ── 1. UTR Ranking ─────────────────────────────────────────────────────────
    @app.callback(
        Output("ranking-plot", "figure"),
        Input("utr-select", "value"),
        Input("cell-select", "value"),
        Input("delivery-select", "value"),
    )
    def update_ranking(selected_utr, selected_cells, selected_deliveries):
        filt = welch
        if selected_cells:
            filt = filt[filt["Cell_Type"].isin(selected_cells)]
        if selected_deliveries:
            filt = filt[filt["Delivery"].isin(selected_deliveries)]

        if filt.empty:
            return go.Figure().add_annotation(text="No data for selected filters", showarrow=False)

        deliveries = sorted(filt["Delivery"].unique())
        cell_types = sorted(filt["Cell_Type"].unique())
        n_rows, n_cols = len(deliveries), len(cell_types)

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{d} | {c}" for d in deliveries for c in cell_types],
            shared_yaxes=False,
        )

        for r, delivery in enumerate(deliveries, start=1):
            for c, cell_type in enumerate(cell_types, start=1):
                sub = filt[(filt["Delivery"] == delivery) & (filt["Cell_Type"] == cell_type)]
                if sub.empty:
                    continue

                sub = sub.sort_values("Mean_Dox_EPR", ascending=False)

                # Keep only top N for rendering speed; always include selected UTR
                top = sub.head(RANKING_TOP_N)
                if selected_utr not in top["UTR_Reference"].values:
                    sel_row_full = sub[sub["UTR_Reference"] == selected_utr]
                    top = pd.concat([top, sel_row_full])

                colors = ["#e63946" if u == selected_utr else "#adb5bd"
                          for u in top["UTR_Reference"]]

                fig.add_trace(go.Bar(
                    x=top["UTR_Reference"],
                    y=top["Mean_Dox_EPR"],
                    marker_color=colors,
                    showlegend=False,
                    hovertemplate="<b>%{x}</b><br>Mean Dox EPR: %{y:.4f}<extra></extra>",
                ), row=r, col=c)

                sel_row = top[top["UTR_Reference"] == selected_utr]
                if not sel_row.empty:
                    fig.add_trace(go.Scatter(
                        x=sel_row["UTR_Reference"],
                        y=sel_row["Mean_Dox_EPR"],
                        mode="markers",
                        marker=dict(color="black", size=12, symbol="diamond",
                                    line=dict(width=1, color="white")),
                        showlegend=False,
                        hovertemplate="<b>%{x}</b><br>Mean Dox EPR: %{y:.4f}<extra>selected</extra>",
                    ), row=r, col=c)

        fig.update_layout(
            title=f"UTR Ranking by Mean Dox EPR (top {RANKING_TOP_N} shown)",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"),
            height=400 * n_rows,
        )
        fig.update_xaxes(tickangle=-45, showticklabels=False)
        fig.update_yaxes(title_text="Mean Dox EPR")
        return fig

    # ── 2. Volcano ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("volcano-plot", "figure"),
        Input("utr-select", "value"),
        Input("cell-select", "value"),
        Input("delivery-select", "value"),
    )
    def update_volcano(selected_utr, selected_cells, selected_deliveries):
        filt = welch.dropna(subset=["log2fc", "fdr"])
        if selected_cells:
            filt = filt[filt["Cell_Type"].isin(selected_cells)]
        if selected_deliveries:
            filt = filt[filt["Delivery"].isin(selected_deliveries)]

        if filt.empty:
            return go.Figure().add_annotation(text="No volcano data available", showarrow=False)

        cell_types = sorted(filt["Cell_Type"].unique())
        deliveries = sorted(filt["Delivery"].unique())
        n_cols     = max(len(cell_types), 1)

        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=cell_types, shared_yaxes=True)

        for col_i, cell_type in enumerate(cell_types, start=1):
            sub = filt[filt["Cell_Type"] == cell_type]
            sel = sub[sub["UTR_Reference"] == selected_utr]
            bg  = sub[sub["UTR_Reference"] != selected_utr]

            # Downsample non-significant background points for speed
            sig_mask = (bg["log2fc"].abs() > 0.5) & (bg["fdr"] < 0.05)
            bg_sig   = bg[sig_mask]
            bg_nonsig = bg[~sig_mask].sample(frac=BG_SAMPLE_FRAC, random_state=0)
            bg = pd.concat([bg_sig, bg_nonsig])

            for d_i, delivery in enumerate(deliveries):
                d_bg  = bg[bg["Delivery"] == delivery]
                color = DELIVERY_COLORS.get(delivery, "#999999")
                fig.add_trace(go.Scatter(
                    x=d_bg["log2fc"],
                    y=d_bg["neg_log10_fdr"],
                    mode="markers",
                    name=delivery,
                    legendgroup=delivery,
                    showlegend=(col_i == 1),
                    marker=dict(color=color, size=4, opacity=0.45,
                                line=dict(width=0)),
                    hovertemplate=(
                        "<b>%{customdata}</b><br>"
                        "log2FC: %{x:.2f}<br>-log10(FDR): %{y:.2f}"
                        "<extra>" + delivery + "</extra>"
                    ),
                    customdata=d_bg["UTR_Reference"],
                ), row=1, col=col_i)

            # Selected UTR — always rendered on top as regular Scatter
            if not sel.empty:
                fig.add_trace(go.Scatter(
                    x=sel["log2fc"],
                    y=sel["neg_log10_fdr"],
                    mode="markers",
                    name=selected_utr,
                    legendgroup="selected",
                    showlegend=(col_i == 1),
                    marker=dict(color="black", size=14, symbol="diamond",
                                line=dict(width=1.5, color="white")),
                    customdata=sel["UTR_Reference"],
                    hovertemplate=(
                        "<b>%{customdata}</b><br>"
                        "log2FC: %{x:.2f}<br>-log10(FDR): %{y:.2f}"
                        "<extra>selected</extra>"
                    ),
                ), row=1, col=col_i)

            # Threshold lines via add_shape — reliable per-subplot axis reference
            xref = "x" if col_i == 1 else f"x{col_i}"
            yref = "y"  # shared y-axis
            line_style = dict(dash="dash", color="black", width=1.5)

            # FDR = 0.05 horizontal line
            fig.add_shape(type="line", xref=xref, yref=yref,
                          x0=-10, x1=10, y0=-np.log10(0.05), y1=-np.log10(0.05),
                          line=line_style)
            # log2FC = 0.5 vertical line
            fig.add_shape(type="line", xref=xref, yref=yref,
                          x0=0.5, x1=0.5, y0=0, y1=8,
                          line=line_style)

        fig.update_xaxes(title_text="log2FC (Dox / NoDox)", range=[-10, 10])
        fig.update_yaxes(title_text="-log10(FDR)", range=[0, 8], col=1)
        fig.update_layout(
            title="Volcano — EPR log2FC vs FDR",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"),
        )
        return fig

    # ── 3. Condition comparison ────────────────────────────────────────────────
    @app.callback(
        Output("condition-plot", "figure"),
        Input("utr-select", "value"),
        Input("cell-select", "value"),
        Input("delivery-select", "value"),
    )
    def update_conditions(selected_utr, selected_cells, selected_deliveries):
        filt = summary[summary["UTR_Reference"] == selected_utr]
        if selected_cells:
            filt = filt[filt["Cell_Type"].isin(selected_cells)]
        if selected_deliveries:
            filt = filt[filt["Delivery"].isin(selected_deliveries)]

        if filt.empty:
            return go.Figure().add_annotation(text="No data", showarrow=False)

        filt = filt.copy()
        filt["Condition"] = filt["Delivery"] + " | " + filt["Cell_Type"] + " | " + filt["Dox_Treatment"]

        grouped = (
            filt.groupby("Condition")
            .agg(
                Mean_Reads=("Total_Reads", "mean"), Std_Reads=("Total_Reads", "std"),
                Mean_EPR=("EPR", "mean"),           Std_EPR=("EPR", "std"),
                Mean_Edits=("Sum_Edits", "mean"),   Std_Edits=("Sum_Edits", "std"),
            )
            .reset_index()
        )

        dox_color, nodox_color = "#e63946", "#457b9d"
        colors = [dox_color if ("Dox" in c and "NoDox" not in c) else nodox_color
                  for c in grouped["Condition"]]

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Total Reads", "EPR", "Sum Edits"])
        for col_i, (mean_col, std_col) in enumerate([
            ("Mean_Reads", "Std_Reads"),
            ("Mean_EPR",   "Std_EPR"),
            ("Mean_Edits", "Std_Edits"),
        ], start=1):
            fig.add_trace(go.Bar(
                x=grouped["Condition"], y=grouped[mean_col],
                error_y=dict(type="data", array=grouped[std_col].fillna(0)),
                marker_color=colors, showlegend=False,
            ), row=1, col=col_i)

        fig.update_layout(
            title=f"{selected_utr} — Reads, EPR, Edits by Condition",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"),
        )
        fig.update_xaxes(tickangle=-40)
        return fig

    # ── 4. Position edit-fraction ──────────────────────────────────────────────
    @app.callback(
        Output("position-plot", "figure"),
        Input("utr-select", "value"),
        Input("cell-select", "value"),
        Input("delivery-select", "value"),
    )
    def update_positions(selected_utr, selected_cells, selected_deliveries):
        filt = detail[detail["UTR_Reference"] == selected_utr]
        if selected_cells:
            filt = filt[filt["Cell_Type"].isin(selected_cells)]
        if selected_deliveries:
            filt = filt[filt["Delivery"].isin(selected_deliveries)]

        if filt.empty:
            return go.Figure().add_annotation(
                text="No position data for this UTR", showarrow=False
            )

        filt = filt.copy()
        filt["Condition"] = filt["Delivery"] + " | " + filt["Cell_Type"] + " | " + filt["Dox_Treatment"]

        pos_grp = (
            filt.groupby(["Condition", "Ref_Pos_1based"])
            .agg(Total_C2T=("C2T_Count", "sum"), Coverage=("Position_Coverage", "max"))
            .reset_index()
        )
        pos_grp["Edit_Fraction"] = pos_grp["Total_C2T"] / pos_grp["Coverage"].replace(0, np.nan)

        fig = go.Figure()
        for cond, grp in pos_grp.groupby("Condition"):
            grp = grp.sort_values("Ref_Pos_1based")
            fig.add_trace(go.Scatter(
                x=grp["Ref_Pos_1based"], y=grp["Edit_Fraction"],
                mode="lines+markers", name=cond,
                line=dict(width=1.5), marker=dict(size=4),
            ))

        fig.update_layout(
            title=f"{selected_utr} — Edit Fraction by CDS Position",
            xaxis_title="CDS Position (1-based)",
            yaxis_title="Edit Fraction (C→T / Coverage)",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e9ecef")
        fig.update_yaxes(showgrid=True, gridcolor="#e9ecef")
        return fig

    # ── 5. Timecourse stability (only registered when timecourse data exists) ──
    if has_timecourse:
        @app.callback(
            Output("timecourse-plot", "figure"),
            Input("utr-select", "value"),
            Input("cell-select", "value"),
            Input("delivery-select", "value"),
        )
        def update_timecourse(selected_utr, selected_cells, selected_deliveries):
            filt = welch[welch["UTR_Reference"] == selected_utr].copy()
            if selected_cells:
                filt = filt[filt["Cell_Type"].isin(selected_cells)]
            if selected_deliveries:
                filt = filt[filt["Delivery"].isin(selected_deliveries)]

            filt = filt[filt["Timepoint"].notna()]
            if filt.empty or "RA_Dox" not in filt.columns:
                return go.Figure().add_annotation(
                    text="No timecourse data for this UTR / selection", showarrow=False
                )

            timepoints = sorted(filt["Timepoint"].unique(), key=_tp_sort_key)
            earliest   = timepoints[0]

            # RA_Dox / RA_NoDox already normalized to earliest timepoint in analysis.py
            long = pd.concat([
                filt[["Delivery", "Cell_Type", "Timepoint", "RA_Dox"]]
                    .rename(columns={"RA_Dox": "RA"})
                    .assign(Dox_Treatment="Dox"),
                filt[["Delivery", "Cell_Type", "Timepoint", "RA_NoDox"]]
                    .rename(columns={"RA_NoDox": "RA"})
                    .assign(Dox_Treatment="NoDox"),
            ], ignore_index=True)
            long = long.sort_values("Timepoint", key=lambda s: s.map(_tp_sort_key))

            fig = go.Figure()
            DASH_STYLE = {"Dox": "solid", "NoDox": "dash"}

            for (delivery, cell_type, dox), grp in long.groupby(
                    ["Delivery", "Cell_Type", "Dox_Treatment"]):
                color     = DELIVERY_COLORS.get(delivery, "#999999")
                dash_line = DASH_STYLE.get(dox, "dot")
                label     = f"{delivery} | {cell_type} | {dox}"

                fig.add_trace(go.Scatter(
                    x=grp["Timepoint"],
                    y=grp["RA"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, dash=dash_line, width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Timepoint: %{{x}}<br>"
                        f"RA (RPKM ÷ {earliest}): %{{y:.4f}}"
                        "<extra></extra>"
                    ),
                ))

            fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=timepoints,
                showgrid=True, gridcolor="#e9ecef",
            )
            fig.update_yaxes(showgrid=True, gridcolor="#e9ecef")
            fig.update_layout(
                title=(f"{selected_utr} — Library Stability "
                       f"(RPKM ÷ {earliest})"),
                xaxis_title="Timepoint",
                yaxis_title=f"Relative Abundance (RPKM ÷ {earliest})",
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Arial"),
            )
            return fig

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    summary, detail, welch, has_timecourse = load_data(args.results)

    app = dash.Dash(__name__)
    app.layout = build_layout(summary, welch, has_timecourse)
    register_callbacks(app, summary, detail, welch, has_timecourse)

    print(f"\nDashboard running at http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
