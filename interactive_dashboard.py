#!/usr/bin/env python3
"""
Interactive Dash dashboard for UTR Translation Efficiency analysis.

Features
--------
• Analyze tab   — Slice/filter data in real time.
                  Volcano, EPR bar, reads, CDF plots (all interactive via Plotly).
                  Download filtered table as CSV.

Launch
------
    # Standalone server (open http://localhost:8050 in browser; SSH tunnel if remote):
    python interactive_dashboard.py --port 8050

    # Pre-load results at startup:
    python interactive_dashboard.py \\
        --welch   /path/to/06_analysis/welches_t_test_EPR.csv \\
        --raw     /path/to/05_mapping_counting/ALL_SAMPLES_c2t_per_utr_reads_edits.csv

Install requirements (once per env):
    pip install dash dash-bootstrap-components plotly pandas
"""

import argparse
import base64
import io
import os
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

# ─── Helpers ──────────────────────────────────────────────────────────────────

COLORS = {"Lenti": "#4747d1", "IVTMods": "#f13636"}


def safe_log10(series, floor=1e-300):
    return -np.log10(series.clip(lower=floor))


def normalize_dox(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"[Nn]o[\s_-]?[Dd]ox|NoDox|No_Dox", "No Dox", regex=True)
        .str.replace(r"^\+?[Dd]ox$", "Dox", regex=True)
        .str.replace("mean_nodox", "No Dox", regex=False)
        .str.replace("mean_dox", "Dox", regex=False)
    )


def parse_sample_stem(df, regex=None):
    if regex is None:
        regex = r"^(?P<Delivery>[^_]+)_(?P<Cell_Type>[^_]+)_(?P<Dox_Treatment>Dox|NoDox)_(?P<Replicate>R\d+)$"
    parsed = df["Sample_Stem"].astype(str).str.extract(regex)
    return pd.concat([df.reset_index(drop=True), parsed], axis=1)


def load_welch(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        df["Cell_Type"] = df["Cell_Type"].astype(str)
        df["Delivery"]  = df["Delivery"].astype(str)
        return df
    except Exception as e:
        print(f"[WARN] Could not load welch CSV: {e}")
        return None


def load_raw(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        required = {"Sample_Stem", "UTR_Reference", "Total_Reads", "EPR"}
        if not required.issubset(df.columns):
            print(f"[WARN] Raw CSV missing columns: {required - set(df.columns)}")
            return None
        df = parse_sample_stem(df)
        df["Dox_Treatment"] = normalize_dox(df["Dox_Treatment"])
        df["Cell_Type"] = df["Cell_Type"].astype(str)
        df["Delivery"]  = df["Delivery"].astype(str)
        return df
    except Exception as e:
        print(f"[WARN] Could not load raw CSV: {e}")
        return None




# ─── Plot builders ────────────────────────────────────────────────────────────

def build_volcano(welch_df, deliveries, cells, fdr_thresh, log2fc_thresh, utr_search):
    df = welch_df.copy()
    if deliveries:
        df = df[df["Delivery"].isin(deliveries)]
    if cells:
        df = df[df["Cell_Type"].isin(cells)]
    if utr_search:
        df = df[df["UTR_Reference"].str.contains(utr_search, case=False, na=False)]
    if df.empty:
        return go.Figure().update_layout(title="No data matching filters")

    df["neg_log10_fdr"] = safe_log10(df["fdr"])
    df["significant"]   = (df["log2fc"].abs() > log2fc_thresh) & (df["fdr"] < fdr_thresh)
    df["color_group"]   = df["Delivery"]

    fig = px.scatter(
        df, x="log2fc", y="neg_log10_fdr",
        color="color_group",
        color_discrete_map=COLORS,
        hover_data={"UTR_Reference": True, "Cell_Type": True,
                    "log2fc": ":.3f", "fdr": ":.2e",
                    "mean_dox": ":.4f", "mean_nodox": ":.4f"},
        opacity=0.6, template="plotly_white",
        labels={"neg_log10_fdr": "-log10(FDR)", "log2fc": "log2FC (Dox / No Dox)",
                "color_group": "Delivery"},
        title="Volcano — EPR Dox vs No Dox",
    )
    fig.add_hline(y=-np.log10(fdr_thresh), line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=log2fc_thresh,  line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=-log2fc_thresh, line_dash="dash", line_color="gray", line_width=1)
    fig.update_traces(marker_size=5)
    return fig


def build_epr_bar(welch_df, deliveries, cells):
    df = welch_df.copy()
    if deliveries:
        df = df[df["Delivery"].isin(deliveries)]
    if cells:
        df = df[df["Cell_Type"].isin(cells)]
    if df.empty:
        return go.Figure().update_layout(title="No data matching filters")

    long = pd.melt(
        df[["Delivery", "Cell_Type", "UTR_Reference", "mean_dox", "mean_nodox"]],
        id_vars=["Delivery", "Cell_Type", "UTR_Reference"],
        value_vars=["mean_dox", "mean_nodox"],
        var_name="Condition", value_name="Mean_EPR",
    )
    long["Condition"] = normalize_dox(long["Condition"])

    stats = long.groupby(["Delivery", "Cell_Type", "Condition"])["Mean_EPR"].agg(
        mean="mean", std="std"
    ).reset_index()

    fig = px.bar(
        stats, x="Cell_Type", y="mean", color="Delivery",
        barmode="group", facet_col="Condition",
        error_y="std", color_discrete_map=COLORS,
        template="plotly_white",
        labels={"mean": "Mean EPR", "Cell_Type": "Cell Type"},
        title="Mean EPR per Cell Type",
    )
    return fig


def build_reads_bar(raw_df, deliveries, cells, treatments):
    df = raw_df.copy()
    if deliveries:
        df = df[df["Delivery"].isin(deliveries)]
    if cells:
        df = df[df["Cell_Type"].isin(cells)]
    if treatments:
        df = df[df["Dox_Treatment"].isin(treatments)]
    if df.empty:
        return go.Figure().update_layout(title="No data matching filters")

    stats = df.groupby(["Delivery", "Cell_Type", "Dox_Treatment"])["Total_Reads"].agg(
        mean="mean", std="std"
    ).reset_index()

    fig = px.bar(
        stats, x="Cell_Type", y="mean", color="Delivery",
        barmode="group", facet_col="Dox_Treatment",
        error_y="std", color_discrete_map=COLORS,
        template="plotly_white",
        labels={"mean": "Average Reads/UTR"},
        title="Reads per UTR per Cell Type",
    )
    return fig


def build_cdf(raw_df, deliveries, cells, treatments):
    df = raw_df.copy()
    if deliveries:
        df = df[df["Delivery"].isin(deliveries)]
    if cells:
        df = df[df["Cell_Type"].isin(cells)]
    if treatments:
        df = df[df["Dox_Treatment"].isin(treatments)]
    if df.empty:
        return go.Figure().update_layout(title="No data matching filters")

    df["lib_total"] = df.groupby("Sample_Stem")["Total_Reads"].transform("sum")
    df["pct"] = df["Total_Reads"] / df["lib_total"] * 100

    avg = (
        df.groupby(["Delivery", "Cell_Type", "Dox_Treatment", "UTR_Reference"], as_index=False)
        ["pct"].mean()
    )

    traces = []
    for (d, c, t), sub in avg.groupby(["Delivery", "Cell_Type", "Dox_Treatment"]):
        sub = sub.sort_values("pct", ascending=False).reset_index(drop=True)
        cdf = np.cumsum(sub["pct"].values) / np.sum(sub["pct"].values)
        traces.append(go.Scatter(
            x=np.arange(1, len(sub) + 1), y=cdf,
            mode="lines",
            name=f"{d} {c} {t}",
            line=dict(color=COLORS.get(d, "#888888"),
                      dash="solid" if "No Dox" not in t else "dot"),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title="CDF of Pool Composition",
        xaxis_title="Member Rank",
        yaxis_title="Cumulative Fraction",
        template="plotly_white",
        yaxis_range=[0, 1],
    )
    return fig


def df_to_download(df: pd.DataFrame) -> str:
    return "data:text/csv;charset=utf-8," + df.to_csv(index=False)


# ─── App layout ───────────────────────────────────────────────────────────────

def build_app(init_welch_path: str | None, init_raw_path: str | None) -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )
    app.title = "UTR Translation Dashboard"

    # Pre-load initial data
    init_welch = load_welch(init_welch_path) if init_welch_path else None
    init_raw   = load_raw(init_raw_path)     if init_raw_path   else None

    # ── Sidebar filter panel ──────────────────────────────────────────────────
    sidebar = dbc.Card([
        html.H5("Filters", className="card-title mb-3"),

        html.Label("Delivery method", className="fw-bold"),
        dcc.Checklist(
            id="filter-delivery",
            options=[{"label": d, "value": d} for d in ["Lenti", "IVTMods"]],
            value=["Lenti", "IVTMods"],
            inline=True, className="mb-2"
        ),

        html.Label("Cell Type", className="fw-bold"),
        dcc.Dropdown(id="filter-cell-type", multi=True, placeholder="All cell types",
                     className="mb-2"),

        html.Label("Dox Treatment", className="fw-bold"),
        dcc.Checklist(
            id="filter-treatment",
            options=[{"label": "Dox", "value": "Dox"},
                     {"label": "No Dox", "value": "No Dox"}],
            value=["Dox", "No Dox"],
            inline=True, className="mb-2"
        ),

        html.Label("UTR search (regex)", className="fw-bold"),
        dbc.Input(id="filter-utr-search", placeholder="e.g. ACTB|GAPDH",
                  debounce=True, className="mb-3"),

        html.Hr(),

        html.Label("Min reads per UTR", className="fw-bold"),
        dcc.Slider(id="filter-min-reads", min=0, max=500, step=10, value=50,
                   marks={0: "0", 100: "100", 500: "500"},
                   tooltip={"placement": "bottom"}, className="mb-3"),

        html.Label("FDR threshold", className="fw-bold"),
        dcc.Slider(id="filter-fdr", min=0.001, max=0.2, step=0.001, value=0.05,
                   marks={0.05: "0.05", 0.1: "0.1", 0.2: "0.2"},
                   tooltip={"placement": "bottom"}, className="mb-3"),

        html.Label("log2FC threshold", className="fw-bold"),
        dcc.Slider(id="filter-log2fc", min=0, max=5, step=0.1, value=0.5,
                   marks={0: "0", 0.5: "0.5", 1: "1", 2: "2", 5: "5"},
                   tooltip={"placement": "bottom"}, className="mb-3"),

    ], body=True, className="h-100")

    # ── Analyze tab content ───────────────────────────────────────────────────
    analyze_tab = dbc.Tab(label="Analyze", tab_id="tab-analyze", children=[
        dbc.Row([
            dbc.Col(sidebar, width=2),
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        dbc.Alert(id="data-status-alert", color="info",
                                  children="No data loaded. Use --welch / --raw flags or upload files below.",
                                  className="mt-2"),
                        width=12,
                    ),
                ]),
                dbc.Tabs([
                    dbc.Tab(label="Volcano",    tab_id="plot-volcano"),
                    dbc.Tab(label="EPR Summary", tab_id="plot-epr"),
                    dbc.Tab(label="Reads",       tab_id="plot-reads"),
                    dbc.Tab(label="CDF",         tab_id="plot-cdf"),
                    dbc.Tab(label="Data Table",  tab_id="plot-table"),
                ], id="plot-tabs", active_tab="plot-volcano", className="mt-2"),
                html.Div(id="plot-content", className="mt-2"),
            ], width=10),
        ]),
    ])


    # ── Main layout ───────────────────────────────────────────────────────────
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("UTR Translation Efficiency Dashboard",
                            className="text-primary mt-3 mb-0"), width=12),
        ]),
        html.Hr(className="mt-1 mb-2"),

        dbc.Tabs(
            [analyze_tab],
            id="main-tabs", active_tab="tab-analyze"
        ),

        # Hidden stores for shared state
        dcc.Store(id="store-welch", data=init_welch.to_json(date_format="iso") if init_welch is not None else None),
        dcc.Store(id="store-raw",   data=init_raw.to_json(date_format="iso")   if init_raw   is not None else None),
    ], fluid=True)

    # ─── Callbacks ────────────────────────────────────────────────────────────

    # Populate cell-type dropdown from current raw store
    @app.callback(
        Output("filter-cell-type", "options"),
        Output("filter-cell-type", "value"),
        Input("store-raw", "data"),
    )
    def update_cell_options(raw_json):
        if raw_json is None:
            return [], []
        df = pd.read_json(io.StringIO(raw_json))
        cells = sorted(df["Cell_Type"].dropna().astype(str).unique().tolist())
        return [{"label": c, "value": c} for c in cells], cells

    # Data status alert
    @app.callback(
        Output("data-status-alert", "children"),
        Output("data-status-alert", "color"),
        Input("store-welch", "data"),
        Input("store-raw",   "data"),
    )
    def update_status(welch_json, raw_json):
        if welch_json and raw_json:
            w = pd.read_json(io.StringIO(welch_json))
            r = pd.read_json(io.StringIO(raw_json))
            return (f"Loaded: {len(w)} UTR tests | {r['Sample_Stem'].nunique()} samples | "
                    f"{r['UTR_Reference'].nunique()} UTRs"), "success"
        if welch_json or raw_json:
            return "Partial data loaded (need both welch + raw CSV).", "warning"
        return "No data loaded. Use --welch / --raw flags or upload files.", "info"

    # Main plot router
    @app.callback(
        Output("plot-content", "children"),
        Input("plot-tabs", "active_tab"),
        Input("store-welch",        "data"),
        Input("store-raw",          "data"),
        Input("filter-delivery",    "value"),
        Input("filter-cell-type",   "value"),
        Input("filter-treatment",   "value"),
        Input("filter-utr-search",  "value"),
        Input("filter-min-reads",   "value"),
        Input("filter-fdr",         "value"),
        Input("filter-log2fc",      "value"),
    )
    def render_plot(active_tab, welch_json, raw_json,
                    deliveries, cells, treatments,
                    utr_search, min_reads, fdr_thresh, log2fc_thresh):

        welch_df = pd.read_json(io.StringIO(welch_json)) if welch_json else None
        raw_df   = pd.read_json(io.StringIO(raw_json))   if raw_json   else None

        # Apply min-reads filter to raw
        if raw_df is not None and min_reads:
            raw_df = raw_df[raw_df["Total_Reads"] >= min_reads]

        no_data = html.Div("No data loaded or no rows pass filters.",
                           className="text-muted p-4")

        if active_tab == "plot-volcano":
            if welch_df is None:
                return no_data
            fig = build_volcano(welch_df, deliveries, cells,
                                fdr_thresh or 0.05, log2fc_thresh or 0.5, utr_search)
            return dcc.Graph(figure=fig, style={"height": "600px"})

        elif active_tab == "plot-epr":
            if welch_df is None:
                return no_data
            fig = build_epr_bar(welch_df, deliveries, cells)
            return dcc.Graph(figure=fig, style={"height": "500px"})

        elif active_tab == "plot-reads":
            if raw_df is None:
                return no_data
            fig = build_reads_bar(raw_df, deliveries, cells, treatments)
            return dcc.Graph(figure=fig, style={"height": "500px"})

        elif active_tab == "plot-cdf":
            if raw_df is None:
                return no_data
            fig = build_cdf(raw_df, deliveries, cells, treatments)
            return dcc.Graph(figure=fig, style={"height": "500px"})

        elif active_tab == "plot-table":
            src = welch_df if welch_df is not None else raw_df
            if src is None:
                return no_data
            df = src.copy()
            if deliveries and "Delivery" in df.columns:
                df = df[df["Delivery"].isin(deliveries)]
            if cells and "Cell_Type" in df.columns:
                df = df[df["Cell_Type"].isin(cells)]
            if utr_search:
                col = "UTR_Reference" if "UTR_Reference" in df.columns else df.columns[0]
                df = df[df[col].str.contains(utr_search, case=False, na=False)]
            df = df.round(6)
            dl_link = html.A(
                "Download filtered CSV",
                id="download-csv-link",
                download="filtered_results.csv",
                href=df_to_download(df),
                target="_blank",
                className="btn btn-sm btn-outline-primary mb-2",
            )
            tbl = dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                filter_action="native",
                sort_action="native",
                page_size=25,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 12},
            )
            return html.Div([dl_link, tbl])

        return no_data

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UTR Translation Efficiency — Interactive Dashboard")
    p.add_argument("--welch", default=None,
                   help="Pre-load Welch t-test results CSV (06_analysis/welches_t_test_EPR.csv)")
    p.add_argument("--raw",   default=None,
                   help="Pre-load raw counts CSV (05_mapping_counting/ALL_SAMPLES_c2t_per_utr_reads_edits.csv)")
    p.add_argument("--port",  type=int, default=8050)
    p.add_argument("--host",  default="0.0.0.0",
                   help="Bind host. Use 127.0.0.1 for local-only (then SSH tunnel).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n  Dashboard starting at http://localhost:{args.port}")
    if args.welch: print(f"  Welch:    {args.welch}")
    if args.raw:   print(f"  Raw data: {args.raw}")
    print("\n  If running on a remote server, SSH tunnel from your laptop:")
    print(f"    ssh -L {args.port}:localhost:{args.port} <user>@<remote_host>\n")

    app = build_app(args.welch, args.raw)
    app.run(host=args.host, port=args.port, debug=args.debug)
