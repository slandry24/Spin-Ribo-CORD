#!/usr/bin/env python3
"""
make_genome_fastas.py

Inputs:
  --utr_csv       : CSV with columns 'name' and 'sequence' (UTR library)
  --cds_seq       : CDS sequence as a STRING (A/C/G/T; whitespace allowed)
  --library_type  : 5UTR or 3UTR (controls concatenation order)

Outputs:
  --out_genome_fasta : FASTA with entries = (5UTR + CDS) OR (CDS + 3UTR)
  --out_cds_fasta    : FASTA with a single entry = CDS only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def wrap_fasta(seq: str, width: int = 60) -> str:
    seq = "".join(seq.split()).upper()
    if width <= 0:
        return seq
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def load_utr_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # allow case-insensitive column names
    cols_lower = {c.lower(): c for c in df.columns}
    if "name" not in cols_lower or "sequence" not in cols_lower:
        raise ValueError(
            f"UTR CSV must have columns 'name' and 'sequence'. Found: {df.columns.tolist()}"
        )

    df = df[[cols_lower["name"], cols_lower["sequence"]]].copy()
    df.columns = ["name", "sequence"]

    df["name"] = df["name"].astype(str)
    df["sequence"] = (
        df["sequence"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )

    df = df.dropna(subset=["name", "sequence"])
    df = df[(df["name"].str.len() > 0) & (df["sequence"].str.len() > 0)]
    return df


def sanitize_dna(seq: str) -> str:
    return "".join(seq.split()).upper()


def build_genome_seq(utr_seq: str, cds_seq: str, library_type: str) -> str:
    lt = library_type.upper()
    if lt == "5UTR":
        return utr_seq + cds_seq
    elif lt == "3UTR":
        return cds_seq + utr_seq
    else:
        raise ValueError(f"Invalid library_type: {library_type}. Use 5UTR or 3UTR.")


def write_genome_fasta(
    utr_df: pd.DataFrame,
    cds_seq: str,
    out_fasta: Path,
    wrap: int,
    library_type: str,
) -> None:
    with out_fasta.open("w") as f:
        for _, row in utr_df.iterrows():
            header = row["name"]
            genome = build_genome_seq(row["sequence"], cds_seq, library_type)
            f.write(f">{header}\n{wrap_fasta(genome, wrap)}\n")


def write_cds_fasta(cds_seq: str, header: str, out_fasta: Path, wrap: int) -> None:
    with out_fasta.open("w") as f:
        f.write(f">{header}\n{wrap_fasta(cds_seq, wrap)}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Make genome FASTA (UTR+CDS or CDS+UTR per entry) and CDS-only FASTA from a CDS string."
    )
    p.add_argument(
        "--utr_csv",
        required=True,
        help="Path to UTR library CSV (columns: name, sequence).",
    )
    p.add_argument(
        "--cds_seq",
        required=True,
        help="CDS sequence as a string (whitespace/newlines allowed).",
    )
    p.add_argument(
        "--library_type",
        choices=["5UTR", "3UTR", "5utr", "3utr"],
        default="5UTR",
        help="Library type controls concatenation: 5UTR => UTR+CDS, 3UTR => CDS+UTR (default: 5UTR).",
    )
    p.add_argument(
        "--out_genome_fasta",
        required=True,
        help="Output FASTA path for genome entries (UTR+CDS or CDS+UTR).",
    )
    p.add_argument(
        "--out_cds_fasta",
        required=True,
        help="Output FASTA path for CDS-only reference.",
    )
    p.add_argument(
        "--cds_header",
        default="cds",
        help="Header for CDS-only FASTA entry (default: cds).",
    )
    p.add_argument(
        "--wrap",
        type=int,
        default=60,
        help="FASTA line wrap width (default: 60). Use 0 to disable wrapping.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    utr_df = load_utr_csv(Path(args.utr_csv))
    cds_seq = sanitize_dna(args.cds_seq)

    if len(cds_seq) == 0:
        raise ValueError("CDS sequence is empty after removing whitespace.")

    out_genome = Path(args.out_genome_fasta)
    out_cds = Path(args.out_cds_fasta)
    out_genome.parent.mkdir(parents=True, exist_ok=True)
    out_cds.parent.mkdir(parents=True, exist_ok=True)

    write_genome_fasta(utr_df, cds_seq, out_genome, args.wrap, args.library_type)
    write_cds_fasta(cds_seq, args.cds_header, out_cds, args.wrap)

    lt = args.library_type.upper()
    order = "UTR+CDS" if lt == "5UTR" else "CDS+UTR"
    print(f"Wrote genome FASTA: {out_genome} ({len(utr_df)} records; order={order})")
    print(f"Wrote CDS FASTA:    {out_cds} (1 record, header={args.cds_header})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)