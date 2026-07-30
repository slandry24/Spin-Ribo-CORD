#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import uuid
import pysam
import pandas as pd


# --------------------------- Counting (MD-based) -----------------------------

def c2t_mismatch_sites_cds(read: pysam.AlignedSegment, min_bq: int) -> list[int]:
    """
    Return the reference positions (0-based) of C→T mismatches for a single CDS read.
    Requires MD tag so get_aligned_pairs(with_seq=True) has ref bases.
    Applies per-base quality filter >= min_bq. Ignores indels/ref-skips.
    """
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return []

    qseq = read.query_sequence
    quals = read.query_qualities
    if qseq is None or quals is None:
        return []

    sites: list[int] = []
    for qpos, rpos, rbase in read.get_aligned_pairs(with_seq=True, matches_only=False):
        if qpos is None or rpos is None:
            continue
        if rbase is None:
            continue
        if quals[qpos] < min_bq:
            continue

        qb = qseq[qpos].upper()
        rb = rbase.upper()

        if rb == "C" and qb == "T":
            sites.append(rpos)

    return sites


# --------------------------- Pairing helpers ---------------------------------

def pair_bams_by_regex(utr_dir: Path, cds_dir: Path, glob_pat: str = "*.bam"):
    """
    Pair UTR + CDS BAMs by sample key.

    IMPORTANT: filenames contain BOTH:
      - replicate tokens like "_R1", "_R2", "_R3"   (underscore)
      - read tokens like ".R1.", ".R2."            (dot)

    We key off the DOT read token (".R1." / ".R2.") so we don't collapse replicates.

    Example pair:
      UTR: Lenti_HEK_Dox_R1.R1.Aligned.sortedByCoord.out.bam
      CDS: Lenti_HEK_Dox_R1.R2.cds.Aligned.sortedByCoord.out.md.bam
      key = "Lenti_HEK_Dox_R1"
    """
    utr_files = list(Path(utr_dir).glob(glob_pat))
    cds_files = list(Path(cds_dir).glob(glob_pat))

    dot_read_re = re.compile(r"^(?P<key>.+?)\.(?:R1|R2)\.")

    def sample_key(stem: str) -> str | None:
        m = dot_read_re.match(stem)
        return m.group("key") if m else None

    utr_map: dict[str, Path] = {}
    bad_utr = []
    for f in utr_files:
        key = sample_key(f.stem)
        if key:
            utr_map[key] = f
        else:
            bad_utr.append(f.name)

    cds_map: dict[str, Path] = {}
    bad_cds = []
    for f in cds_files:
        key = sample_key(f.stem)
        if key:
            cds_map[key] = f
        else:
            bad_cds.append(f.name)

    pairs = []
    unmatched_utr, unmatched_cds = [], []

    for key, f1 in utr_map.items():
        f2 = cds_map.get(key)
        if f2 is not None:
            pairs.append((key, f1, f2))
        else:
            unmatched_utr.append(f1.name)

    for key, f2 in cds_map.items():
        if key not in utr_map:
            unmatched_cds.append(f2.name)

    # Helpful debugging if something still goes wrong
    if not pairs:
        print("[DEBUG] No pairs. Example UTR filenames (up to 10):",
              [f.name for f in utr_files[:10]], file=sys.stderr)
        print("[DEBUG] No pairs. Example CDS filenames (up to 10):",
              [f.name for f in cds_files[:10]], file=sys.stderr)
        if bad_utr:
            print("[DEBUG] UTR files that did not match key regex (up to 10):",
                  bad_utr[:10], file=sys.stderr)
        if bad_cds:
            print("[DEBUG] CDS files that did not match key regex (up to 10):",
                  bad_cds[:10], file=sys.stderr)

    return pairs, unmatched_utr, unmatched_cds


# --------------------------- Worker per sample -------------------------------

def process_sample_worker(stem: str, bam_utr: str, bam_cds: str,
                          min_bq: int, mapq_utr: int, mapq_cds: int,
                          tmp_dir: str):
    """
    Worker does NOT write per-sample final outputs.

    It writes a TEMP per-sample detail CSV (edited reads only) into tmp_dir,
    and returns:
      - temp_detail_csv_path
      - per-sample per-UTR summary rows (with Total_Reads including zero-edit reads)
      - matched_pairs count
    """
    tmp_dir_p = Path(tmp_dir)
    tmp_dir_p.mkdir(parents=True, exist_ok=True)

    # ------- Pass 1: scan CDS (KEEP all reads, even 0-edit) -------
    cds_pass_ids = set()
    cds_edits = {}              # read_id -> Counter(pos0) (only if edits exist)
    cds_ref_for_read = {}       # read_id -> CDS reference

    with pysam.AlignmentFile(bam_cds, "rb") as cdsbam:
        for read in cdsbam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq_cds:
                continue

            rid = read.query_name
            ref = read.reference_name or "UNMAPPED"
            cds_pass_ids.add(rid)
            cds_ref_for_read[rid] = ref

            sites = c2t_mismatch_sites_cds(read, min_bq)
            if sites:
                cds_edits[rid] = Counter(sites)

    # ------- Pass 2: scan UTR — record matches and accumulate summaries -------
    rows = []
    matched_pairs = 0
    matched_utr_for_read = {}   # read_id -> utr_ref (only matched pairs)

    total_reads_by_utr = Counter()
    sum_edits_by_utr = Counter()

    with pysam.AlignmentFile(bam_utr, "rb") as utrbam:
        for read in utrbam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq_utr:
                continue

            rid = read.query_name
            utr_ref = read.reference_name or "UNMAPPED"

            if rid not in cds_pass_ids:
                continue

            matched_pairs += 1
            total_reads_by_utr[utr_ref] += 1
            matched_utr_for_read[rid] = utr_ref

            pos_counts = cds_edits.get(rid)
            if pos_counts:
                sum_edits_by_utr[utr_ref] += sum(pos_counts.values())

    # ------- Pass 3: rescan CDS — build per-UTR coverage (only matched reads) -------
    # Coverage at a position only counts reads whose paired UTR read mapped to
    # the same UTR_Reference, not all reads at that position.
    utr_cds_coverage = {}       # utr_ref -> Counter(pos0)

    with pysam.AlignmentFile(bam_cds, "rb") as cdsbam:
        for read in cdsbam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq_cds:
                continue

            rid = read.query_name
            utr_ref = matched_utr_for_read.get(rid)
            if utr_ref is None:
                continue

            if utr_ref not in utr_cds_coverage:
                utr_cds_coverage[utr_ref] = Counter()
            for pos0 in read.get_reference_positions():
                utr_cds_coverage[utr_ref][pos0] += 1

    # ------- Build detail rows using per-UTR coverage -------
    for rid, utr_ref in matched_utr_for_read.items():
        pos_counts = cds_edits.get(rid)
        if not pos_counts:
            continue

        cds_ref = cds_ref_for_read.get(rid, "UNMAPPED")
        ref_cov = utr_cds_coverage.get(utr_ref, {})
        for pos0, count in pos_counts.items():
            rows.append({
                "Sample_Stem": stem,
                "Read_ID": rid,
                "UTR_Reference": utr_ref,
                "CDS_Reference": cds_ref,
                "Ref_Pos_0based": pos0,
                "Ref_Pos_1based": pos0 + 1,
                "C2T_Count": count,
                "Position_Coverage": ref_cov.get(pos0, 0),
            })

    # ------- TEMP detail CSV (edited reads only) -------
    df_detail = pd.DataFrame(rows)
    if not df_detail.empty:
        df_detail = df_detail.sort_values(
            ["Sample_Stem", "UTR_Reference", "Read_ID", "Ref_Pos_0based"]
        ).reset_index(drop=True)

    tmp_detail_csv = tmp_dir_p / f"TMP_{stem}_{uuid.uuid4().hex}_detail.csv"
    df_detail.to_csv(tmp_detail_csv, index=False)

    # ------- Per-UTR summary rows (ALL reads) returned to main -------
    summary_rows = []
    for utr_ref, total_reads in total_reads_by_utr.items():
        sum_edits = int(sum_edits_by_utr.get(utr_ref, 0))
        summary_rows.append({
            "Sample_Stem": stem,
            "UTR_Reference": utr_ref,
            "Total_Reads": int(total_reads),
            "Sum_Edits": sum_edits,
            # EPR computed in main after combining (safe either way)
        })

    return str(tmp_detail_csv), summary_rows, matched_pairs


# --------------------------- Index helpers -----------------------------------

def ensure_bam_indexes(bam_paths):
    for b in bam_paths:
        b = Path(b)
        if not Path(str(b) + ".bai").exists():
            print(f"[INFO] Indexing BAM: {b}")
            pysam.index(str(b))


# --------------------------- CLI / main --------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pair UTR/CDS BAMs by sample key, count CDS C→T edits (MD tag required), "
            "retain zero-edit reads for denominators, and write ONLY:\n"
            "  1) ALL_SAMPLES_c2t_per_read_per_position.csv (edited reads only)\n"
            "  2) ALL_SAMPLES_c2t_per_utr_reads_edits.csv (per UTR: Total_Reads, Sum_Edits, EPR)\n"
        )
    )
    p.add_argument("--utr-dir", required=True, type=Path)
    p.add_argument("--cds-dir", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--glob-pattern", default="*.bam")
    p.add_argument("--min-bq", type=int, default=15)
    p.add_argument("--mapq-utr", type=int, default=100)
    p.add_argument("--mapq-cds", type=int, default=100)
    p.add_argument("--cores", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    pairs, unutr, uncds = pair_bams_by_regex(
        args.utr_dir, args.cds_dir, args.glob_pattern
    )

    if not pairs:
        print("[ERROR] No UTR/CDS pairs found.", file=sys.stderr)
        sys.exit(1)

    if unutr:
        print("[WARN] Unmatched UTR BAMs (first 10):", unutr[:10], file=sys.stderr)
    if uncds:
        print("[WARN] Unmatched CDS BAMs (first 10):", uncds[:10], file=sys.stderr)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.outdir / "_tmp_details"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_bams = [str(b1) for _, b1, _ in pairs] + [str(b2) for _, _, b2 in pairs]
    ensure_bam_indexes(all_bams)

    tmp_detail_csvs: list[str] = []
    all_summary_rows: list[dict] = []

    print(f"[INFO] Processing {len(pairs)} sample(s) with cores={args.cores}")

    if args.cores == 1:
        # Run directly in the main process — avoids subprocess fork overhead and OOM risk
        for stem, b1, b2 in pairs:
            tmp_detail_csv, summary_rows, matched = process_sample_worker(
                stem, str(b1), str(b2),
                args.min_bq, args.mapq_utr, args.mapq_cds,
                str(tmp_dir),
            )
            tmp_detail_csvs.append(tmp_detail_csv)
            all_summary_rows.extend(summary_rows)
            print(f"[OK] {stem} (matched reads: {matched})")
    else:
        with ProcessPoolExecutor(max_workers=args.cores) as ex:
            futs = {
                ex.submit(
                    process_sample_worker,
                    stem, str(b1), str(b2),
                    args.min_bq, args.mapq_utr, args.mapq_cds,
                    str(tmp_dir),
                ): stem
                for stem, b1, b2 in pairs
            }

            for fut in as_completed(futs):
                stem = futs[fut]
                tmp_detail_csv, summary_rows, matched = fut.result()
                tmp_detail_csvs.append(tmp_detail_csv)
                all_summary_rows.extend(summary_rows)
                print(f"[OK] {stem} (matched reads: {matched})")

    # ---- Write ONLY combined detail ----
    combined_detail = args.outdir / "ALL_SAMPLES_c2t_per_read_per_position.csv"
    if tmp_detail_csvs:
        # Read and concat all temp detail csvs
        dfs = []
        for p in tmp_detail_csvs:
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if not df.empty:
                dfs.append(df)

        if dfs:
            df_detail_all = pd.concat(dfs, ignore_index=True)
        else:
            # Keep an empty file with correct headers if nothing edited anywhere
            df_detail_all = pd.DataFrame(columns=[
                "Sample_Stem", "Read_ID", "UTR_Reference", "CDS_Reference",
                "Ref_Pos_0based", "Ref_Pos_1based", "C2T_Count"
            ])

        df_detail_all.to_csv(combined_detail, index=False)
        print(f"[INFO] Wrote combined detail CSV: {combined_detail}")

    # ---- Create ALL_SAMPLES per-UTR summary (from per-sample totals) ----
    combined_utr = args.outdir / "ALL_SAMPLES_c2t_per_utr_reads_edits.csv"
    if all_summary_rows:
        df_utr = pd.DataFrame(all_summary_rows)

        # If (Sample_Stem, UTR_Reference) can appear multiple times for any reason,
        # enforce aggregation here (safe even if already unique)
        df_utr = (
            df_utr.groupby(["Sample_Stem", "UTR_Reference"], as_index=False)
                 .agg({"Total_Reads": "sum", "Sum_Edits": "sum"})
        )

        df_utr["EPR"] = df_utr["Sum_Edits"] / df_utr["Total_Reads"]
        df_utr = df_utr.sort_values(["Sample_Stem", "UTR_Reference"]).reset_index(drop=True)
        df_utr.to_csv(combined_utr, index=False)
        print(f"[INFO] Wrote combined per-UTR summary CSV: {combined_utr}")
    else:
        # Write an empty summary with headers if nothing matched
        pd.DataFrame(columns=["Sample_Stem", "UTR_Reference", "Total_Reads", "Sum_Edits", "EPR"])\
          .to_csv(combined_utr, index=False)
        print(f"[INFO] Wrote empty per-UTR summary CSV (no matches): {combined_utr}")

    # ---- Cleanup temp detail files ----
    removed = 0
    for p in tmp_detail_csvs:
        try:
            Path(p).unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass

    try:
        # remove tmp dir if empty
        tmp_dir.rmdir()
    except Exception:
        pass

    print(f"[INFO] Cleaned up {removed} temp detail file(s).")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()