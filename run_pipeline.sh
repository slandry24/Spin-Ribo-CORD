#!/bin/bash
# ─── UTR Translation Efficiency Pipeline V5 — Launcher ───────────────────────
# Edit pipeline.conf, then run:  bash run_pipeline.sh
#                        or:    bash run_pipeline.sh --resume
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source user config (PIPELINE_CONF env var overrides default location)
source "${PIPELINE_CONF:-${SCRIPT_DIR}/pipeline.conf}"

mkdir -p "${OUTDIR}/logs"

nextflow -log "${OUTDIR}/logs/nextflow.log" run "${SCRIPT_DIR}/main.nf" \
    --sample_sheet  "${SAMPLE_SHEET}" \
    --utr_csv       "${UTR_CSV}" \
    --cds_seq       "${CDS_SEQ}" \
    --outdir        "${OUTDIR}" \
    --library_type  "${LIBRARY_TYPE}" \
    --cds_header    "${CDS_HEADER}" \
    --threads       "${THREADS:-4}" \
    --min_reads     "${MIN_READS:-0}" \
    --delivery1     "${DELIVERY1:-Lenti}" \
    --delivery2     "${DELIVERY2:-IVTMods}" \
    ${GENOME_FASTA:+--genome_fasta "${GENOME_FASTA}"} \
    ${CDS_FASTA:+--cds_fasta "${CDS_FASTA}"} \
    ${STAR_IDX_UTR_CDS:+--star_index_utrcds "${STAR_IDX_UTR_CDS}"} \
    ${STAR_IDX_CDS:+--star_index_cds "${STAR_IDX_CDS}"} \
    -w             "${OUTDIR}/work" \
    -with-report "${OUTDIR}/logs/report.html" \
    "${@}"
