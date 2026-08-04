# Pooled Ribo-CORD

A Nextflow pipeline for measuring cell-type-specific translational efficiency of 5'UTR (or 3'UTR) libraries using RiboSTAMP-derived C→T editing rates as a proxy for ribosome engagement.

## Overview

The pipeline aligns paired-end sequencing reads to a pooled UTR+CDS reference, counts cytosine-to-uracil (C→T) mismatches introduced by STAMP editing, and computes an Edits Per Read (EPR) metric. It then performs a Welch's t-test between Dox (editor-on) and NoDox (editor-off) replicates to identify UTRs with significantly elevated translational activity.

```
Raw FASTQs
    │
    ▼
Adapter trimming (cutadapt)
    │
    ▼
FastQC (per-sample read quality)
    │
    ▼
Alignment to UTR+CDS reference (STAR)
Alignment to CDS-only reference (STAR)
    │
    ▼
C→T mismatch counting per UTR per read
    │
    ▼
EPR summary per sample
    │
    ▼
Welch's t-test (Dox vs NoDox) + FDR correction
TPM / RPKM normalisation
    │
    ▼
QC plots + results CSV
    │
    ▼
MultiQC report (aggregated QC across all samples)
```

## Requirements

- [Nextflow](https://www.nextflow.io/) ≥ 23.10
- [Conda](https://docs.conda.io/) (miniconda or anaconda)
- A Linux or macOS machine to run it on (laptop, workstation, or any generic compute node — no cluster scheduler required)

Everything else (STAR, samtools, cutadapt, FastQC, MultiQC, and the full Python/analysis stack) is installed automatically from `environment.yml`.

## Setup

**First time only** — create the conda environment from the included `environment.yml`:

```bash
conda env create -f environment.yml
```

This installs Nextflow itself plus every tool the pipeline needs — `star`, `samtools`, `cutadapt`, `fastqc`, `multiqc`, and the Python stack (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `dash`, `pysam`, etc.).

Then activate it before running anything:
```bash
conda activate utr_pipeline
```

To update an existing environment after pulling changes:
```bash
conda env update -f environment.yml --prune
```

Nextflow itself also manages a per-process conda environment automatically (`conda.enabled = true` in `nextflow.config`, built from the same `environment.yml`) — no manual environment activation inside the pipeline's processes is needed.

## Quick Start

1. **Create the conda environment** (first time only — see Setup above).
2. **Edit `pipeline.conf`** — set paths to your sample sheet, UTR CSV, and output directory. Or keep separate config files per experiment (see [Using a Custom Config File](#using-a-custom-config-file) below).
3. **Run the pipeline:**
   ```bash
   bash run_pipeline.sh
   ```
4. **Resume a failed run:**
   ```bash
   bash run_pipeline.sh -resume
   ```

## Using a Custom Config File

By default `run_pipeline.sh` reads `pipeline.conf` from the pipeline directory. If you maintain separate config files per experiment (recommended for large multi-experiment projects), point to it with the `PIPELINE_CONF` environment variable:

```bash
PIPELINE_CONF=/path/to/Expt175/pipeline.conf bash run_pipeline.sh
```

Any arguments after `run_pipeline.sh` (such as `-resume`) are forwarded to Nextflow unchanged.

## Inputs

### 1. Sample Sheet (`SAMPLE_SHEET`)

A CSV file describing each sequencing library. **Required columns:**

| Column | Description |
|---|---|
| `Sample` | Human-readable sample label (used for naming outputs) |
| `Fastq_path_R1` | Absolute path to R1 FASTQ (gzipped) |
| `Fastq_path_R2` | Absolute path to R2 FASTQ (gzipped) |
| `Treatment` | `Dox` or `NoDox` (and variants: `No Dox`, `No-Dox`, `No_Dox`, `-Dox`) |
| `Read1_Read2` | `UTR_CDS` — R1 contains the UTR side; `CDS_UTR` — R2 contains the UTR side |
| `Replicate` | Integer replicate number, or `R1` / `R2` / `R3` format |
| `Timepoint` | *(Optional)* Time label for timecourse experiments (e.g. `3h`, `24h`, `48h`). If present, it is inserted into the sample ID and enables timecourse-specific QC plots. |

The `Sample` column is parsed into four fields using the pattern `{Delivery}_{CellType}_{Treatment}_{Replicate}` (e.g. `Lenti_HEK_Dox_R1` or `Mods_Jurkat_NoDox_R2`). For timecourse runs the pattern becomes `{Delivery}_{CellType}_{Treatment}_{Timepoint}_{Replicate}`. The `Delivery` field must match `DELIVERY1` or `DELIVERY2` in `pipeline.conf` — a mismatch causes QC plots to fail with a palette error. These fields drive statistical grouping.

See [examples/sample_sheet.csv](examples/sample_sheet.csv) for a template.

### 2. UTR Library CSV (`UTR_CSV`)

A CSV file with one row per UTR sequence to test. **Required columns:**

| Column | Description |
|---|---|
| `name` | Unique identifier for this UTR (e.g. `ENST00000005178.6_PDK4`) |
| `sequence` | DNA sequence of the UTR (A/C/G/T; case-insensitive; whitespace ignored) |

Column names are case-insensitive. The name becomes the reference sequence header in the STAR index and the key in all output files.

See [examples/utr_sequences.csv](examples/utr_sequences.csv) for a template.

### 3. CDS Sequence (`CDS_SEQ`)

The full nucleotide sequence of the reporter CDS (e.g. Citrine) provided as a string directly in `pipeline.conf`. Whitespace is stripped automatically. Must be in-frame.

See [examples/cds_sequence.txt](examples/cds_sequence.txt) for the default Citrine sequence.

### 4. Optional Pre-built References

If you have already run the pipeline once, set these in `pipeline.conf` to skip reference building and STAR indexing:

```bash
GENOME_FASTA="${OUTDIR}/reference/UTR_CDS.fasta"
CDS_FASTA="${OUTDIR}/reference/CDS.fasta"
STAR_IDX_UTR_CDS="${OUTDIR}/reference/star_index_utr_cds"
STAR_IDX_CDS="${OUTDIR}/reference/star_index_cds"
```

## Configuration (`pipeline.conf`)

| Parameter | Description | Default |
|---|---|---|
| `SAMPLE_SHEET` | Path to sample sheet CSV | — |
| `UTR_CSV` | Path to UTR library CSV | — |
| `CDS_SEQ` | Reporter CDS nucleotide sequence | Citrine |
| `OUTDIR` | Output directory (scratch recommended) | — |
| `LIBRARY_TYPE` | `5UTR` (UTR precedes CDS) or `3UTR` (CDS precedes UTR) | `5UTR` |
| `CDS_HEADER` | FASTA header name for the CDS entry | `citrine` |
| `MIN_READS` | Minimum reads per replicate to include a UTR in stats | `10` |
| `DELIVERY1` | First delivery method label — must match the first `_`-delimited field of your `Sample` names (e.g. `Lenti`, `Mods`) | `Lenti` |
| `DELIVERY2` | Second delivery method label — must match the first field of your `Sample` names (e.g. `IVTMods`, `noMods`) | `IVTMods` |
| `THREADS` | Threads passed to cutadapt/STAR — set to your machine's core count | `4` |

## Outputs

All outputs are written to `OUTDIR/`:

```
OUTDIR/
├── reference/
│   ├── UTR_CDS.fasta              # One entry per UTR: UTR+CDS concatenated
│   ├── CDS.fasta                  # Single-entry CDS reference
│   ├── star_index_utr_cds/        # STAR index for UTR+CDS
│   └── star_index_cds/            # STAR index for CDS only
├── 02_fastqc/                     # FastQC HTML reports + zip archives (per sample)
├── 03_star_align_utr_cds/         # BAMs aligned to UTR+CDS reference
├── 04_star_align_cds_md/          # BAMs aligned to CDS reference (MD-tagged)
├── 05_mapping_counting/
│   ├── ALL_SAMPLES_c2t_per_read_per_position.csv   # Per-read, per-position edit data
│   └── ALL_SAMPLES_c2t_per_utr_reads_edits.csv     # Per-UTR EPR summary
├── 06_analysis/
│   └── welches_t_test_EPR.csv     # Statistical results with TPM/RPKM
├── 07_qc_plots/                   # PNG quality-control figures (see below)
├── 08_multiqc/
│   ├── multiqc_report.html        # Aggregated QC report across all samples
│   └── multiqc_data/              # Raw data tables backing the MultiQC report
└── logs/
    ├── nextflow.log               # Nextflow master process log
    ├── report.html                # Nextflow execution report
    ├── pipeline_trace.txt
    └── pipeline_timeline.html
```

### Key Output: `welches_t_test_EPR.csv`

One row per (Delivery, Cell_Type, UTR) combination. Key columns:

| Column | Description |
|---|---|
| `Delivery` | Delivery method parsed from Sample_Stem |
| `Cell_Type` | Cell type parsed from Sample_Stem |
| `Timepoint` | Timepoint label (timecourse runs only) |
| `UTR_Reference` | UTR identifier from the input library |
| `n_dox` / `n_nodox` | Number of replicates passing `MIN_READS` filter |
| `Dox_EPR_Rep1/2/3` | EPR value per Dox replicate |
| `NoDox_EPR_Rep1/2/3` | EPR value per NoDox replicate |
| `mean_dox` / `mean_nodox` | Mean EPR across replicates |
| `log2fc` | log2(mean_dox / mean_nodox) with pseudocount |
| `t_stat` / `p_value` | Welch's t-test statistic and p-value |
| `fdr` | Benjamini-Hochberg FDR-corrected p-value |
| `Dox_TPM_Rep1/2/3` | TPM per Dox replicate |
| `NoDox_TPM_Rep1/2/3` | TPM per NoDox replicate |
| `RA_Dox` / `RA_NoDox` | Mean RPKM relative abundance (timecourse runs only) |

### Key Output: `ALL_SAMPLES_c2t_per_utr_reads_edits.csv`

Per-sample, per-UTR edit summary:

| Column | Description |
|---|---|
| `Sample_Stem` | Sample identifier (e.g. `Lenti_HEK_Dox_R1`) |
| `UTR_Reference` | UTR identifier |
| `Total_Reads` | Reads mapped to this UTR |
| `Sum_Edits` | Total C→T edits detected |
| `EPR` | Edits per read (Sum_Edits / Total_Reads) |

## QC Plots (`07_qc_plots/`)

All QC plots are 300 dpi PNGs. Plots that depend on timecourse data are generated only when a `Timepoint` column is present in the sample sheet.

### Standard plots (all runs)

| File | Description |
|---|---|
| `mean_epr_per_library_barplot.png` | Mean EPR across the full UTR library, split into +Dox and −Dox panels. Bars show the grand mean ± SD across replicates, grouped by cell type and colored by delivery method. Highlights whether the editor is active (+Dox) versus at baseline (−Dox). |
| `read_distribution_per_contig_barplot.png` | Average reads per individual UTR library member per sample, split into +Dox and −Dox panels. Low values indicate poor library representation; a flat distribution indicates even sampling. |
| `total_reads_barplot.png` | Total reads per sample (summed across all UTRs), split into +Dox and −Dox panels. Use this to flag samples with low sequencing depth before interpreting EPR values. |
| `CDF_Avg_byCondition_{CellType}.png` | Cumulative distribution of pool composition for each cell type. The x-axis ranks UTR members by descending abundance; the y-axis shows the cumulative fraction of the library. A steep early rise indicates that a small number of UTRs dominate the pool. One plot per cell type. |
| `{CellType}_volcano.png` | Volcano plot of log2(EPR Dox / EPR NoDox) versus −log10(FDR) for each cell type. Dashed lines mark the log2FC = ±0.5 and FDR = 0.05 thresholds. Points are colored by delivery method. The number of UTRs passing both thresholds is annotated in the top-left corner. One plot per cell type. |
| `spearman.png` | Clustered Spearman correlation heatmap across all samples (EPR-based). Samples are hierarchically clustered. Strong within-condition clustering and separation between Dox/NoDox groups indicates high data quality and reproducible editing. |
| `{CellType}_pca.png` | PCA of samples using per-UTR EPR values, one plot per cell type. Each point is a sample, colored by delivery method and Dox treatment. PC1/PC2 variance explained is shown on the axes. Expected clustering: replicates group together, Dox and NoDox separate. |

### Timecourse plots (runs with `Timepoint` column)

| File | Description |
|---|---|
| `timecourse_stability_{CellType}.png` | Library stability over time per cell type. Shows mean normalized relative abundance (RPKM divided by the earliest timepoint value) for each delivery method and Dox condition across all timepoints. A flat line near 1.0 indicates stable library representation; a declining line indicates dropout over time. |
| `epr_trajectories_{Delivery}_{CellType}.png` | Per-construct EPR trajectories over time for each delivery × cell-type combination, split into +Dox and −Dox panels. Thin transparent lines show individual UTR trajectories; the bold line shows the mean. Useful for identifying early versus late translational responses. |
| `ra_trajectories_{Delivery}_{CellType}.png` | Per-construct relative abundance (RPKM ÷ earliest timepoint) trajectories over time, split into +Dox and −Dox panels. A dashed gray line at 1.0 marks the baseline. Divergence from 1.0 indicates changes in library representation independent of editing. |

## MultiQC Report (`08_multiqc/`)

The MultiQC report at `08_multiqc/multiqc_report.html` aggregates per-sample QC metrics from FastQC, cutadapt, and STAR into a single interactive HTML report. Open it in any browser — no server required.

Key sections in the report:

| Section | Source | What to check |
|---|---|---|
| **FastQC: Sequence Quality** | FastQC | Per-base quality scores should be ≥ Q30 across most of the read. Drops at the 3′ end are normal and corrected by trimming. |
| **FastQC: Sequence Duplication** | FastQC | High duplication (>50%) in pooled library sequencing is expected; it reflects highly abundant UTR members, not a technical problem. |
| **Cutadapt: Trimming Stats** | cutadapt | Check the fraction of reads that had adapters trimmed and the fraction discarded for being too short (<30 bp). A high discard rate may indicate degraded input RNA. |
| **STAR: Alignment Summary** | STAR logs | Uniquely mapped rate should be >70% for a well-constructed library. High multi-mapper rates suggest the UTR reference has repetitive sequences. Low alignment rates may indicate a mismatch between the reference and the actual library. |

Raw data tables backing every plot are in `08_multiqc/multiqc_data/`.

## Dashboard

Results can be explored interactively via the Dash dashboard:

```bash
bash launch_dashboard.sh /path/to/results [port]
```

This prints the local URL (default `http://localhost:8050`). If you're running on a remote machine, SSH tunnel from your laptop:
```bash
ssh -L 8050:localhost:8050 <user>@<remote_host>
```

## Pipeline Architecture

The pipeline uses **Nextflow DSL2** with the local executor — every process runs as a subprocess on the machine you launch it from. Tool dependencies are supplied by Nextflow's built-in conda integration (`conda.enabled = true` in `nextflow.config`), which builds and caches an environment from `environment.yml` automatically — no `module load` or manual environment activation required.

`conf/local.config` controls resource allocation (`cpus`, `memory`) for the local executor — adjust it to match your machine.

## Example Files

| File | Description |
|---|---|
| [examples/sample_sheet.csv](examples/sample_sheet.csv) | Sample sheet with 2 Dox + 2 NoDox replicates |
| [examples/utr_sequences.csv](examples/utr_sequences.csv) | 5 example 5'UTR sequences |
| [examples/cds_sequence.txt](examples/cds_sequence.txt) | Default Citrine CDS sequence |
