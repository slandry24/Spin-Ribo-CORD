#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * UTR Translation Efficiency Pipeline — V5
 *
 * Maps R1 reads to UTR library, uses C→T edits in CDS-mapped R2 as a proxy
 * for translational activity. Runs Welch's t-test (Dox vs NoDox) and generates
 * comprehensive QC plots.
 *
 * Runs locally with conda — no cluster scheduler or environment modules
 * required (see environment.yml / README.md).
 *
 * Usage:
 *   nextflow run main.nf \
 *     --sample_sheet sample_sheet.csv \
 *     --utr_csv V2_pool_UTR_sequences.csv \
 *     --cds_seq "ATGGTC..." \
 *     --outdir results \
 *     -resume
 *
 *   # Skip reference building if already done:
 *   nextflow run main.nf ... \
 *     --genome_fasta results/reference/UTR_CDS.fasta \
 *     --cds_fasta    results/reference/CDS.fasta
 *
 *   # Skip STAR indexing too:
 *   nextflow run main.nf ... \
 *     --star_index_utrcds results/reference/star_index_utr_cds \
 *     --star_index_cds    results/reference/star_index_cds
 */

// ─── Parameters ───────────────────────────────────────────────────────────────
params.sample_sheet      = null         // Required: CSV (Sample,Fastq_path_R1,Fastq_path_R2,Treatment,Read1_Read2,Replicate)
params.utr_csv           = null         // UTR library CSV (name,sequence) — skip if --genome_fasta set
params.cds_seq           = null         // CDS sequence string — skip if --genome_fasta set
params.outdir            = "results"
params.library_type      = "5UTR"       // "5UTR" or "3UTR"
params.cds_header        = "citrine"
params.threads           = 4     // adjust to match your machine's core count
params.min_bq            = 15
params.mapq_utr          = 100
params.mapq_cds          = 100
params.min_reads         = 50
params.pseudo            = 1e-5
params.delivery1         = "Lenti"
params.delivery2         = "IVTMods"
params.color1            = "#4747d1"
params.color2            = "#f13636"

// Pre-built reference shortcuts (skip slow steps)
params.genome_fasta      = null
params.cds_fasta         = null
params.star_index_utrcds = null
params.star_index_cds    = null

// ─── Validation ───────────────────────────────────────────────────────────────
if (!params.sample_sheet)
    error "ERROR: --sample_sheet is required"
if (!params.genome_fasta && !params.utr_csv)
    error "ERROR: --utr_csv is required (or --genome_fasta to skip reference building)"
if (!params.genome_fasta && !params.cds_seq)
    error "ERROR: --cds_seq is required (or --genome_fasta to skip reference building)"

// ─── Processes ────────────────────────────────────────────────────────────────

process MAKE_REFERENCE {
    tag "make_reference"
    publishDir "${params.outdir}/reference", mode: 'copy'

    input:
    path utr_csv

    output:
    path "UTR_CDS.fasta", emit: genome_fasta
    path "CDS.fasta",     emit: cds_fasta

    script:
    """
    making_genome_fasta.py \\
        --utr_csv    ${utr_csv} \\
        --cds_seq    "${params.cds_seq}" \\
        --library_type ${params.library_type} \\
        --out_genome_fasta UTR_CDS.fasta \\
        --out_cds_fasta    CDS.fasta \\
        --cds_header ${params.cds_header}
    """
}

process STAR_INDEX_UTR_CDS {
    tag "star_index_utr_cds"
    publishDir "${params.outdir}/reference", mode: 'copy'

    input:
    path genome_fasta

    output:
    path "star_index_utr_cds", emit: index

    script:
    """
    mkdir -p star_index_utr_cds
    STAR \\
        --runThreadN        ${params.threads} \\
        --runMode           genomeGenerate \\
        --genomeDir         star_index_utr_cds \\
        --genomeFastaFiles  ${genome_fasta} \\
        --genomeSAindexNbases 11 \\
        --genomeChrBinNbits   10
    """
}

process STAR_INDEX_CDS {
    tag "star_index_cds"
    publishDir "${params.outdir}/reference", mode: 'copy'

    input:
    path cds_fasta

    output:
    path "star_index_cds", emit: index

    script:
    """
    mkdir -p star_index_cds
    STAR \\
        --runThreadN        ${params.threads} \\
        --runMode           genomeGenerate \\
        --genomeDir         star_index_cds \\
        --genomeFastaFiles  ${cds_fasta} \\
        --genomeSAindexNbases 3 \\
        --genomeChrBinNbits   9
    """
}

process CUTADAPT {
    tag "${sample_id}"

    input:
    tuple val(sample_id), path(r1), path(r2)

    output:
    tuple val(sample_id),
          path("${sample_id}_R1.trimmed.fastq.gz"),
          path("${sample_id}_R2.trimmed.fastq.gz"),
          emit: trimmed
    path "${sample_id}_cutadapt.json", emit: json

    script:
    """
    cutadapt \\
        -j ${params.threads} \\
        -q 25,25 \\
        -m 30 \\
        --pair-filter=any \\
        --json ${sample_id}_cutadapt.json \\
        -o ${sample_id}_R1.trimmed.fastq.gz \\
        -p ${sample_id}_R2.trimmed.fastq.gz \\
        ${r1} ${r2}
    """
}

process FASTQC {
    tag "${sample_id}"
    publishDir "${params.outdir}/02_fastqc", mode: 'copy'

    input:
    tuple val(sample_id), path(r1), path(r2)

    output:
    path "*.{html,zip}"

    script:
    """
    fastqc -t 4 ${r1} ${r2}
    """
}

process STAR_ALIGN_UTR {
    tag "${sample_id}"
    publishDir "${params.outdir}/03_star_align_utr_cds", mode: 'symlink'

    input:
    tuple val(sample_id), val(utr_label), path(utr_fastq)
    path star_index

    output:
    tuple val(sample_id),
          path("${sample_id}.${utr_label}.Aligned.sortedByCoord.out.bam"),
          path("${sample_id}.${utr_label}.Aligned.sortedByCoord.out.bam.bai"),
          emit: bam
    path "${sample_id}.${utr_label}.Log.final.out", emit: log

    script:
    """
    STAR \\
        --runThreadN          ${params.threads} \\
        --genomeDir           ${star_index} \\
        --readFilesIn         ${utr_fastq} \\
        --readFilesCommand    zcat \\
        --alignIntronMax      1 \\
        --alignEndsType       Local \\
        --outSAMtype          BAM SortedByCoordinate \\
        --outFileNamePrefix   ${sample_id}.${utr_label}. \\
        --outSAMattributes    NH HI AS nM \\
        --outBAMsortingThreadN ${params.threads} \\
        --limitBAMsortRAM     1000000000000

    mv ${sample_id}.${utr_label}.Aligned.sortedByCoord.out.bam _tmp.bam
    samtools view -@ ${params.threads} -b -h -F 0x100 -F 0x800 -F 0x4 _tmp.bam \\
        > ${sample_id}.${utr_label}.Aligned.sortedByCoord.out.bam
    rm -f _tmp.bam
    samtools index -@ ${params.threads} ${sample_id}.${utr_label}.Aligned.sortedByCoord.out.bam
    """
}

process STAR_ALIGN_CDS {
    tag "${sample_id}"
    publishDir "${params.outdir}/04_star_align_cds_md", mode: 'symlink'

    input:
    tuple val(sample_id), val(cds_label), path(cds_fastq)
    path star_index
    path cds_fasta

    output:
    tuple val(sample_id),
          path("${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.md.bam"),
          path("${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.md.bam.bai"),
          emit: bam
    path "${sample_id}.${cds_label}.cds.Log.final.out", emit: log

    script:
    """
    STAR \\
        --runThreadN          ${params.threads} \\
        --genomeDir           ${star_index} \\
        --readFilesIn         ${cds_fastq} \\
        --readFilesCommand    zcat \\
        --alignIntronMax      1 \\
        --alignEndsType       Local \\
        --outSAMtype          BAM SortedByCoordinate \\
        --outFileNamePrefix   ${sample_id}.${cds_label}.cds. \\
        --outSAMattributes    NH HI AS nM \\
        --outBAMsortingThreadN ${params.threads} \\
        --limitBAMsortRAM     1000000000000

    mv ${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.bam _tmp.bam
    samtools view -@ ${params.threads} -b -h -F 0x100 -F 0x800 -F 0x4 _tmp.bam \\
        > ${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.bam
    rm -f _tmp.bam

    samtools calmd -@ ${params.threads} -bAr \\
        ${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.bam \\
        ${cds_fasta} \\
        > ${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.md.bam

    samtools index -@ ${params.threads} \\
        ${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.md.bam

    rm -f ${sample_id}.${cds_label}.cds.Aligned.sortedByCoord.out.bam
    """
}

process MAPPING_COUNTING_SAMPLE {
    tag "${sample_id}"

    input:
    tuple val(sample_id), path(utr_bam), path(utr_bai), path(cds_bam), path(cds_bai)

    output:
    tuple path("${sample_id}_detail.csv"), path("${sample_id}_summary.csv"), emit: counts

    script:
    """
    mkdir -p utr_dir cds_dir
    cp ${utr_bam} utr_dir/
    cp ${utr_bai} utr_dir/
    cp ${cds_bam} cds_dir/
    cp ${cds_bai} cds_dir/

    mapping_counting_argparse_v3.py \\
        --utr-dir      utr_dir \\
        --cds-dir      cds_dir \\
        --outdir       . \\
        --glob-pattern "*.bam" \\
        --min-bq       ${params.min_bq} \\
        --mapq-utr     ${params.mapq_utr} \\
        --mapq-cds     ${params.mapq_cds} \\
        --cores        1

    mv ALL_SAMPLES_c2t_per_read_per_position.csv ${sample_id}_detail.csv
    mv ALL_SAMPLES_c2t_per_utr_reads_edits.csv   ${sample_id}_summary.csv
    """
}

process CONCAT_COUNTING {
    publishDir "${params.outdir}/05_mapping_counting", mode: 'copy'

    input:
    path detail_csvs
    path summary_csvs

    output:
    path "ALL_SAMPLES_c2t_per_read_per_position.csv", emit: detail_csv
    path "ALL_SAMPLES_c2t_per_utr_reads_edits.csv",  emit: summary_csv

    script:
    """
    python3 -c "
import csv, glob, os

def stream_concat(pattern, outfile):
    files = sorted(glob.glob(pattern))
    header_written = False
    with open(outfile, 'w', newline='') as out:
        writer = None
        for f in files:
            with open(f, newline='') as inp:
                reader = csv.reader(inp)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer = csv.writer(out)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)

stream_concat('*_detail.csv',  'ALL_SAMPLES_c2t_per_read_per_position.csv')
stream_concat('*_summary.csv', 'ALL_SAMPLES_c2t_per_utr_reads_edits.csv')
"
    """
}

process ANALYSIS {
    tag "welch_ttest"
    memory '32 GB'
    publishDir "${params.outdir}/06_analysis", mode: 'copy'

    input:
    path summary_csv
    path genome_fasta

    output:
    path "welches_t_test_EPR.csv", emit: welch_csv

    script:
    """
    analysis.py \\
        --input      ${summary_csv} \\
        --outdir     . \\
        --fasta      ${genome_fasta} \\
        --min-reads  ${params.min_reads} \\
        --pseudo     ${params.pseudo}
    """
}

process QC_PLOTS {
    tag "qc_plots"
    memory '32 GB'
    publishDir "${params.outdir}/07_qc_plots", mode: 'copy'

    input:
    path welch_csv
    path summary_csv

    output:
    path "*.png"

    script:
    """
    qc_plots.py \\
        --welch-results ${welch_csv} \\
        --raw-data      ${summary_csv} \\
        --outdir        . \\
        --sample-col    Sample_Stem \\
        --utr-col       UTR_Reference \\
        --reads-col     Total_Reads \\
        --epr-col       EPR \\
        --delivery1     "${params.delivery1}" \\
        --delivery2     "${params.delivery2}" \\
        --color1        "${params.color1}" \\
        --color2        "${params.color2}"
    """
}

process MULTIQC {
    tag "multiqc"
    publishDir "${params.outdir}/08_multiqc", mode: 'copy'

    input:
    path files

    output:
    path "multiqc_report.html"
    path "multiqc_data"

    script:
    """
    export PYTHONNOUSERSITE=1
    multiqc .
    """
}

// ─── Main Workflow ─────────────────────────────────────────────────────────────

workflow {

    // ── 1. Register run provenance ────────────────────────────────────────────
    // ── 2. Parse sample sheet ─────────────────────────────────────────────────
    ch_samples = Channel
        .fromPath(params.sample_sheet)
        .splitCsv(header: true, strip: true)
        .map { row ->
            def sample    = row.Sample.replaceAll(/[\/ ]/, '_')
            def treatment = row.Treatment
                               .replaceAll(/[Nn]o[\s_-]?[Dd]ox/, 'NoDox')
                               .replaceAll(/[\s-]/, '')
            def replicate = row.Replicate.toString().trim()
            def readmap   = row.Read1_Read2.trim().toUpperCase().replaceAll(/[-\s]/, '_')
            // Optional Timepoint column: if present and non-empty, inserted between treatment and replicate
            def timepoint = row.containsKey('Timepoint') ? (row.Timepoint ?: '').toString().trim() : ''
            def sample_id = timepoint ? "${sample}_${treatment}_${timepoint}_R${replicate}"
                                      : "${sample}_${treatment}_R${replicate}"
            [ sample_id, file(row.Fastq_path_R1.trim()), file(row.Fastq_path_R2.trim()), readmap ]
        }

    // ── 3. Reference genomes ──────────────────────────────────────────────────
    if (params.genome_fasta && params.cds_fasta) {
        ch_genome = Channel.fromPath(params.genome_fasta).first()
        ch_cds    = Channel.fromPath(params.cds_fasta).first()
    } else {
        MAKE_REFERENCE(Channel.fromPath(params.utr_csv))
        ch_genome = MAKE_REFERENCE.out.genome_fasta.first()
        ch_cds    = MAKE_REFERENCE.out.cds_fasta.first()
    }

    // ── 4. STAR indices ───────────────────────────────────────────────────────
    if (params.star_index_utrcds) {
        ch_idx_utr = Channel.fromPath(params.star_index_utrcds).first()
    } else {
        STAR_INDEX_UTR_CDS(ch_genome)
        ch_idx_utr = STAR_INDEX_UTR_CDS.out.index.first()
    }

    if (params.star_index_cds) {
        ch_idx_cds = Channel.fromPath(params.star_index_cds).first()
    } else {
        STAR_INDEX_CDS(ch_cds)
        ch_idx_cds = STAR_INDEX_CDS.out.index.first()
    }

    // ── 5. Trim ───────────────────────────────────────────────────────────────
    CUTADAPT(ch_samples.map { id, r1, r2, rm -> [ id, r1, r2 ] })

    // ── 6. FastQC ─────────────────────────────────────────────────────────────
    FASTQC(CUTADAPT.out.trimmed)

    // ── 7. Route UTR vs CDS reads by readmap column ───────────────────────────
    ch_routed = ch_samples
        .join(CUTADAPT.out.trimmed)
        .map { id, r1_raw, r2_raw, readmap, r1_trim, r2_trim ->
            def utr_label = (readmap == 'UTR_CDS') ? 'R1' : 'R2'
            def cds_label = (readmap == 'UTR_CDS') ? 'R2' : 'R1'
            def utr_fq    = (readmap == 'UTR_CDS') ? r1_trim : r2_trim
            def cds_fq    = (readmap == 'UTR_CDS') ? r2_trim : r1_trim
            [ id, utr_label, cds_label, utr_fq, cds_fq ]
        }

    // ── 8. Align ──────────────────────────────────────────────────────────────
    STAR_ALIGN_UTR(
        ch_routed.map { id, ul, cl, uf, cf -> [ id, ul, uf ] },
        ch_idx_utr
    )

    STAR_ALIGN_CDS(
        ch_routed.map { id, ul, cl, uf, cf -> [ id, cl, cf ] },
        ch_idx_cds,
        ch_cds
    )

    // ── 9. Per-sample counting then concat ───────────────────────────────────
    ch_paired_bams = STAR_ALIGN_UTR.out.bam
        .join(STAR_ALIGN_CDS.out.bam)

    MAPPING_COUNTING_SAMPLE(ch_paired_bams)

    ch_detail_csvs  = MAPPING_COUNTING_SAMPLE.out.counts.map { d, s -> d }.collect()
    ch_summary_csvs = MAPPING_COUNTING_SAMPLE.out.counts.map { d, s -> s }.collect()

    CONCAT_COUNTING(ch_detail_csvs, ch_summary_csvs)

    // ── 10. Statistical analysis ──────────────────────────────────────────────
    ANALYSIS(CONCAT_COUNTING.out.summary_csv, ch_genome)

    // ── 11. QC plots ──────────────────────────────────────────────────────────
    QC_PLOTS(ANALYSIS.out.welch_csv, CONCAT_COUNTING.out.summary_csv)

    QC_PLOTS.out.view { "Pipeline complete. Results: ${params.outdir}" }

    // ── 12. MultiQC ───────────────────────────────────────────────────────────
    ch_mqc = FASTQC.out
        .mix(CUTADAPT.out.json)
        .mix(STAR_ALIGN_UTR.out.log)
        .mix(STAR_ALIGN_CDS.out.log)
        .flatten()
        .collect()

    MULTIQC(ch_mqc)
}
