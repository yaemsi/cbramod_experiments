# Part 2 — Data Harmonization for Large-Scale EEG Pretraining

## Objective

The goal is to aggregate heterogeneous EEG data such as:

- SHU-MI: pre-segmented MAT or LMDB-style examples, plus continuous EDF/event data;
- HBN: BIDS-organized SET/BDF recordings with high-density channels and richer metadata;

and produce a reproducible corpus that can scale to several terabytes and feed a large training cluster at approximately 1 GB/s aggregate throughput.

The project includes a working vertical prototype. It processes the full SHU-MI dataset through the same representation used for training and supports a representative HBN/BIDS subset through a generalized reader.

## 1. Data architecture

### Raw layer

Raw data remains immutable in source format, ideally in versioned object storage:

```text
raw/shu-mi/...
raw/hbn/...
```

This preserves auditability, enables new preprocessing versions, and avoids losing source metadata or introducing irreversible transformations.

### Reader boundary

Dataset-specific complexity is isolated behind readers:

```text
SHU-MI MAT ──────────────┐
SHU-MI EDF + events ─────┤
HBN BIDS SET/BDF ─────────┤
                         v
                 canonical EEG schema
```

Downstream transforms and training do not depend on source file type.

### Canonical recording

A canonical continuous recording contains:

- signal `[channels, time]`;
- sampling rate and physical units;
- channel names and types;
- subject, session, task, and dataset identity;
- events;
- reference and electrode metadata when available;
- source URI and source format;
- arbitrary provenance metadata.

It does not impose a universal channel count, sample rate, duration, or label vocabulary.

### Canonical window

After preprocessing/windowing, each example contains:

- fixed signal `[channels, samples]`;
- channel names and channel mask;
- sample rate and time boundaries;
- dataset/subject/session/task;
- split and optional label;
- quality flags/score;
- preprocessing version and source information.

### Storage layout

The prototype uses:

```text
harmonized_dataset/
├── manifest.parquet
├── summary.json
└── shards/
    ├── shard-00000.arrow
    └── ...
```

- **Parquet** stores searchable metadata and sample locations, enabling column pruning and predicate filtering.
- **Arrow IPC** stores dense float32 tensors in compressed record batches for efficient local/memory-mapped reads.
- **Zarr** would be a reasonable production addition for canonical continuous recordings requiring arbitrary temporal slicing.

Arrow and Parquet are complementary rather than competing choices.

## 2. Processing pipeline

### Versioned deterministic DAG

```text
discover
  -> validate source/metadata
  -> load recording
  -> normalize units and channel names
  -> apply reference/channel policy
  -> filter/resample
  -> quality control
  -> event-based or sliding window extraction
  -> shard write
  -> manifest publication
```

Every output should be traceable to:

- source checksum and URI;
- dataset/release version;
- preprocessing configuration hash;
- code commit/container image;
- transform parameters;
- output checksum and pipeline version.

### Parallelization

The natural work unit is a subject/session/task recording. These tasks are mostly independent and can be scheduled with Ray, Spark, Kubernetes jobs, Slurm, or a cloud batch service.

A production task should be:

- idempotent;
- independently retryable;
- written to a temporary destination before atomic publication;
- accompanied by a manifest fragment and status record.

### Offline and online transforms

Materialize stable expensive operations offline:

- unit normalization;
- channel-name normalization;
- fixed filtering/reference policy;
- resampling;
- QC statistics;
- deterministic window indexing.

Keep stochastic augmentations online:

- temporal crop/jitter;
- channel dropout;
- amplitude/noise perturbation;
- frequency masking;
- self-supervised masking.

### Quality control

Candidate checks include:

- NaN/Inf values;
- flat lines and disconnected channels;
- clipping/extreme amplitudes;
- channel/sample-rate inconsistencies;
- excessive line noise or high-frequency energy;
- impossible duration/event bounds;
- missing metadata and electrode positions.

Prefer recording quality flags/scores in the manifest over irreversible early deletion, unless a sample is clearly corrupt.

## 3. Channel and sampling-rate harmonization

SHU-MI uses 32 channels while HBN commonly uses a 128-channel EGI system. A production pipeline should support several policies:

1. **preserve native channels:** keep native order plus channel identity/coordinates;
2. **common subset:** select channels present across target datasets;
3. **global vocabulary plus mask:** map into a fixed channel universe and mask absent channels;
4. **spatial interpolation:** project to a canonical montage using electrode coordinates.

The prototype implements native preservation and configured selection/masking. Spatial interpolation is deliberately left as an extension because it adds scientific assumptions and potential artifacts.

Sample rate is configuration-driven. SHU-MI reproduction uses 200 Hz. HBN can retain its source rate or be downsampled as required by a chosen pretraining objective. The pipeline should not upsample lower-rate data solely to imitate another dataset's tensor shape.

## 4. Data streaming at cluster scale

The target path is:

```text
object storage
    -> rank/node shard assignment
    -> asynchronous download
    -> node-local NVMe cache
    -> parallel decompression/decoding
    -> pinned-memory batches
    -> non-blocking GPU transfer
```

### Sharding

Avoid both a single huge file and millions of tiny files. A reasonable starting point is:

- 512 MB to 4 GB per shard;
- tens to hundreds of MB per internal batch/chunk;
- enough shards to distribute work across nodes/ranks;
- subject-aware partitioning to simplify leakage controls.

### Shuffling

Use two-level shuffling:

1. shuffle shard/record-batch order each epoch;
2. shuffle examples within a bounded buffer or record batch.

The implemented Arrow loader follows this idea with record-batch-aware shuffling. Fully random row access across compressed shards can repeatedly decompress unrelated blocks and severely reduce throughput.

### Caching and overlap

- cache remote shards on node-local NVMe;
- share cache among GPUs on a node;
- asynchronously prefetch future shards;
- use persistent data workers;
- overlap remote I/O, local reads, decompression, collation, host-to-device transfer, and GPU compute;
- verify shard checksums before use;
- use an LRU/epoch-aware eviction policy.

### Throughput measurement

Treat 1 GB/s as an end-to-end requirement, not merely a storage specification. Measure separately:

- object store to NVMe;
- NVMe to RAM;
- decompression/deserialization;
- online augmentation and collation;
- pinned-memory transfer to GPU;
- dataloader idle time observed by the trainer.

The local prototype validates architecture and parity; it does not claim a measured distributed 1 GB/s result.

## 5. Kafka and streaming frameworks

Kafka should not be the primary repeated training-data store for a static multi-terabyte corpus. It introduces broker retention, duplication, partitioning, and replay complexity while offering poor random/repeated epoch access compared with object storage plus shards.

Kafka is useful in the control/ingestion plane:

```text
new recording uploaded
  -> Kafka event
  -> validation/preprocessing task
  -> shard written to object storage
  -> manifest/catalog update
  -> dataset-version publication event
```

Events should generally contain source URIs and metadata, not full EEG tensors. Flink, Beam, or Spark Structured Streaming become useful when recordings arrive continuously and require incremental validation or transformation. Static public releases are simpler to process with batch orchestration.

## 6. Sampling, leakage, and governance

### Dataset-aware sampling

Naively concatenating corpora can let HBN dominate pretraining. Use a dataset-aware sampler, such as probabilities proportional to `size^alpha` with `alpha < 1`, or manually specified weights. Also consider subject/task/quality balance so long recordings do not dominate.

### Leakage prevention

Assign dataset/subject splits before windowing and sharding. Enforce:

```text
(dataset_id, subject_id) -> exactly one split
```

All sessions for a subject normally remain together. Deduplicate recordings that may appear in multiple distributions or preprocessing versions.

### Governance

Track:

- dataset license and permitted uses;
- participant consent and clinical restrictions;
- personally identifying metadata removal;
- dataset/version lineage;
- withdrawal/deletion propagation;
- which exact corpus generated each checkpoint.

## 7. Implemented prototype

### SHU-MI full path

The entire SHU-MI MAT corpus can be converted to the canonical Arrow/Parquet representation and used directly by CBraMod or EEGSimpleConv.

```bash
make harmonize-shu RAW_DIR=/path/to/shu/mat_files OVERWRITE=1
make compare-backends DATASET=data/processed/shu_mi.h5
make train-cbramod DATASET=data/harmonized/shu_mi/manifest.parquet DATA_BACKEND=arrow
```

### SHU-MI EDF/event validation

A separate reader reconstructs trials from continuous EDF plus event TSV. On the bundled real session:

- MAT and EDF/event paths both produce `[100, 32, 1000]` raw trial tensors;
- labels are identical;
- signal correlation exceeds 0.999999 after unit alignment;
- HDF5 and Arrow processed signals match exactly.

This validates that the generalized continuous-recording path produces the same examples used by the paper-oriented MAT path.

### HBN/BIDS subset path

The BIDS reader supports EDF, BDF, and SET/FDT files plus local sidecars. A small subset can be harmonized with:

```bash
make harmonize-hbn \
  HBN_ROOT=/path/to/hbn_subset \
  HBN_LIMIT_RECORDINGS=3 \
  OVERWRITE=1
```

HBN windows are kept unlabeled and separate from SHU-MI metrics. Their purpose is to prove that the same canonical schema, transforms, writer, manifest, and PyTorch backend handle another source with different channels, sampling rates, tasks, and metadata.

### Prototype tests

The tests verify:

- canonical schema validation;
- BIDS file and sidecar discovery;
- EDF materialization;
- MAT/EDF event reconstruction equivalence;
- Arrow write/read round trips;
- worker-safe loading;
- block-aware shuffling;
- exact HDF5/Arrow signal and metadata parity;
- one complete training cycle through Arrow.
