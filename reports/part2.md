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

The implemented prototype uses one shared bundle-parallel engine for SHU-MI MAT, SHU-MI EDF/event, and HBN/BIDS inputs. A source-specific reader discovers recordings, then a deterministic first-fit-decreasing planner groups them by estimated source bytes. EEGLAB estimates include sibling `.fdt` payloads. Each worker keeps one writer open across its bundle, allowing large shards to span many recordings while orchestration, failure handling, progress reporting, publication, and manifest merging remain common.

```text
source-specific discovery
        -> deterministic size-estimated recording bundles
        -> spawn-based worker processes
        -> worker-local Arrow shards and Parquet fragment
        -> coordinator/rank-0 deterministic merge
        -> final manifest, source audit, summary, and shards
```

This design is intentionally different from both a shared writer and one shard per recording:

- Arrow writers are process-local, avoiding corruption and lock contention;
- each writer remains open across many recordings, packing record batches efficiently across recording boundaries;
- large EEG arrays are written by the worker rather than serialized back to the parent;
- only small job results cross process boundaries;
- the coordinator is the only process that writes final shared metadata;
- the coordinator alone owns the `tqdm` progress bar, so worker output does not overlap;
- native BLAS/FFT thread pools are constrained in the Makefile to avoid worker × thread oversubscription.

The same engine runs with `num_workers=1`, making serial and parallel output directly comparable. Final merge order follows discovery order rather than completion order, so the manifest and shard naming are deterministic.

Each worker writes under `_work/job-XXXXXX/` and creates a success marker only after its shard and fragment are complete. The marker includes a source/configuration fingerprint, so resume mode does not reuse output after an input file or preprocessing option changes. On success, the coordinator:

1. validates equal schemas;
2. rejects duplicate sample IDs;
3. builds the complete rewritten shard-path plan without modifying files;
4. stages the final Parquet manifest;
5. moves worker shards with `shutil.move` into `_publishing_shards/` (normally a same-filesystem rename rather than a second corpus copy);
6. atomically renames `_publishing_shards/` to `shards/`;
7. atomically publishes `manifest.parquet` and `summary.json`;
8. removes `_work/`.

If strict mode encounters invalid recordings, no final manifest is published and `_work/` is retained for diagnosis. If final shard publication fails, already moved shards are returned to their worker directories, leaving no partial `shards/` namespace and preserving resume capability. Before moving, rank 0 writes `_PUBLICATION_PLAN.json`; resume mode uses this durable mapping to recover staged or finalized shards after an abrupt coordinator interruption. An interrupted run can reuse completed jobs with `--resume`; `--overwrite` instead starts from a clean output and is mutually exclusive with resume. Lenient mode skips invalid recordings, records path/type/message, and publishes the remaining valid corpus.

Although futures complete per bundle, the rank-0 progress bar advances by the number of recordings attempted and reports current example and skipped-recording counts. For example:

```text
Harmonizing bids:  47%|███████████▎            | 47/100 [08:31<09:18, examples=28413, skipped=2]
```

Bundle fingerprints include every source path, size, modification time, and preprocessing setting, so resume reuses only compatible completed bundles. This local process-pool implementation maps naturally to Ray, Slurm, Kubernetes jobs, or cloud batch execution because the worker contract is already bundle-local and produces independent artifacts.

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

### Shared execution engine

All source types use `harmonize_recordings` from `data_harmonization/parallel.py`. Public wrappers perform source discovery and supply reader-specific options:

- `harmonize_shu_mat`;
- `harmonize_shu_edf`;
- `harmonize_bids`.

The CLI exposes `--num-workers`, `--skip-invalid-recordings`, `--resume`, `--overwrite`, and `--no-progress`. The Makefile defaults to four workers and a rank-0 progress bar.

Every completed output has:

```text
output_dir/
├── manifest.parquet
├── summary.json
├── source_audit.json
└── shards/
    ├── shard-00000.arrow
    └── ...
```

`summary.json` records total/processing/merge time plus recordings, examples, and signal MiB per second. `source_audit.json` records discovered, processed, skipped, and resumed recordings.

### SHU-MI full MAT path

The complete SHU-MI MAT corpus is harmonized in parallel and remains directly usable by CBraMod or EEGSimpleConv:

```bash
make harmonize-shu \
  RAW_DIR=/path/to/one/authoritative/shu/mat_tree \
  HARMONIZE_WORKERS=4 \
  OVERWRITE=1

make inspect-harmonized
make compare-backends
```

The reader rejects duplicate subject/session source files before processing, preventing accidentally extracted duplicate trees from doubling the corpus. The Arrow audit then verifies 11,988 unique examples and the paper subject split.

### SHU-MI EDF/event validation

The optional EDF reader reconstructs trials from continuous recordings and event TSV files through the same engine:

```bash
make harmonize-shu-edf HARMONIZE_WORKERS=4 OVERWRITE=1
```

Some distributed EDF/event files are malformed. The Makefile therefore enables lenient mode for this optional path by default; failures remain visible in `source_audit.json`. Reported CBraMod and EEGSimpleConv results continue to use the MAT files.

On valid matching recordings, MAT and EDF/event paths produce identical labels, aligned trial shapes, correlation above 0.999999, and differences consistent with EDF quantization.

### HBN/BIDS subset path

The BIDS reader discovers the supported files actually present rather than requiring BDF specifically. It supports EEGLAB SET/FDT, EDF, and BDF together with recording/task sidecars. Continuous signals are resampled optionally, windowed, and written through the same worker/merge engine. The BIDS recording stem, including entities such as `run` and `acq`, is retained in the canonical `recording_id`; this prevents separate runs of the same subject/task from colliding during final sample-ID validation.

```bash
make harmonize-hbn \
  HBN_ROOT=/path/to/hbn_subset \
  HBN_LIMIT_RECORDINGS=10 \
  HBN_WINDOW_SECONDS=4 \
  HBN_STRIDE_SECONDS=4 \
  HARMONIZE_WORKERS=4 \
  OVERWRITE=1
```

HBN windows are unlabeled in this prototype and remain separate from SHU-MI motor-imagery metrics. Their role is to validate heterogeneous ingestion, montage metadata, continuous windowing, and scalable materialization.

### Training and streaming

The final Parquet/Arrow dataset supports two training access modes:

- random-access Arrow with record-batch-aware shuffling;
- iterable Arrow streaming with rank/worker shard partitioning and bounded-buffer shuffling.

```bash
make train-cbramod \
  DATASET=outputs/data/harmonized/shu_mi/manifest.parquet \
  DATA_BACKEND=arrow_streaming

make benchmark-streaming \
  STREAM_MANIFEST=outputs/data/harmonized/hbn/manifest.parquet
```

For an end-to-end data-only measurement, the project also iterates one complete epoch through the same `EEGDataModule` used by training:

```bash
make benchmark-dataloader \
  DATALOADER_DATA=outputs/data/harmonized/shu_mi/manifest.parquet \
  DATALOADER_BACKEND=arrow_streaming \
  DATALOADER_NUM_WORKERS=4
```

This verifies that every selected example is visited exactly once and records first-batch latency, full-epoch wall time, examples/s, and uncompressed signal MiB/s. The interactive notebook [`../notebooks/harmonized_dataloader_benchmark.ipynb`](../notebooks/harmonized_dataloader_benchmark.ipynb) shows the first batch, manifest view, full result, and an optional worker-count sweep. The benchmark excludes model compute; its purpose is to determine whether storage, decoding, collation, or host-to-device transfer can keep up with the training system.

### Prototype tests

The tests cover:

- canonical schema validation;
- MAT, EDF/event, and BIDS reader behavior;
- Arrow write/read and streaming paths;
- exact HDF5/Arrow parity;
- MAT/EDF reconstruction equivalence;
- serial-versus-parallel manifest and tensor equivalence;
- deterministic final ordering;
- strict failure without partial publication;
- lenient failure auditing;
- resume from completed worker directories;
- BIDS run-aware sample identities;
- duplicate-ID detection before publication;
- move-based publication and rollback after simulated failure;
- duplicate sample-ID rejection;
- one complete training cycle through Arrow;
- full-epoch dataloader accounting for random-access and streaming Arrow.

## 8. Current limitations and next-step subject-aware layout

The implemented prototype proves heterogeneous ingestion, parallel materialization,
provenance, and full-epoch streaming. It should nevertheless be viewed as a
working vertical slice rather than the final pretraining layout.

### Measured local baseline

On the harmonized HBN subset, the streaming loader consumed all 105,151
four-second windows exactly once. These windows contain approximately 101 GiB of
uncompressed float32 signal data. With batch size 128, four persistent workers,
prefetch factor 2, CPU output, and no effective example-level shuffling, one
complete epoch took approximately 1,238.7 seconds:

```text
examples/s:       84.9
signal throughput: 83.5 MiB/s
first batch:       [128, 129, 2000]
complete epoch:    true
```

This is a useful correctness and local-throughput baseline, but remains well below
a hypothetical 1 GB/s aggregate cluster target. It also reveals several areas
where the current format and loader can be improved.

### Limitations of the current design

1. **Per-example decoding and collation**

   The streaming reader currently reconstructs individual examples from Arrow
   payloads and yields one tensor at a time. PyTorch's default collator then
   allocates another tensor and copies those examples into a model batch. For a
   129-channel, 2,000-sample float32 window, every example is close to 1 MiB, so
   row-wise conversion and collation generate substantial Python overhead and
   memory traffic.

2. **Subject identity is preserved for provenance but not exposed as a typed
   training column**

   The manifest records `subject_id`, `recording_id`, task, run, and source URI,
   but the common training interface primarily returns signal and label tensors.
   This makes subject-level metrics, balanced subject sampling, and leak checks
   less convenient at training time.

3. **Bundle planning is based on estimated source bytes, not exact output
   compatibility or output volume**

   Source-file size is a practical proxy and prevents one shard per recording,
   but files of similar size can produce different numbers of windows depending
   on duration, sampling rate, channels, and preprocessing. The current planner
   also does not make a formal compatibility key part of shard placement.

4. **Heterogeneous shapes require explicit views**

   A standard model batch can stack examples only when channel count, sample
   count, dtype, and normally channel layout agree. A first HBN batch of
   `[128, 129, 2000]` demonstrates one homogeneous view, but the general pipeline
   must not assume that every future dataset or HBN release has the same montage,
   rate, and duration.

5. **The HBN throughput corpus has no experiment split**

   `split=None` is appropriate for measuring one complete data pass, but not for
   model selection. A real pretraining experiment should assign subjects to
   train/validation/test or train/validation before materializing experiment
   views. All recordings from one subject must remain in one split.

6. **Sequential recording locality can produce correlated batches**

   Keeping windows from one recording contiguous improves I/O, but a small
   shuffle buffer can yield batches dominated by neighboring windows or one
   participant. This may reduce gradient diversity and can affect models using
   batch normalization.

7. **Configured and effective shuffling can differ**

   In the current loader, an unsplit `split=None` view defaults to deterministic
   iteration unless shuffling is requested explicitly. Benchmark output should
   therefore distinguish configured buffer size from the effective buffer used
   by the dataset.

### Proposed next-step storage and batching design

The next iteration should retain subject and recording boundaries as metadata
while physically mixing compatible recordings from many subjects into large,
well-filled shards:

```text
subject A / recording 1 ─┐
subject B / recording 1 ─┤
subject C / recording 2 ─┼─> compatibility bucket
subject A / recording 3 ─┘       ↓
                           size-packed Arrow shards
                                  ↓
                      batched signal + metadata tensors
```

#### 1. Add compact typed identity columns

Store compact categorical indices alongside each signal example:

```text
subject_index:   int32
recording_index: int32
task_index:      int16 or int32
split_index:     int8
```

Keep the original pseudonymous strings and source provenance in Parquet lookup
tables or the main manifest:

```text
subject_index -> subject_id
recording_index -> recording_id, source_uri, task, run, acquisition
```

The subject index is intended for sampling, auditing, and evaluation. It should
not normally be passed to the EEG backbone, because doing so can encourage
participant memorization and does not generalize to unseen subjects.

#### 2. Assign subject-level splits before shard planning

For an actual HBN pretraining experiment, derive a deterministic split from
`(dataset_id, subject_id, split_seed)` and enforce:

```text
(dataset_id, subject_id) -> exactly one split
```

All sessions, tasks, and runs from one participant remain together. The complete
canonical corpus can remain unsplit, while lightweight Parquet experiment views
assign train/validation/test without rewriting raw sources.

#### 3. Bucket examples by tensor compatibility

Before size packing, group recordings using a compatibility key such as:

```python
compatibility_key = (
    split,
    sampling_rate_hz,
    num_channels,
    num_samples,
    signal_dtype,
    channel_layout_id,
)
```

`task` may also be included when task-specific loaders are important. Recordings
from different subjects may share a shard only when their output examples are
batch-compatible. Different shapes are written to separate shard groups or
handled through an explicit padding-and-mask policy.

#### 4. Pack multiple subjects and recordings into each shard

A shard should not be tied to one subject. Subject-aligned shards tend to be
small and uneven because participant recording volumes differ. Instead:

- keep windows from one recording contiguous where practical;
- allow many recordings and subjects in each 1–4 GiB shard;
- avoid mixing experiment splits within a shard;
- retain `subject_index`, `recording_index`, and source offsets for every row;
- size-pack within each compatibility bucket using estimated output bytes.

This preserves provenance while improving shard utilization, sequential I/O,
shuffle quality, and load balancing across ranks.

#### 5. Decode Arrow record batches into already-batched tensors

The highest-value loader optimization is to replace row-wise decoding with a
record-batch-oriented path:

```text
current:
Arrow rows -> N Python objects -> N NumPy arrays -> N tensors
           -> default collate -> [B, C, T]

next step:
Arrow RecordBatch -> contiguous NumPy [N, C, T]
                  -> one torch.Tensor [N, C, T]
                  -> batched metadata tensors [N]
```

The iterable dataset can then emit a structured pre-batched object and the
PyTorch `DataLoader` can use `batch_size=None`:

```python
{
    "signals": Tensor[B, C, T],
    "labels": Tensor[B],
    "subject_indices": Tensor[B],
    "recording_indices": Tensor[B],
    "channel_masks": Tensor[B, C],
}
```

This removes one Python loop layer, many small allocations, and the default
collation copy. Arrow record-batch size remains a storage/decode parameter and
need not equal the optimizer batch size; a decoded record batch can be sliced
into several model batches.

#### 6. Add optional subject-aware batch construction

For general pretraining, a useful sampling policy is to represent several
participants in each model batch, for example:

```text
8 subjects x 8 windows per subject = batch size 64
```

Benefits include better subject balance, less dominance by long recordings, and
more diverse batch-normalization statistics. The trade-offs are additional
sampler complexity, potentially less sequential locality, and the need to avoid
oversampling subjects with very little data. A bounded shuffle buffer remains a
reasonable simpler baseline.

### Benefits of the proposed design

- processes all subjects through one common loader rather than one subject at a
  time;
- retains subject and recording boundaries without physically isolating them;
- produces fewer, larger, shape-compatible shards;
- supports subject-level metrics, leak checks, and balanced sampling;
- improves rank/worker load balancing;
- enables vectorized record-batch decoding and avoids an extra collation copy;
- permits task-, montage-, and split-specific views without duplicating the
  signal corpus.

### Risks and trade-offs

- compatibility bucketing can fragment a highly heterogeneous corpus into many
  small groups;
- storing metadata in every Arrow row duplicates some manifest information,
  although compact integer indices keep the overhead small;
- subject-aware batches may reduce sequential I/O and require a more complex
  distributed sampler;
- batches dominated by one subject or recording can produce correlated gradients
  and unstable batch-normalization behavior;
- subject identifiers require pseudonymization and controlled lookup tables;
- feeding subject identity into the model would create a memorization risk and is
  not part of the proposed default;
- vectorized decoding may require changing the iterable-dataset contract and
  carefully handling partial final record batches.

### Acceptance tests for the next iteration

The next design should be considered correct when:

```text
[ ] every subject belongs to exactly one experiment split
[ ] every sample retains subject and recording provenance
[ ] each shard contains only shape-compatible examples
[ ] at least some shards contain multiple recordings and subjects
[ ] sample IDs remain unique after multi-subject packing
[ ] the streaming loader observes every selected example exactly once
[ ] structured metadata tensors align with the signal batch
[ ] serial and parallel harmonization remain logically equivalent
[ ] row-batched decoding improves throughput without changing signals
[ ] subject-aware sampling does not duplicate or omit examples unexpectedly
```

This next step keeps the current canonical representation and parallel
materialization architecture, but improves the physical layout and loader path
for genuine large-scale pretraining.

## 9. Questions to raise during the debrief

### Scientific

- How should source referencing schemes be reconciled?
- What unit and filtering history can be trusted from each dataset?
- Should the foundation model preserve native montages or require a canonical spatial representation?
- How should continuous, event-driven, clinical, and cognitive tasks share objectives?
- How do we prevent the model from learning dataset/site identity rather than neurophysiology?

### Data quality and bias

- What thresholds define unusable channels or recordings?
- How should quality influence sampling rather than simple exclusion?
- Are demographics, devices, sites, and clinical populations balanced?
- How are duplicated or near-duplicated recordings detected?

### Operations

- How is schema evolution handled without invalidating old checkpoints?
- How are preprocessing failures resumed safely?
- How are corrupt shards quarantined and rebuilt?
- How is the exact dataset manifest associated with every checkpoint?
- What cache size and shard size maximize real cluster throughput?

### Governance

- Which licenses permit commercial foundation-model training?
- How are consent withdrawals and deletion requests propagated?
- Which metadata must be removed or access-controlled?

## 10. Debrief summary

> I would retain the original SHU-MI and HBN files as immutable, versioned sources and implement dataset-specific readers that convert MAT, EDF, BDF, or SET data and BIDS sidecars into a canonical EEG recording schema. Deterministic processing—unit and channel normalization, filtering, resampling, quality control, and windowing—would run as a versioned, restartable pipeline partitioned by recording. Metadata and locations would be stored in Parquet, while dense fixed windows would be materialized in large Arrow or WebDataset-style shards.
>
> Training ranks would receive non-overlapping shuffled shards, prefetch them from object storage into node-local NVMe, decode with persistent workers, and overlap pinned-memory GPU transfer with computation. Kafka can announce new recordings and trigger preprocessing, but object storage plus sharded datasets should serve repeated training epochs. The primary scientific risks are montage/reference differences, data quality, subject leakage, dataset imbalance, and governance.
>
> The submitted prototype exercises this architecture end to end. One size-bundled parallel engine serves SHU-MI MAT, SHU-MI EDF/event, and HBN/BIDS sources. Spawned workers keep private Arrow writers open across deterministic recording bundles and manifest fragments; the coordinator/rank-0 process displays progress, validates and merges outputs deterministically, and records failures and timing. Full SHU-MI feeds both training pipelines through the harmonized backend, the EDF/event path validates continuous ingestion against MAT, and a small HBN/BIDS subset uses the same canonical schema without being mixed into supervised SHU-MI metrics.
>
> The next production-oriented iteration would add compact subject/recording columns, subject-level experiment splits, shape-compatibility buckets, multi-subject shard packing, and record-batch-to-tensor decoding so metadata and signals are batched together without processing each subject separately.
