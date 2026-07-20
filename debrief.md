# Questions to raise during the debrief

## Scientific

- How should source referencing schemes be reconciled?
- What unit and filtering history can be trusted from each dataset?
- Should the foundation model preserve native montages or require a canonical spatial representation?
- How should continuous, event-driven, clinical, and cognitive tasks share objectives?
- How do we prevent the model from learning dataset/site identity rather than neurophysiology?

## Data quality and bias

- What thresholds define unusable channels or recordings?
- How should quality influence sampling rather than simple exclusion?
- Are demographics, devices, sites, and clinical populations balanced?
- How are duplicated or near-duplicated recordings detected?

## Operations

- How is schema evolution handled without invalidating old checkpoints?
- How are preprocessing failures resumed safely?
- How are corrupt shards quarantined and rebuilt?
- How is the exact dataset manifest associated with every checkpoint?
- What cache size and shard size maximize real cluster throughput?

## Governance

- Which licenses permit commercial foundation-model training?
- How are consent withdrawals and deletion requests propagated?
- Which metadata must be removed or access-controlled?

## Summary

> I would retain the original SHU-MI and HBN files as immutable, versioned sources and implement dataset-specific readers that convert MAT, EDF, BDF, or SET data and BIDS sidecars into a canonical EEG recording schema. Deterministic processing—unit and channel normalization, filtering, resampling, quality control, and windowing—would run as a versioned, restartable pipeline partitioned by recording. Metadata and locations would be stored in Parquet, while dense fixed windows would be materialized in large Arrow or WebDataset-style shards.
>
> Training ranks would receive non-overlapping shuffled shards, prefetch them from object storage into node-local NVMe, decode with persistent workers, and overlap pinned-memory GPU transfer with computation. Kafka can announce new recordings and trigger preprocessing, but object storage plus sharded datasets should serve repeated training epochs. The primary scientific risks are montage/reference differences, data quality, subject leakage, dataset imbalance, and governance.
>
> The submitted prototype exercises this architecture end to end: full SHU-MI feeds both training pipelines through the harmonized backend, the EDF/event reader reconstructs the MAT examples, and a small HBN/BIDS subset passes through the same canonical schema and storage layer without being mixed into the supervised SHU-MI comparison.
