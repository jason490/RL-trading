# Immutable storage and accumulation

`data/accumulation/backfill.py` retrieves each symbol/year chunk independently. The request for the project is daily SPY/QQQ/IWM/DIA from 2010-01-01 to 2026-10-01 exclusive. A provider capability check rejects unsupported records/intervals before retrieval. Provider timeouts, connection failures and known transient provider errors receive at most three outer attempts; the Alpaca SDK additionally has two bounded rate-limit retries. There is no feed/provider fallback.

## Layout and identity

```text
data/raw/
  cursor-PROVIDER_CONFIG_HASH.json
  CHUNK_HASH/snapshot.json
  FULL_SNAPSHOT_HASH/snapshot.json
  FULL_SNAPSHOT_HASH/normalized.parquet       # optional derived export
data/datasets/
  DATASET_HASH/dataset.json
```

Each canonical snapshot contains normalized frozen bar dictionaries, provenance and source payload/reference. Yahoo snapshots retain the original DataFrame columns including action data; Alpaca snapshots retain SDK response data. The combined snapshot references its immutable source chunks. Synthetic generation itself is the declared source. CSV imports preserve every canonical field; their authority remains the importer's responsibility.

The identity is SHA-256 of canonical sorted compact JSON with nonfinite numbers forbidden. Content is written to a unique temporary file, flushed/fsynced, then linked atomically into its final immutable path. A pre-existing path must have identical bytes. Loading recomputes the hash and rejects corruption. No existing snapshot is revised in place.

The accumulator cursor is mutable operational state, not an experiment input. Its identity includes provider capabilities, the synthetic seed or a file-content hash where applicable. A successfully persisted chunk is recorded atomically before the next request. Retrying a failed job verifies and reuses those chunks. The final bar order is canonical by symbol/session, regardless of retrieval ordering.

## Refresh and incremental use

`accumulate(provider, request, root, verify_calendar=True, refresh=False)` uses exact-request cached chunks. Set `refresh=True` explicitly to retrieve new historical revisions; those create new content hashes and advance the cursor. Existing experiments still reference their original immutable snapshots. Extending the request reuses unchanged year/symbol chunks and fetches new/changed-range chunks.

A symbol/year returning empty or invalid data fails clearly rather than silently substituting another provider. Inspect the provider audit when a requested earlier range is unsupported. Coverage validation reports missing sessions; no arbitrary gap is filled.

`export_parquet(snapshot_path)` derives normalized Parquet from the verified canonical snapshot. Its bytes are not the canonical experiment identity. Training currently consumes the canonical JSON dataset, so Parquet is an optional interchange/research export, not a hidden storage requirement. Repeated export leaves an existing derived file alone.

Keep all generated `data/`, `.runtime/` and `artifacts/` outside Git. Back up immutable snapshots and their manifests together; a cursor alone cannot reproduce data if its referenced chunks are lost. Tests verify repeated writes, checksum rejection, CSV/synthetic interchangeability and resumable reuse without refetching.
