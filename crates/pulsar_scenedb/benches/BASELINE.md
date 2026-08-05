# `replication_bench` reference numbers

Measured 2026-08-05 on the maintainer's dev machine (Windows 11, `cargo bench
-p pulsar_scenedb --no-default-features --bench replication_bench`, 100
samples per benchmark, criterion defaults). **These are a human reference
point, not a CI regression gate** — see the note at the bottom for why.

| Benchmark | Time (median) |
|---|---|
| `delta_apply/apply_1000` | 11.5 µs |
| `delta_apply/apply_with_scratch_1000` | 11.6 µs |
| `delta_apply/apply_10000` | 117.2 µs |
| `delta_apply/apply_with_scratch_10000` | 117.2 µs |
| `snapshot_capture/capture_full_1000` | 109.6 µs |
| `snapshot_capture/capture_full_10000` | 1.23 ms |
| `replicable_encode_decode/pod_f32x3_encode` | 60.5 ns |
| `replicable_encode_decode/pod_f32x3_decode` | 3.5 ns |
| `replicable_encode_decode/string_encode` | 31.0 ns |
| `replicable_encode_decode/string_decode` | 31.2 ns |
| `replicable_encode_decode/vec_u32x16_encode` | 510.6 ns |
| `replicable_encode_decode/vec_u32x16_decode` | 45.4 ns |

## `apply` vs `apply_with_scratch`: no measurable difference

At both 1,000 and 10,000 entities, `apply_with_scratch` is statistically
indistinguishable from plain `apply` (well within the ~1% noise band). This
is expected, not a regression: since the `Replicable`-based redesign,
`decode_field_value`'s intermediate byte buffer is gone — `Replicable::
replicate_decode` produces the final owned value directly (a `Pod` field
decodes into a stack `MaybeUninit`, not a heap `Vec` at all), so there's no
longer a per-field scratch allocation for `apply_with_scratch` to amortize.
The method is kept for API stability (see its doc comment) but doing real
work through it buys nothing today.

## Why this file is a snapshot, not a CI gate

Hard-failing CI on wall-clock regressions is a well-known false-positive
generator on shared, unpinned-CPU runners (GitHub Actions' `ubuntu-latest`
included) — a run can look "20% slower" purely from noisy-neighbor
scheduling, with zero code change. The `bench` CI job (see
`.github/workflows/rust.yml`) runs these benchmarks on every PR and uploads
the full criterion HTML report as a build artifact so a human can eyeball
trends over time, but it does not fail the build on its own. If you want a
hard regression gate, the reliable way to get one is dedicated,
consistently-provisioned hardware (self-hosted runner or a bare-metal box) —
not a shared cloud runner.

Re-measure and update this table whenever a change specifically targets one
of these code paths' performance.
