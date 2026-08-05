# Replication fuzz targets

Coverage-guided (libFuzzer) fuzzing for the network-facing byte decoders in
`pulsar_scenedb::replication` — the actual boundary untrusted peer bytes
cross. This complements (doesn't replace) the seeded, bounded property tests
in `replication.rs` itself: those run a fixed few hundred iterations per
`cargo test`; these run for as long as you let them, exploring inputs no
fixed seed would ever generate, guided by code coverage.

| Target | What it fuzzes |
|---|---|
| `delta_from_bytes` | `Delta::from_bytes` — the per-frame state-change wire format |
| `handshake_from_bytes` | `ReplicationRegistry::from_handshake` — the once-per-connection schema exchange |
| `replicable_decode` | Every built-in `Replicable::replicate_decode` impl (`String`, `Vec<T>`, `Option<T>`, `[f32; N]`, Pod scalars) |
| `delta_apply` | The full pipeline: decode a `Delta`, then `apply` it to a real `World` — catches bugs `Delta::from_bytes` alone can't (adversarial-but-well-formed content reaching the archetype/column machinery) |

## Running

```bash
cargo install cargo-fuzz
cargo +nightly fuzz run delta_from_bytes         # runs until Ctrl-C
cargo +nightly fuzz run delta_from_bytes -- -max_total_time=300   # bounded, e.g. for CI
cargo +nightly fuzz run delta_apply -- -max_total_time=300
```

> [!WARNING]
> **These targets cannot be run on Windows.** `cargo fuzz` compiles cleanly
> here (`cargo +nightly check` in this directory works fine — verified), but
> linking fails with `LNK1104: cannot open file
> 'clang_rt.asan_dynamic_runtime_thunk-x86_64.lib'`: libFuzzer requires
> AddressSanitizer's runtime, and the standard Visual Studio 2022 install
> only ships the **x86 (32-bit)** ASan runtime libs by default — `lib\x86\`
> has `clang_rt.asan_dynamic_runtime_thunk-i386.lib`, `lib\x64\` has nothing
> matching. Getting this working would need either the VS Installer's
> "C++ AddressSanitizer" x64 component (if that ships one on your VS
> version) or a separate LLVM/clang distribution with x86_64 compiler-rt —
> neither of which this repo's tooling assumes. **Run these on Linux or
> macOS** (a standard nightly toolchain there has everything libFuzzer
> needs out of the box), or via CI — see `.github/workflows/rust.yml`'s
> `fuzz` job, which runs each target for a bounded time on every PR.

## Adding a new target

```bash
cargo +nightly fuzz add my_target_name
```
then edit `fuzz_targets/my_target_name.rs` and wire it into `Cargo.toml`'s
`[[bin]]` list the same way the existing four are.
