//! Every piece of process-wide state in this crate is listed, with why a
//! plugin's static copy may keep its own (Pulsar-Native#1083). State a
//! plugin's copy must share with the editor's (component ids, counters,
//! registration tables) goes through `crate::runtime`; anything else would
//! silently split between the editor and its plugins.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `(file under src/, declarations, why)`.
const LISTED: &[(&str, usize, &str)] = &[
    (
        "component.rs",
        2,
        "CID_CACHE: per-copy TypeId -> ComponentId cache, filled from the runtime",
    ),
    (
        "component_methods.rs",
        4,
        "the world method inventory list and the method table, extended through the runtime",
    ),
    (
        "gpu/mod.rs",
        1,
        "the test device shared by one copy's GPU unit tests",
    ),
    (
        "gpu/tier_layout.rs",
        1,
        "per-copy segment layouts by TypeId, derived from the type alone",
    ),
    (
        "gpu/world_mirror.rs",
        6,
        "the GPU mirror inventory lists and their tables, extended through the runtime",
    ),
    (
        "handle.rs",
        1,
        "Handle's RuntimeTypeInfo, immutable once built",
    ),
    (
        "handle_ledger.rs",
        4,
        "the handle ledger inventory list and its table, extended through the runtime; a \
         per-thread scratch buffer",
    ),
    (
        "runtime.rs",
        6,
        "the runtime itself: OWN, ATTACHED, and the state it owns",
    ),
    ("simd.rs", 1, "a constant lookup table"),
    (
        "telemetry.rs",
        2,
        "per-copy query logs; each copy reports its own queries",
    ),
];

fn is_state(line: &str) -> bool {
    let line = line.trim_start();
    if line.starts_with("//") {
        return false;
    }
    if ["thread_local!", "inventory::collect!", "lazy_static!"]
        .iter()
        .any(|needle| line.contains(needle))
    {
        return true;
    }
    let mut rest = line;
    if let Some(after) = rest.strip_prefix("pub") {
        rest = match after.strip_prefix('(') {
            Some(scoped) => scoped.split_once(')').map_or("", |(_, r)| r),
            None => after,
        }
        .trim_start();
    }
    let Some(rest) = rest.strip_prefix("static ") else {
        return false;
    };
    let rest = rest.trim_start();
    let rest = rest.strip_prefix("mut ").unwrap_or(rest);
    rest.starts_with(|c: char| c.is_ascii_uppercase() || c == '_')
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn found() -> BTreeMap<String, usize> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&src, &mut files);
    let mut out = BTreeMap::new();
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        let count = text.lines().filter(|line| is_state(line)).count();
        if count > 0 {
            let rel = file.strip_prefix(&src).unwrap();
            out.insert(rel.to_string_lossy().replace('\\', "/"), count);
        }
    }
    out
}

fn problems(found: &BTreeMap<String, usize>) -> Vec<String> {
    let mut problems = Vec::new();
    for (file, count) in found {
        match LISTED.iter().find(|(f, _, _)| f == file) {
            Some((_, listed, _)) if listed == count => {}
            Some((_, listed, _)) => problems.push(format!(
                "src/{file}: {count} process-wide state declaration(s), {listed} listed"
            )),
            None => problems.push(format!(
                "src/{file}: {count} unlisted process-wide state declaration(s)"
            )),
        }
    }
    for (file, _, _) in LISTED {
        if !found.contains_key(*file) {
            problems.push(format!("src/{file}: listed state is gone; remove it"));
        }
    }
    problems
}

#[test]
fn process_wide_state_goes_through_the_runtime() {
    let problems = problems(&found());
    assert!(
        problems.is_empty(),
        "state a plugin's copy must share goes through crate::runtime \
         (Pulsar-Native#1083); list a per-copy exception with its reason:\n{}",
        problems.join("\n")
    );
}

#[test]
fn unlisted_state_is_reported() {
    assert!(is_state("static NEXT: AtomicU64 = AtomicU64::new(1);"));
    assert!(is_state(
        "    pub(crate) static mut TABLE: [u8; 4] = [0; 4];"
    ));
    assert!(is_state("thread_local! {"));
    assert!(!is_state("fn f() -> &'static str {"));
    assert!(!is_state("// static NOTE: u8 = 0;"));

    let mut found = found();
    *found.entry("component.rs".to_string()).or_default() += 1;
    found.insert("new_state.rs".to_string(), 1);
    assert_eq!(problems(&found).len(), 2);
}
