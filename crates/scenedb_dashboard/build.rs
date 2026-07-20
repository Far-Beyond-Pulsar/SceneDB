use std::path::Path;
use std::process::Command;

fn main() {
    let dashboard = Path::new("dashboard");
    let out = dashboard.join("out");

    // If npm is unavailable, create a placeholder so the crate still compiles.
    if !has_npm() {
        println!("cargo:warning=⚠ npm not found — creating placeholder static site");
        create_placeholder(&out);
        return;
    }

    if !dashboard.join("node_modules").exists() {
        println!("cargo:warning=Installing npm dependencies...");
        if !run_npm(dashboard, &["install", "--silent"]) {
            panic!("npm install failed");
        }
    }

    println!("cargo:warning=Building Next.js static site...");
    if !run_npm(dashboard, &["run", "build"]) {
        panic!("Next.js build failed — see dashboard/.next for errors");
    }

    if !out.join("index.html").exists() {
        panic!("Next.js output missing dashboard/out/index.html after build");
    }

    println!("cargo:warning=✓ Site built: {} files", count_files(&out));

    println!("cargo:rerun-if-changed=dashboard/src");
    println!("cargo:rerun-if-changed=dashboard/package.json");
    println!("cargo:rerun-if-changed=dashboard/next.config.ts");
    println!("cargo:rerun-if-changed=dashboard/tailwind.config.ts");
    println!("cargo:rerun-if-changed=dashboard/tsconfig.json");
    println!("cargo:rerun-if-changed=dashboard/postcss.config.mjs");
}

fn has_npm() -> bool {
    // Try npm directly, then npm.cmd (Windows), then cmd /c npm
    Command::new("npm")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
        || Command::new("npm.cmd")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        || Command::new("cmd")
            .args(["/c", "npm", "--version"])
            .output()
            .is_ok_and(|o| o.status.success())
}

fn run_npm(dir: &Path, args: &[&str]) -> bool {
    // Try npm.cmd first (Windows), then npm, then cmd /c
    if Command::new("npm.cmd")
        .args(args)
        .current_dir(dir)
        .status()
        .is_ok_and(|s| s.success())
    {
        return true;
    }
    if Command::new("npm")
        .args(args)
        .current_dir(dir)
        .status()
        .is_ok_and(|s| s.success())
    {
        return true;
    }
    Command::new("cmd")
        .args(["/c", "npm"])
        .args(args)
        .current_dir(dir)
        .status()
        .is_ok_and(|s| s.success())
}

fn create_placeholder(out: &Path) {
    std::fs::create_dir_all(out.join("_next")).unwrap();
    std::fs::write(
        out.join("index.html"),
        r#"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>SceneDB Dashboard</title><style>
body{margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh}
.container{text-align:center;max-width:480px}
h1{font-size:1.5rem;margin-bottom:.5rem}
p{color:#8b949e;font-size:.875rem}
code{background:#21262d;padding:2px 6px;border-radius:4px;font-size:.75rem}
.hint{border-top:1px solid #30363d;margin-top:1.5rem;padding-top:1rem;font-size:.75rem}
</style></head><body><div class='container'><h1>SceneDB Dashboard</h1><p>Static site not built. Install Node.js + npm, then run <code>cargo build -p scenedb-dashboard</code>.</p><div class='hint'><p>Or build manually:</p><code>cd crates/scenedb_dashboard/dashboard && npm install && npm run build</code></div></div></body></html>"#,
    ).unwrap();
    // Minimal placeholder to avoid rust-embed errors
    std::fs::write(
        out.join("placeholder.txt"),
        "placeholder — run with Node.js to build the real dashboard",
    ).unwrap();
    println!("cargo:warning=Created placeholder at {}", out.display());
}

fn count_files(dir: &Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += count_files(&path);
            } else {
                count += 1;
            }
        }
    }
    count
}
