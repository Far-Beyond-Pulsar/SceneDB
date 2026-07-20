//! SceneDB Telemetry Demo — feeds live data to the dashboard.
//!
//! Creates cells with synthetic entity churn and serves snapshots
//! via the TelemetryServer so the Next.js dashboard can display them.
//!
//! Run:  cargo run -p pulsar_scenedb --example telemetry_demo --features telemetry
//! Dash: cargo run -p scenedb_dashboard -- --telemetry-port 8081

use pulsar_scenedb::*;
use rand::Rng;
use std::time::{Duration, Instant};

fn main() {
    let server = TelemetryServer::start(8081).expect("telemetry server on 8081");

    let caps: [u32; 6] = [64, 96, 128, 160, 192, 256];
    let mut cells: Vec<CellStorage> = caps
        .iter()
        .map(|&cap| {
            CellStorage::new(&[ColumnDesc::of::<f32>(), ColumnDesc::of::<f64>()], cap).unwrap()
        })
        .collect();

    let mut handles: Vec<Vec<Handle>> = cells.iter().map(|_| Vec::new()).collect();

    // Spatial cell for query logging
    let mut spatial = SpatialCell::new(256).unwrap();
    spatial.set_telemetry_cell_id(10);
    let mut rng = rand::thread_rng();
    for _ in 0..200 {
        let min = [rng.gen::<f32>() * 100.0 - 50.0; 3];
        let max = [min[0] + rng.gen::<f32>() * 20.0 + 0.1, min[1] + rng.gen::<f32>() * 20.0 + 0.1, min[2] + rng.gen::<f32>() * 20.0 + 0.1];
        spatial.alloc(Aabb { min, max });
    }
    let mut out: Vec<u32> = vec![0u32; 256];
    let mut liveness_words: Vec<u64> = vec![0u64; 4];

    // Seed cells
    for (i, cell) in cells.iter_mut().enumerate() {
        for _ in 0..caps[i] / 2 {
            if let Some(h) = cell.alloc() {
                if let Some(row) = cell.row_of(h) {
                    if let Some(col) = cell.column_for_mut::<f32>() {
                        col[row as usize] = i as f32 * 10.0;
                    }
                    if let Some(col) = cell.column_for_mut::<f64>() {
                        col[row as usize] = (i * 100) as f64;
                    }
                }
                handles[i].push(h);
            }
        }
    }

    println!("Telemetry demo — 6 cells, serving on port 8081");
    println!("Run: cargo run -p scenedb_dashboard");

    let mut frame = 0u64;
    loop {
        let t0 = Instant::now();

        for (i, cell) in cells.iter_mut().enumerate() {
            let hs = &mut handles[i];
            let cap = caps[i];

            // Alloc up to 75%
            if hs.len() < (cap as usize * 3 / 4) {
                if let Some(h) = cell.alloc() {
                    if let Some(row) = cell.row_of(h) {
                        if let Some(col) = cell.column_for_mut::<f32>() {
                            col[row as usize] = frame as f32 * 0.01;
                        }
                        if let Some(col) = cell.column_for_mut::<f64>() {
                            col[row as usize] = (frame % 1000) as f64;
                        }
                    }
                    hs.push(h);
                }
            }

            // Free some
            let to_free = (hs.len() / 8).max(1);
            for _ in 0..to_free {
                if let Some(h) = hs.pop() {
                    cell.free(h);
                }
            }
        }

        // Compact every 30 frames
        if frame % 30 == 0 {
            for cell in cells.iter_mut() {
                cell.compact();
            }
            // Re-map handles after compaction
            for (i, cell) in cells.iter().enumerate() {
                handles[i].retain(|&h| cell.row_of(h).is_some());
            }
        }

        // Run spatial queries every 4 frames to populate query log
        if frame % 4 == 0 {
            let len = spatial.storage().rows_in_use() as usize;
            let nw = len.div_ceil(64);
            let words = spatial.storage().liveness().words();
            for (i, w) in words.iter().enumerate().take(nw) {
                liveness_words[i] = w.load(std::sync::atomic::Ordering::Relaxed);
            }
            // AABB query
            let q = Aabb { min: [-20.0; 3], max: [20.0; 3] };
            let _ = spatial.query_aabb_in(&q, &liveness_words[..nw], &mut out[..len]);
            // Frustum query
            let planes = [[1.0,0.0,0.0,50.0],[-1.0,0.0,0.0,50.0],[0.0,1.0,0.0,50.0],[0.0,-1.0,0.0,50.0],[0.0,0.0,1.0,50.0],[0.0,0.0,-1.0,50.0]];
            let _ = spatial.query_frustum_in(&Frustum { planes }, &liveness_words[..nw], &mut out[..len]);
        }

        // Push snapshot
        let cell_pairs: Vec<(u32, &CellStorage)> =
            cells.iter().enumerate().map(|(id, c)| (id as u32, c)).collect();
        server.push_snapshot(TelemetrySnapshot::from_cells(&cell_pairs));

        let elapsed = t0.elapsed();
        let sleep = Duration::from_millis(66).saturating_sub(elapsed); // ~15 fps
        if sleep > Duration::ZERO {
            std::thread::sleep(sleep);
        }

        frame += 1;
        if frame % 40 == 0 {
            let total_rows: u32 = cells.iter().map(|c| c.rows_in_use()).sum();
            println!(
                "Frame {frame}: {} cells, {total_rows} rows, {:.0}ms/frame",
                cells.len(),
                elapsed.as_secs_f64() * 1000.0,
            );
        }

        if frame >= 60000 {
            break;
        }
    }

    drop(server);
    println!("Telemetry demo stopped.");
}
