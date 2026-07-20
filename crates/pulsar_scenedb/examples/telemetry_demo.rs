//! SceneDB Telemetry Demo — feeds live data to the dashboard.
//!
//! Creates cells with synthetic entity churn and serves snapshots
//! via the TelemetryServer so the Next.js dashboard can display them.
//!
//! Run:  cargo run -p pulsar_scenedb --example telemetry_demo --features telemetry
//! Dash: cargo run -p scenedb_dashboard -- --telemetry-port 8081

use pulsar_scenedb::*;
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
