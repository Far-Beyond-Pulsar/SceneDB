//! `scenedb_inspector` -- standalone egui client for
//! `scenedb-inspector-agent` (lives in the Helio repo, since it's the
//! in-process half that a specific host app depends on -- this crate is
//! the generic viewer). Launches a target executable with
//! `SCENEDB_INSPECTOR_SHM` set, attaches to the shared-memory segment the
//! agent publishes into once the target starts, and renders the live
//! `pulsar_scenedb::World` snapshot (archetypes -> columns -> per-row raw
//! hex bytes / reflected JSON) as a tree.
//!
//! Deliberately has zero *compile-time* dependency on `pulsar_scenedb`
//! (even though it now lives in this repo): the wire format is plain JSON
//! (see `scenedb-inspector-agent`), decoded here as a dynamic
//! `serde_json::Value` rather than typed structs, so this client never
//! needs compile-time knowledge of any specific component type and stays
//! usable against any `WorldSnapshot` shape, including future fields.
//!
//! `ring.rs` is a byte-for-byte copy of `scenedb-inspector-agent`'s own
//! copy -- see that file's doc comment for why it's duplicated rather than
//! shared via a crate dependency (the agent lives in the Helio repo, a
//! separate git repository from this one).
//!
//! Usage: `scenedb_inspector [target-exe] [target-args...]`. The target
//! path/args can also be typed into the toolbar and relaunched from there.

mod ring;

use std::process::Child;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use ring::RingView;
use shared_memory::Shmem;

fn main() -> eframe::Result<()> {
    env_logger::try_init().ok();

    let mut args = std::env::args().skip(1);
    let target_path = args.next().unwrap_or_default();
    let target_args: Vec<String> = args.collect();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1450.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SceneDB Inspector",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(InspectorApp::new(target_path, target_args)) as Box<dyn eframe::App>)
        }),
    )
}

/// A live target process + its shared-memory attachment. `shmem` must
/// outlive `view` (whose `RingView` holds a raw pointer derived from it) --
/// both are dropped together with this struct, `view` first by declaration
/// order, though `RingView` has no `Drop` of its own so the order isn't
/// actually load-bearing today; kept for clarity if that ever changes.
struct Session {
    child: Child,
    shm_name: String,
    view: Option<RingView>,
    #[allow(dead_code)] // kept alive only to keep the mapping valid for `view`
    shmem: Option<Shmem>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // Launch-only design: this client owns the process it started, so
        // closing/relaunching cleans it up rather than leaving it orphaned.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct InspectorApp {
    target_path: String,
    target_args: String,
    session: Option<Session>,
    status: Option<String>,
    is_error: bool,
    last_snapshot: Option<serde_json::Value>,
    last_poll: Instant,
    selected_archetype: usize,
    selected_column: usize,
    row_filter: String,
    /// Row index within the selected column, shown in the detail pane
    /// below the (virtualized) row list. `None` selects nothing.
    selected_row: Option<usize>,
    gpu_tab: bool,
    selected_gpu_buffer: Option<String>,
    gpu_zoom: f32,
    gpu_pan: egui::Vec2,
    selected_gpu_cell: Option<usize>,
}

impl InspectorApp {
    fn new(target_path: String, target_args: Vec<String>) -> Self {
        let mut app = Self {
            target_path,
            target_args: target_args.join(" "),
            session: None,
            status: None,
            is_error: false,
            last_snapshot: None,
            last_poll: Instant::now() - Duration::from_secs(1),
            selected_archetype: 0,
            selected_column: 0,
            row_filter: String::new(),
            selected_row: None,
            gpu_tab: false,
            selected_gpu_buffer: None,
            gpu_zoom: 1.0,
            gpu_pan: egui::Vec2::ZERO,
            selected_gpu_cell: None,
        };
        // Auto-launch when a target was given on the command line, so
        // `scenedb_inspector target.exe` works without an extra click.
        if !app.target_path.trim().is_empty() {
            app.launch();
        }
        app
    }

    fn launch(&mut self) {
        // Dropping the previous session (if any) kills its child first.
        self.session = None;
        self.last_snapshot = None;
        self.status = None;
        self.is_error = false;

        let path = self.target_path.trim();
        if path.is_empty() {
            self.status = Some("no target executable path set".into());
            self.is_error = true;
            return;
        }

        let shm_name = format!(
            "scenedb-inspector-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        );

        let args: Vec<&str> = self.target_args.split_whitespace().collect();
        match std::process::Command::new(path)
            .args(&args)
            .env("SCENEDB_INSPECTOR_SHM", &shm_name)
            .spawn()
        {
            Ok(child) => {
                log::info!("launched '{path}' (pid {}), shm '{shm_name}'", child.id());
                self.status = Some(format!("launched, waiting for '{shm_name}' ..."));
                self.is_error = false;
                self.session = Some(Session {
                    child,
                    shm_name,
                    view: None,
                    shmem: None,
                });
            }
            Err(e) => {
                self.status = Some(format!("failed to launch '{path}': {e}"));
                self.is_error = true;
            }
        }
    }

    fn stop(&mut self) {
        self.session = None; // Drop kills the child.
        self.status = Some("stopped".into());
        self.is_error = false;
    }

    /// Try to attach to the session's shared memory if not already
    /// attached, check child liveness, and (throttled) pull the latest
    /// snapshot. Called once per egui frame.
    fn poll(&mut self) {
        let Some(session) = &mut self.session else {
            return;
        };

        if session.view.is_none() {
            match shared_memory::ShmemConf::new().os_id(&session.shm_name).open() {
                Ok(shmem) => {
                    // SAFETY: we just opened a segment the agent creates
                    // and initializes before this name can be opened at
                    // all; `attach` itself validates magic/version.
                    match unsafe { RingView::attach(shmem.as_ptr()) } {
                        Ok(view) => {
                            log::info!("attached to '{}'", session.shm_name);
                            session.shmem = Some(shmem);
                            session.view = Some(view);
                        }
                        Err(e) => {
                            self.status = Some(e);
                            self.is_error = true;
                        }
                    }
                }
                Err(_) => {
                    // Not created yet -- normal while the target is still
                    // starting up. Nothing to do but wait for a later poll.
                }
            }
        }

        if let Ok(Some(exit_status)) = session.child.try_wait() {
            self.status = Some(format!("target exited: {exit_status}"));
            self.is_error = !exit_status.success();
        }

        if self.last_poll.elapsed() >= Duration::from_millis(150) {
            self.last_poll = Instant::now();
            if let Some(view) = &self.session.as_ref().and_then(|s| s.view.as_ref()) {
                if let Some(bytes) = view.read_latest(4) {
                    match serde_json::from_slice::<serde_json::Value>(&bytes) {
                        Ok(v) => self.last_snapshot = Some(v),
                        Err(e) => log::warn!("scenedb_inspector: snapshot decode failed: {e}"),
                    }
                }
            }
        }
    }
}

/// Render a `serde_json::Value` as a single-line, nesting-aware string
/// (`{x: 1.5, y: 0, nested: {a: 1, b: 2}}`, `[1, 2, 3]`) for the row grid --
/// recurses into objects/arrays instead of dumping raw JSON syntax, since
/// `rows_reflected` entries can nest arbitrarily deep (structs containing
/// structs, `Vec`/`Option` wrappers, enums).
fn format_json_inline(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(format_json_inline).collect();
            format!("[{}]", inner.join(", "))
        }
        serde_json::Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{k}: {}", format_json_inline(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

impl eframe::App for InspectorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        // Keep polling live even while the window is otherwise idle.
        ctx.request_repaint_after(Duration::from_millis(100));

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Target:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.target_path)
                        .hint_text("path\\to\\target.exe")
                        .desired_width(320.0),
                );
                ui.label("Args:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.target_args)
                        .hint_text("space-separated")
                        .desired_width(200.0),
                );
                if ui.button("Launch").clicked() {
                    self.launch();
                }
                if self.session.is_some() && ui.button("Stop").clicked() {
                    self.stop();
                }
                if let Some(session) = &self.session {
                    let attached = session.view.is_some();
                    ui.separator();
                    ui.label(format!(
                        "pid {} | shm '{}' | {}",
                        session.child.id(),
                        session.shm_name,
                        if attached { "attached" } else { "waiting..." }
                    ));
                }
            });
            if let Some(status) = &self.status {
                let color = if self.is_error {
                    egui::Color32::from_rgb(220, 90, 90)
                } else {
                    ui.visuals().weak_text_color()
                };
                ui.colored_label(color, status);
            }
            ui.add_space(4.0);
        });

        let snapshot = self.last_snapshot.clone();
        let Some(snapshot) = snapshot else {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label("No snapshot yet -- launch a target built with scenedb-inspector-agent.");
                });
            });
            return;
        };

        let empty_vec = Vec::new();
        let archetypes = snapshot
            .get("archetypes")
            .and_then(|a| a.as_array())
            .unwrap_or(&empty_vec);
        let entity_count = snapshot.get("entity_count").and_then(|v| v.as_u64()).unwrap_or(0);

        egui::TopBottomPanel::top("view_tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(!self.gpu_tab, "SceneDB").clicked() {
                    self.gpu_tab = false;
                }
                if ui.selectable_label(self.gpu_tab, "SceneDB / GPU + Upload").clicked() {
                    self.gpu_tab = true;
                }
            });
        });

        if self.gpu_tab {
            egui::CentralPanel::default().show(ctx, |ui| show_gpu_tab(ui, &snapshot, &mut self.selected_gpu_buffer, &mut self.gpu_zoom, &mut self.gpu_pan, &mut self.selected_gpu_cell));
            return;
        }
        egui::SidePanel::left("archetypes")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading(format!("{entity_count} entities"));
                ui.label(format!("{} archetypes", archetypes.len()));
                ui.separator();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    for (i, arch) in archetypes.iter().enumerate() {
                        let id = arch.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                        let count = arch
                            .get("entity_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let columns = arch
                            .get("columns")
                            .and_then(|c| c.as_array())
                            .map(|c| c.len())
                            .unwrap_or(0);
                        let label = format!("archetype {id}  ({count} entities, {columns} columns)");
                        if ui
                            .selectable_label(self.selected_archetype == i, label)
                            .clicked()
                        {
                            self.selected_archetype = i;
                            self.selected_column = 0;
                            self.selected_row = None;
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(arch) = archetypes.get(self.selected_archetype) else {
                ui.label("select an archetype");
                return;
            };
            let empty_vec2 = Vec::new();
            let columns = arch
                .get("columns")
                .and_then(|c| c.as_array())
                .unwrap_or(&empty_vec2);
            let entities = arch
                .get("entities")
                .and_then(|e| e.as_array())
                .cloned()
                .unwrap_or_default();

            ui.horizontal(|ui| {
                for (i, col) in columns.iter().enumerate() {
                    let type_name = col
                        .get("type_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let short = type_name.rsplit("::").next().unwrap_or(type_name);
                    if ui
                        .selectable_label(self.selected_column == i, short)
                        .on_hover_text(type_name)
                        .clicked()
                    {
                        self.selected_column = i;
                        self.selected_row = None;
                    }
                }
            });
            ui.separator();

            let Some(col) = columns.get(self.selected_column) else {
                ui.label("this archetype has no components");
                return;
            };
            let type_name = col.get("type_name").and_then(|v| v.as_str()).unwrap_or("?");
            let element_size = col.get("element_size").and_then(|v| v.as_u64()).unwrap_or(0);
            let component_id = col.get("component_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let empty_vec3 = Vec::new();
            let rows_hex = col
                .get("rows_hex")
                .and_then(|r| r.as_array())
                .unwrap_or(&empty_vec3);
            // Structured, nested-aware view when this component's type is
            // registered with pulsar_reflection; None means only rows_hex
            // is available (see ArchetypeColumnSnapshot's doc comment).
            let rows_reflected = col.get("rows_reflected").and_then(|r| r.as_array());
            let row_count = rows_reflected.map(|r| r.len()).unwrap_or(rows_hex.len());

            ui.label(format!(
                "{type_name}  (component_id {component_id}, {element_size} bytes/row, {row_count} rows{})",
                if rows_reflected.is_some() { ", reflected" } else { ", raw hex only \u{2014} not #[derive(Reflectable)]" }
            ));
            ui.horizontal(|ui| {
                ui.label("Filter (entity bits, decimal):");
                ui.text_edit_singleline(&mut self.row_filter);
            });
            ui.separator();

            // Filtered index list, built once per frame: `show_rows` below
            // virtualizes over row *position* in a dense 0..N range (it
            // needs uniform row height to do the scrollbar math), so
            // filtering has to happen before that, not inside it -- letting
            // `show_rows` iterate every unfiltered row just to `continue`
            // past the ones that don't match would defeat the point.
            let filter = self.row_filter.trim();
            let visible_rows: Vec<usize> = (0..row_count)
                .filter(|&row| {
                    filter.is_empty()
                        || entities
                            .get(row)
                            .and_then(|v| v.as_u64())
                            .is_some_and(|bits| bits.to_string().contains(filter))
                })
                .collect();

            // Top: a plain, fixed-height-per-row (virtualized) list -- only
            // the ~20-30 rows actually scrolled into view ever get an
            // overview string built each frame, regardless of whether the
            // column has 10 rows or 10,000. This is what actually fixes the
            // lag a full CollapsingHeader-per-row list had: that scaled
            // with total row count every frame (laying out every row's
            // header, even collapsed), this scales with viewport height.
            let list_height = ui.available_height() * 0.55;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(list_height)
                .show_rows(ui, 18.0, visible_rows.len(), |ui, range| {
                    for i in range {
                        let row = visible_rows[i];
                        let entity_bits = entities
                            .get(row)
                            .and_then(|v| v.as_u64())
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "?".into());
                        let reflected_value = rows_reflected.and_then(|r| r.get(row));
                        let overview = reflected_value
                            .and_then(extract_overview)
                            .unwrap_or_default();

                        let selected = self.selected_row == Some(row);
                        let label = format!(
                            "#{row}  entity {entity_bits}  \u{2014}  {element_size}B{}{overview}",
                            if overview.is_empty() { "" } else { "  \u{2014}  " }
                        );
                        if ui.selectable_label(selected, label).clicked() {
                            self.selected_row = Some(row);
                        }
                    }
                });

            ui.separator();

            // Bottom: full nested tree for the one selected row only --
            // cheap regardless of how many total rows there are, since it's
            // never built for anything but the single selection.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .id_salt("detail_scroll")
                .show(ui, |ui| match self.selected_row {
                    Some(row) => {
                        ui.label(format!("row #{row}"));
                        // Scoped by row so switching the selection doesn't
                        // carry over another row's per-field expand/collapse
                        // state just because two rows share a field name.
                        ui.push_id(row, |ui| match rows_reflected.and_then(|r| r.get(row)) {
                            Some(value) => show_json_tree(ui, value),
                            None => {
                                let hex = rows_hex.get(row).and_then(|v| v.as_str()).unwrap_or("");
                                ui.monospace(hex);
                            }
                        });
                    }
                    None => {
                        ui.label("select a row above to see its full contents");
                    }
                });
        });
    }
}

fn show_gpu_tab(
    ui: &mut egui::Ui,
    snapshot: &serde_json::Value,
    selected: &mut Option<String>,
    zoom: &mut f32,
    pan: &mut egui::Vec2,
    selected_cell: &mut Option<usize>,
) {
    ui.heading("SceneDB GPU storage");
    let Some(gpu) = snapshot.get("gpu") else {
        ui.colored_label(egui::Color32::YELLOW, "No SceneDB GPU storage snapshot is attached yet.");
        ui.label("The target must publish a WorldSnapshot after creating its SceneGpuStore.");
        return;
    };
    let Some(buffers) = gpu.get("registry_buffers").and_then(|value| value.as_array()) else {
        ui.label("SceneDB has not registered any GPU storage buffers yet.");
        return;
    };
    let mut entries = buffers.to_vec();
    entries.push(synthetic_pixel_buffer());
    if selected.as_ref().map(|name| !entries.iter().any(|b| b.get("name").and_then(|v| v.as_str()) == Some(name))).unwrap_or(true) {
        *selected = buffers.first().and_then(|b| b.get("name")).and_then(|v| v.as_str()).map(str::to_owned);
        *selected_cell = None;
    }
    ui.label(format!("{} registered SceneDB GPU entries", entries.len()));
    ui.separator();

    let selected_buffer = selected.as_deref().and_then(|name| entries.iter().find(|b| b.get("name").and_then(|v| v.as_str()) == Some(name)));

    egui::SidePanel::left("scenedb_gpu_buffers")
        .resizable(true)
        .default_width(220.0)
        .min_width(160.0)
        .show_inside(ui, |ui| {
            ui.heading("Buffers");
            egui::ScrollArea::vertical()
                .id_salt("scenedb_gpu_buffer_list")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for buffer in &entries {
                        let Some(name) = buffer.get("name").and_then(|v| v.as_str()) else { continue };
                        let kind = buffer.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
                        let label_width = ui.available_width();
                        if ui
                            .allocate_ui_with_layout(
                                egui::vec2(label_width, 24.0),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |row_ui| {
                                    row_ui.add(egui::SelectableLabel::new(
                                        selected.as_deref() == Some(name),
                                        format!("{name}  [{kind}]"),
                                    ))
                                },
                            )
                            .inner
                            .clicked()
                        {
                            *selected = Some(name.to_owned());
                            *selected_cell = None;
                        }
                    }
                });
        });

    egui::SidePanel::right("scenedb_gpu_details")
        .resizable(true)
        .default_width(460.0)
        .min_width(300.0)
        .max_width(700.0)
        .show_inside(ui, |ui| {
            ui.heading("Details panel");
            let Some(buffer) = selected_buffer else {
                ui.label("Select a SceneDB GPU entry.");
                return;
            };
            if is_pixel_buffer(buffer) {
                show_pixel_canvas(ui, buffer, zoom, pan);
                return;
            }

            let cells = buffer.get("cells").and_then(|value| value.as_array()).map(Vec::as_slice).unwrap_or(&[]);
            let Some(index) = *selected_cell else {
                ui.label("Select a cell to inspect its reflected contents.");
                return;
            };
            let Some(cell) = cells.iter().find(|cell| cell.get("index").and_then(|v| v.as_u64()) == Some(index as u64)) else {
                ui.label("Selected cell is not in this snapshot.");
                return;
            };
            ui.label(format!("cell {index}"));
            ui.label("reflected value");
            egui::ScrollArea::vertical().id_salt("scenedb_gpu_cell_detail").auto_shrink([false, false]).show(ui, |ui| {
                match cell.get("reflected").filter(|v| !v.is_null()) {
                    Some(value) => { ui.push_id(index, |ui| show_json_tree(ui, value)); },
                    None => { ui.label("No reflection value was published for this row."); }
                }
                ui.separator();
                ui.label("raw bytes");
                ui.monospace(cell.get("bytes_hex").and_then(|v| v.as_str()).unwrap_or(""));
            });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        let Some(buffer) = selected_buffer else {
            ui.label("Select a SceneDB GPU entry.");
            return;
        };
        ui.heading(selected.as_deref().unwrap_or("Buffer"));
        egui::Grid::new("scenedb_gpu_buffer_metadata").striped(true).show(ui, |ui| {
            for key in ["kind", "element_size", "capacity_bytes", "epoch", "access", "mirror_mode", "element_type_name", "cells_truncated"] {
                if let Some(value) = buffer.get(key) {
                    ui.label(key);
                    ui.monospace(format_json_inline(value));
                    ui.end_row();
                }
            }
        });
        ui.separator();
        let element_size = buffer
            .get("element_size")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let cells = buffer.get("cells").and_then(|value| value.as_array()).map(Vec::as_slice).unwrap_or(&[]);
        ui.heading(format!("Cells ({})", cells.len()));
        if cells.is_empty() {
            let raw_chunks = buffer.get("raw_chunks").and_then(|value| value.as_array()).map(Vec::as_slice).unwrap_or(&[]);
            if raw_chunks.is_empty() {
                ui.label("This SceneDB entry has no readable rows.");
            } else {
                ui.label("No row cells; raw bytes are available in Details panel.");
            }
            return;
        }
        if selected_cell.map(|row| row >= cells.len()).unwrap_or(true) {
            *selected_cell = Some(0);
        }
        ui.label(format!("{} cells read back; click a row for details", cells.len()));
        egui::ScrollArea::vertical()
            .id_salt("scenedb_gpu_cells")
            .auto_shrink([false, false])
            .show_rows(ui, 24.0, cells.len(), |ui, range| {
                for position in range {
                    let cell = &cells[position];
                    let index = cell.get("index").and_then(|v| v.as_u64()).unwrap_or(position as u64) as usize;
                    let reflected = cell.get("reflected").filter(|v| !v.is_null());
                    let summary = reflected
                        .and_then(extract_overview)
                        .or_else(|| reflected.map(format_json_inline))
                        .unwrap_or_else(|| "no reflection value".to_owned());

                    let max_summary_chars = ((ui.available_width() - 150.0) / 7.0).max(8.0) as usize;
                    let summary = if summary.chars().count() > max_summary_chars {
                        let mut short = summary.chars().take(max_summary_chars.saturating_sub(1)).collect::<String>();
                        short.push('…');
                        short
                    } else {
                        summary
                    };
                    let mut row_text = egui::text::LayoutJob::default();
                    row_text.append(
                        &format!("cell {index}  ·  {element_size}B  ·  "),
                        0.0,
                        egui::TextFormat::default(),
                    );
                    row_text.append(
                        &summary,
                        0.0,
                        egui::TextFormat {
                            color: ui.visuals().weak_text_color(),
                            ..Default::default()
                        },
                    );
                    let row_width = ui.available_width();
                    if ui
                        .allocate_ui_with_layout(
                            egui::vec2(row_width, 24.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |row_ui| {
                                row_ui.add(egui::SelectableLabel::new(
                                    *selected_cell == Some(index),
                                    row_text,
                                ))
                            },
                        )
                        .inner
                        .clicked()
                    {
                        *selected_cell = Some(index);
                    }
                }
            });
    });
}
fn synthetic_pixel_buffer() -> serde_json::Value {
    let width = 32usize;
    let height = 32usize;
    let cells = (0..width * height).map(|i| {
        let x = i % width;
        let y = i / width;
        serde_json::json!({
            "index": i,
            "bytes_hex": format!("{:02x}{:02x}{:02x}ff", x * 8, y * 8, ((x + y) * 4) % 256),
            "reflected": [x * 8, y * 8, ((x + y) * 4) % 256, 255]
        })
    }).collect::<Vec<_>>();
    serde_json::json!({
        "name": "inspector_test_rgba8 [synthetic]",
        "kind": "pixel",
        "element_size": 4,
        "capacity_bytes": width * height * 4,
        "epoch": 0,
        "access": "InspectorOnly",
        "mirror_mode": null,
        "element_type_name": "SyntheticRgba8",
        "width": width,
        "height": height,
        "cells": cells,
        "cells_truncated": false,
        "raw_chunks": []
    })
}
fn is_pixel_buffer(buffer: &serde_json::Value) -> bool {
    let name = buffer.get("name").and_then(|v| v.as_str()).unwrap_or("").to_ascii_lowercase();
    let ty = buffer.get("element_type_name").and_then(|v| v.as_str()).unwrap_or("").to_ascii_lowercase();
    ["pixel", "rgba", "bgra", "color", "texel", "texture"].iter().any(|needle| name.contains(needle) || ty.contains(needle))
}

fn show_pixel_canvas(ui: &mut egui::Ui, buffer: &serde_json::Value, zoom: &mut f32, pan: &mut egui::Vec2) {
    ui.heading("Pixel canvas");
    ui.label("Scroll over the canvas to zoom; drag to pan. Double-click resets the view.");
    let Some(cells) = buffer.get("cells").and_then(|v| v.as_array()) else { ui.label("No pixel cells were read back."); return; };
    if cells.is_empty() { ui.label("No pixel cells were read back."); return; }
    let width = buffer.get("width").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or_else(|| (cells.len() as f32).sqrt().ceil() as usize).max(1);
    let height = buffer.get("height").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or_else(|| (cells.len() + width - 1) / width).max(1);
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    if response.double_clicked() { *zoom = 1.0; *pan = egui::Vec2::ZERO; }
    if response.dragged() { *pan += response.drag_delta(); }
    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
    if response.hovered() && scroll.abs() > f32::EPSILON {
        let old_zoom = *zoom;
        *zoom = (*zoom * (1.0 + scroll * 0.001)).clamp(0.25, 32.0);
        if let Some(cursor) = response.hover_pos() {
            let cursor_local = cursor - rect.min;
            let content_at_cursor = (cursor_local - *pan) / old_zoom;
            *pan = cursor_local - content_at_cursor * *zoom;
        }
    }
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(24));
    let origin = rect.min + *pan;
    for (i, cell) in cells.iter().enumerate() {
        let x = i % width; let y = i / width; if y >= height { break; }
        let color = pixel_color(cell).unwrap_or(egui::Color32::from_gray(90));
        let size = (*zoom).max(1.0);
        painter.rect_filled(egui::Rect::from_min_size(origin + egui::vec2(x as f32 * size, y as f32 * size), egui::vec2(size, size)), 0.0, color);
    }
}

fn pixel_color(cell: &serde_json::Value) -> Option<egui::Color32> {
    let reflected = cell.get("reflected").filter(|v| !v.is_null())?;
    let channels = reflected.as_array().or_else(|| reflected.get("channels").and_then(|v| v.as_array()))?;
    let c: Vec<u8> = channels.iter().take(4).map(|v| (v.as_f64().unwrap_or(0.0).clamp(0.0, 255.0)) as u8).collect();
    (c.len() >= 3).then(|| egui::Color32::from_rgba_premultiplied(c[0], c[1], c[2], *c.get(3).unwrap_or(&255)))
}
/// One-line overview for a row's collapsed header: a position/bounds-ish
/// field when one can be found by common naming (or, failing that, a
/// translation column pulled out of a 4x4 `transform` matrix), so the
/// collapsed list still reads as "where is this thing" at a glance instead
/// of needing to expand every row just to tell entities apart.
fn extract_overview(value: &serde_json::Value) -> Option<String> {
    let obj = value.as_object()?;

    for key in ["position", "pos", "translation"] {
        if let Some(v) = obj.get(key) {
            return Some(format!("pos {}", format_json_inline(v)));
        }
    }

    // Column-major 4x4: the translation is the last column's first 3 lanes.
    if let Some(serde_json::Value::Array(cols)) = obj.get("transform") {
        if let Some(serde_json::Value::Array(last_col)) = cols.get(3) {
            let xyz: Vec<String> = last_col.iter().take(3).map(format_json_inline).collect();
            return Some(format!("pos [{}]", xyz.join(", ")));
        }
    }

    if let Some(v) = obj.get("bounds") {
        return Some(format!("bounds {}", format_json_inline(v)));
    }

    None
}

/// Render a JSON object's fields as an indented, individually-collapsible
/// tree -- nested objects/arrays of objects get their own `CollapsingHeader`
/// (collapsed by default), leaves render inline as `key: value`. Scope the
/// call site in `ui.push_id` (per row) so identical field names across
/// different rows/columns don't collide on the same egui `Id`.
fn show_json_tree(ui: &mut egui::Ui, value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, v) in map {
                show_json_field(ui, key, v);
            }
        }
        other => {
            ui.monospace(format_json_inline(other));
        }
    }
}

fn show_json_field(ui: &mut egui::Ui, key: &str, value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) if !map.is_empty() => {
            egui::CollapsingHeader::new(key)
                .default_open(false)
                .show(ui, |ui| {
                    for (k, v) in map {
                        show_json_field(ui, k, v);
                    }
                });
        }
        serde_json::Value::Array(items)
            if !items.is_empty() && items.iter().any(|v| v.is_object() || v.is_array()) =>
        {
            egui::CollapsingHeader::new(format!("{key}  [{}]", items.len()))
                .default_open(false)
                .show(ui, |ui| {
                    for (i, v) in items.iter().enumerate() {
                        show_json_field(ui, &format!("[{i}]"), v);
                    }
                });
        }
        other => {
            ui.horizontal(|ui| {
                ui.label(format!("{key}:"));
                ui.monospace(format_json_inline(other));
            });
        }
    }
}
