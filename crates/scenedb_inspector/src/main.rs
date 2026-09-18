//! `scenedb_inspector` -- standalone egui client for
//! `scenedb-inspector-agent` (lives in the Helio repo, since it's the
//! in-process half that a specific host app depends on -- this crate is
//! the generic viewer). Launches a target executable with
//! `SCENEDB_INSPECTOR_SHM` set, attaches to the shared-memory segment the
//! agent publishes into once the target starts, and renders the live
//! `pulsar_scenedb::World` state as standard ECS-inspector panes:
//! archetypes | entities | components | details.
//!
//! The agent only publishes *metadata* (`telemetry_snapshot_metadata`), so
//! entity bits and row values are fetched on demand through
//! `InspectorCpuRequest` windows. Because the shared-memory bridge delivers
//! only the latest request, the client batches every window it needs for
//! the current selection into one request's `cpu_ranges` (entity viewport
//! plus the selected entity's row across all columns) and splices each
//! returned range back into the snapshot.
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
    detail_cache: Vec<(String, serde_json::Value)>,
    last_poll: Instant,
    selected_archetype: usize,
    /// Archetype row index identifying the selected entity. Every column in
    /// an archetype stores the same rows in the same order, so a single
    /// index addresses that entity across all of its components.
    selected_entity: Option<usize>,
    /// Index into the selected archetype's `columns` for the component
    /// whose value is shown in the details pane.
    selected_component: Option<usize>,
    entity_filter: String,
    gpu_tab: bool,
    selected_gpu_buffer: Option<String>,
    gpu_textures_tab: bool,
    gpu_zoom: f32,
    gpu_pan: egui::Vec2,
    selected_gpu_cell: Option<usize>,
    gpu_visible_start: usize,
    gpu_visible_count: usize,
    request_slot: u32,
    next_request_id: u64,
    last_request_key: Option<String>,
    entity_visible_start: usize,
    entity_visible_count: usize,
}

fn cpu_range_key(cpu: &serde_json::Value) -> String {
    format!(
        "cpu:{}:{}:{}",
        cpu.get("archetype_id").and_then(|value| value.as_u64()).unwrap_or(0),
        cpu.get("component_id").and_then(|value| value.as_u64()).unwrap_or(0),
        cpu.get("row_start").and_then(|value| value.as_u64()).unwrap_or(0),
    )
}

fn detail_cache_key(response: &serde_json::Value) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(cpu) = response.get("cpu").filter(|value| value.is_object()) {
        parts.push(cpu_range_key(cpu));
    }
    if let Some(ranges) = response.get("cpu_ranges").and_then(|value| value.as_array()) {
        for cpu in ranges {
            parts.push(cpu_range_key(cpu));
        }
    }
    if let Some(gpu) = response.get("gpu").filter(|value| value.is_object()) {
        parts.push(format!(
            "gpu:{}:{}:{}:{}",
            gpu.get("buffer").and_then(|value| value.as_str()).unwrap_or(""),
            gpu.get("byte_offset").and_then(|value| value.as_u64()).unwrap_or(0),
            gpu.get("byte_len").and_then(|value| value.as_u64()).unwrap_or(0),
            gpu.get("cell").and_then(|value| value.as_u64()).unwrap_or(usize::MAX as u64),
        ));
    }
    (!parts.is_empty()).then(|| parts.join("|"))
}

fn decode_hex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks(2)
        .filter_map(|pair| {
            if pair.len() != 2 { return None; }
            let high = (pair[0] as char).to_digit(16)? as u8;
            let low = (pair[1] as char).to_digit(16)? as u8;
            Some((high << 4) | low)
        })
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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
            detail_cache: Vec::new(),
            last_poll: Instant::now() - Duration::from_secs(1),
            selected_archetype: 0,
            selected_entity: None,
            selected_component: None,
            entity_filter: String::new(),
            gpu_tab: false,
            selected_gpu_buffer: None,
            gpu_textures_tab: false,
            gpu_zoom: 1.0,
            gpu_pan: egui::Vec2::ZERO,
            selected_gpu_cell: None,
            gpu_visible_start: 0,
            gpu_visible_count: 16,
            request_slot: 0,
            next_request_id: 1,
            last_request_key: None,
            entity_visible_start: 0,
            entity_visible_count: 32,
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
        self.detail_cache.clear();
        self.last_request_key = None;
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

        let args = split_args(&self.target_args);
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
                        Ok(v) if v.get("kind").and_then(|v| v.as_str()) == Some("detail") => {
                            self.remember_detail(&v);
                            self.merge_detail(v);
                        }
                        Ok(v) => {
                            self.last_snapshot = Some(v);
                            for detail in self.detail_cache.iter().map(|(_, detail)| detail.clone()).collect::<Vec<_>>() {
                                self.merge_detail(detail);
                            }
                            // Metadata changes are the live-refresh boundary;
                            // ask for the same visible/selected details again
                            // against the new world state.
                            self.last_request_key = None;
                        },
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
        self.request_current_details();
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
            egui::CentralPanel::default().show(ctx, |ui| show_gpu_tab(ui, &snapshot, &mut self.selected_gpu_buffer, &mut self.gpu_textures_tab, &mut self.gpu_zoom, &mut self.gpu_pan, &mut self.selected_gpu_cell, &mut self.gpu_visible_start, &mut self.gpu_visible_count));
            return;
        }
        egui::SidePanel::left("archetypes")
            .resizable(true)
            .default_width(240.0)
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
                        let label = format!("archetype {id}  ({count} entities, {columns} components)");
                        if ui
                            .selectable_label(self.selected_archetype == i, label)
                            .clicked()
                        {
                            self.selected_archetype = i;
                            self.selected_entity = None;
                            self.selected_component = None;
                        }
                    }
                });
            });

        egui::SidePanel::left("entities")
            .resizable(true)
            .default_width(300.0)
            .min_width(180.0)
            .show(ctx, |ui| {
                let Some(arch) = archetypes.get(self.selected_archetype) else {
                    ui.label("select an archetype");
                    return;
                };
                let empty_entities = Vec::new();
                let entity_count = arch.get("entity_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                // Entity bits are archetype-wide: every column stores the
                // same rows in the same order, so any one column's injected
                // `entities` array is the archetype's entity list. Use the
                // first. (The agent publishes metadata only, so this array
                // arrives via the viewport request below and is spliced in
                // by `merge_cpu_response`.)
                let entity_bits = arch
                    .get("columns")
                    .and_then(|c| c.as_array())
                    .and_then(|c| c.first())
                    .and_then(|c| c.get("entities"))
                    .and_then(|e| e.as_array())
                    .unwrap_or(&empty_entities);
                ui.heading(format!("{entity_count} entities"));
                ui.horizontal(|ui| {
                    ui.label("Filter:");
                    ui.text_edit_singleline(&mut self.entity_filter);
                });
                ui.separator();
                let filter = self.entity_filter.trim().to_string();
                let visible_rows: Vec<usize> = (0..entity_count)
                    .filter(|&row| {
                        filter.is_empty()
                            || entity_bits
                                .get(row)
                                .and_then(|v| v.as_u64())
                                .is_some_and(|bits| bits.to_string().contains(&filter))
                    })
                    .collect();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show_rows(ui, 18.0, visible_rows.len(), |ui, range| {
                        if let Some(&first) = visible_rows.get(range.start) {
                            self.entity_visible_start = first;
                            self.entity_visible_count = visible_rows
                                .get(range.end.saturating_sub(1))
                                .map(|last| last.saturating_sub(first) + 1)
                                .unwrap_or(range.len());
                        }
                        for i in range {
                            let row = visible_rows[i];
                            let bits = entity_bits.get(row).and_then(|v| v.as_u64());
                            let label = match bits {
                                Some(bits) => format!("#{row}  entity {bits}"),
                                None => format!("#{row}  entity \u{2026}"),
                            };
                            if ui
                                .selectable_label(self.selected_entity == Some(row), label)
                                .clicked()
                            {
                                self.selected_entity = Some(row);
                            }
                        }
                    });
            });

        egui::SidePanel::right("details")
            .resizable(true)
            .default_width(460.0)
            .min_width(260.0)
            .max_width(820.0)
            .show(ctx, |ui| {
                let Some(arch) = archetypes.get(self.selected_archetype) else {
                    ui.label("select an archetype");
                    return;
                };
                let empty_columns = Vec::new();
                let columns = arch.get("columns").and_then(|c| c.as_array()).unwrap_or(&empty_columns);
                let Some(component_index) = self.selected_component else {
                    ui.heading("Details");
                    ui.label("Select a component to inspect its value.");
                    return;
                };
                let Some(col) = columns.get(component_index) else {
                    ui.label("component out of range");
                    return;
                };
                let type_name = col.get("type_name").and_then(|v| v.as_str()).unwrap_or("?");
                let short = type_name.rsplit("::").next().unwrap_or(type_name);
                let element_size = col.get("element_size").and_then(|v| v.as_u64()).unwrap_or(0);
                let component_id = col.get("component_id").and_then(|v| v.as_u64()).unwrap_or(0);
                ui.heading(short);
                ui.label(format!(
                    "{type_name}  (component_id {component_id}, {element_size} bytes/row)"
                ));
                ui.separator();
                let Some(row) = self.selected_entity else {
                    ui.label("Select an entity to see this component's value.");
                    return;
                };
                ui.label(format!("row #{row}"));
                let reflected = col.get("rows_reflected").and_then(|r| r.get(row));
                let hex = col
                    .get("rows_hex")
                    .and_then(|r| r.get(row))
                    .and_then(|v| v.as_str())
                    .filter(|text| !text.is_empty());
                egui::ScrollArea::vertical()
                    .id_salt("entity_component_detail")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Scoped by (component, row) so switching the
                        // selection doesn't carry over another value's
                        // per-field expand/collapse state just because two
                        // values share a field name.
                        ui.push_id((component_index, row), |ui| match reflected {
                            Some(value) if !value.is_null() => show_json_tree(ui, value),
                            _ => match hex {
                                Some(hex) => {
                                    ui.monospace(hex);
                                }
                                None => {
                                    ui.label("value not fetched yet \u{2026}");
                                }
                            },
                        });
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(arch) = archetypes.get(self.selected_archetype) else {
                ui.label("select an archetype");
                return;
            };
            let empty_columns = Vec::new();
            let columns = arch.get("columns").and_then(|c| c.as_array()).unwrap_or(&empty_columns);
            ui.heading("Components");
            let Some(row) = self.selected_entity else {
                ui.label("Select an entity to list its components.");
                ui.label(
                    "Every entity in an archetype has the same component set; \
                     the values shown are for the selected row.",
                );
                return;
            };
            ui.label(format!("row #{row}"));
            ui.separator();
            if columns.is_empty() {
                ui.label("this archetype has no components");
                return;
            }
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                for (i, col) in columns.iter().enumerate() {
                    let type_name = col.get("type_name").and_then(|v| v.as_str()).unwrap_or("?");
                    let short = type_name.rsplit("::").next().unwrap_or(type_name);
                    let overview = col
                        .get("rows_reflected")
                        .and_then(|r| r.get(row))
                        .filter(|v| !v.is_null())
                        .and_then(extract_overview);
                    let label = match overview {
                        Some(overview) => format!("{short}  \u{2014}  {overview}"),
                        None => short.to_string(),
                    };
                    if ui
                        .selectable_label(self.selected_component == Some(i), label)
                        .on_hover_text(type_name)
                        .clicked()
                    {
                        self.selected_component = Some(i);
                    }
                }
            });
        });
    }
}

fn show_gpu_tab(
    ui: &mut egui::Ui,
    snapshot: &serde_json::Value,
    selected: &mut Option<String>,
    textures_tab: &mut bool,
    zoom: &mut f32,
    pan: &mut egui::Vec2,
    selected_cell: &mut Option<usize>,
    visible_start: &mut usize,
    visible_count: &mut usize,
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
    let mut entries = if *textures_tab {
        gpu.get("textures").and_then(|value| value.as_array()).map(|textures| textures.iter().map(texture_entry).collect()).unwrap_or_default()
    } else {
        buffers.to_vec()
    };
    if !*textures_tab { entries.push(synthetic_pixel_buffer()); }
    if selected.as_ref().map(|name| !entries.iter().any(|b| b.get("name").and_then(|v| v.as_str()) == Some(name))).unwrap_or(true) {
        *selected = entries.first().and_then(|b| b.get("name")).and_then(|v| v.as_str()).map(str::to_owned);
        *selected_cell = None;
        *visible_start = 0;
        *visible_count = 16;
    }
    ui.label(format!("{} registered SceneDB GPU entries", entries.len()));
    ui.separator();

    let selected_buffer = selected.as_deref().and_then(|name| entries.iter().find(|b| b.get("name").and_then(|v| v.as_str()) == Some(name)));

    egui::SidePanel::left("scenedb_gpu_buffers")
        .resizable(true)
        .default_width(220.0)
        .min_width(160.0)
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(!*textures_tab, "Buffers").clicked() { *textures_tab = false; *selected = None; *selected_cell = None; *visible_start = 0; *visible_count = 16; }
                if ui.selectable_label(*textures_tab, "Textures").clicked() { *textures_tab = true; *selected = None; *selected_cell = None; *visible_start = 0; *visible_count = 16; }
            });
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
                            *visible_start = 0;
                            *visible_count = 16;
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
        if element_size != 0 {
            let capacity_bytes = buffer
                .get("capacity_bytes")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let max_cell = capacity_bytes
                .checked_div(element_size)
                .unwrap_or(0)
                .saturating_sub(1);
            let mut cell_index = (*selected_cell).unwrap_or(0) as u64;
            cell_index = cell_index.min(max_cell);
            ui.horizontal(|ui| {
                ui.label("Request cell");
                if ui
                    .add(egui::DragValue::new(&mut cell_index).range(0..=max_cell))
                    .changed()
                {
                    *selected_cell = Some(cell_index as usize);
                }
            });
        }
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
        if selected_cell
            .map(|row| !cells.iter().any(|cell| cell.get("index").and_then(|v| v.as_u64()) == Some(row as u64)))
            .unwrap_or(true)
        {
            *selected_cell = cells
                .first()
                .and_then(|cell| cell.get("index").and_then(|v| v.as_u64()))
                .map(|index| index as usize);
        }
        ui.label(format!("{} cells read back; click a row for details", cells.len()));
        egui::ScrollArea::vertical()
            .id_salt("scenedb_gpu_cells")
            .auto_shrink([false, false])
            .show_rows(ui, 24.0, cells.len(), |ui, range| {
                let new_start = range.start;
                let new_count = range.len().max(1);
                if new_start != *visible_start || new_count != *visible_count {
                    *visible_start = new_start;
                    *visible_count = new_count;
                    // Scrolling resumes viewport range reads; a selected cell
                    // remains an explicit granular-detail request only until
                    // the user moves the viewport again.
                    *selected_cell = None;
                }
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
fn texture_entry(texture: &serde_json::Value) -> serde_json::Value {
    let slot = texture.get("slot").and_then(|v| v.as_u64()).unwrap_or(0);
    serde_json::json!({ "name": format!("builtin_texture[{slot}]"), "kind": "texture", "element_size": 0, "capacity_bytes": null, "epoch": null, "access": "ReadOnly", "mirror_mode": null, "element_type_name": texture.get("format").cloned().unwrap_or(serde_json::Value::Null), "cells": [], "cells_truncated": false, "raw_chunks": [], "texture": texture })
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

impl InspectorApp {
    fn remember_detail(&mut self, response: &serde_json::Value) {
        let Some(key) = detail_cache_key(response) else { return; };
        if let Some((_, cached)) = self.detail_cache.iter_mut().find(|(cached_key, _)| cached_key == &key) {
            *cached = response.clone();
            return;
        }
        if self.detail_cache.len() >= 64 {
            self.detail_cache.remove(0);
        }
        self.detail_cache.push((key, response.clone()));
    }

    fn merge_detail(&mut self, response: serde_json::Value) {
        let Some(snapshot) = self.last_snapshot.as_mut() else { return; };
        if let Some(cpu) = response.get("cpu") {
            merge_cpu_response(snapshot, cpu);
        }
        if let Some(ranges) = response.get("cpu_ranges").and_then(|v| v.as_array()) {
            for cpu in ranges {
                merge_cpu_response(snapshot, cpu);
            }
        }
        if let Some(gpu) = response.get("gpu") {
            if let Some(name) = gpu.get("buffer").and_then(|v| v.as_str()) {
                if let Some(buffer) = snapshot.get_mut("gpu").and_then(|v| v.get_mut("registry_buffers")).and_then(|v| v.as_array_mut()).and_then(|a| a.iter_mut().find(|b| b.get("name").and_then(|v| v.as_str()) == Some(name))) {
                    let byte_offset = gpu.get("byte_offset").and_then(|v| v.as_u64()).unwrap_or(0);
                    let bytes_hex = gpu.get("bytes_hex").and_then(|v| v.as_str()).unwrap_or("");
                    buffer["raw_chunks"] = serde_json::json!([{ "offset": byte_offset, "bytes_hex": bytes_hex }]);
                    if let Some(cell) = gpu.get("cell").and_then(|v| v.as_u64()) {
                        buffer["cells"] = serde_json::json!([{ "index": cell, "bytes_hex": bytes_hex, "reflected": null }]);
                    } else if let Some(element_size) = buffer.get("element_size").and_then(|v| v.as_u64()).filter(|size| *size != 0) {
                        let bytes = decode_hex(bytes_hex);
                        let first_cell = byte_offset / element_size;
                        buffer["cells"] = serde_json::Value::Array(bytes.chunks(element_size as usize).enumerate().map(|(offset, bytes)| serde_json::json!({ "index": first_cell + offset as u64, "bytes_hex": encode_hex(bytes), "reflected": null })).collect());
                    }
                }
            }
        }
    }

    fn request_current_details(&mut self) {
        let Some(snapshot) = self.last_snapshot.as_ref() else { return; };
        // The bridge only delivers the *latest* request, so everything the
        // current selection needs goes into one request's `cpu_ranges`:
        //  1. the entity viewport (entity bits + values for the visible rows),
        //     sourced from the first column since entity bits are
        //     archetype-wide, and
        //  2. one row per component for the selected entity, so the
        //     components pane can preview every value and the details pane
        //     already has the one it needs.
        let mut cpu_ranges: Vec<serde_json::Value> = Vec::new();
        if !self.gpu_tab {
            if let Some(arch) = snapshot.get("archetypes").and_then(|v| v.as_array()).and_then(|a| a.get(self.selected_archetype)) {
                let aid = arch.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                let columns = arch.get("columns").and_then(|v| v.as_array());
                if let Some(first) = columns.and_then(|c| c.first()) {
                    let cid = first.get("component_id").and_then(|v| v.as_u64()).unwrap_or(0);
                    cpu_ranges.push(serde_json::json!({
                        "archetype_id": aid,
                        "component_id": cid,
                        "row_start": self.entity_visible_start,
                        "row_count": self.entity_visible_count.max(1),
                    }));
                }
                if let (Some(row), Some(columns)) = (self.selected_entity, columns) {
                    for col in columns {
                        let cid = col.get("component_id").and_then(|v| v.as_u64()).unwrap_or(0);
                        cpu_ranges.push(serde_json::json!({
                            "archetype_id": aid,
                            "component_id": cid,
                            "row_start": row,
                            "row_count": 1,
                        }));
                    }
                }
            }
        }
        let mut gpu = None;
        if self.gpu_tab { if let Some(name) = self.selected_gpu_buffer.as_deref() {
            let buffer = snapshot.get("gpu").and_then(|v| v.get("registry_buffers")).and_then(|v| v.as_array()).and_then(|a| a.iter().find(|b| b.get("name").and_then(|v| v.as_str()) == Some(name)));
            let kind = buffer.and_then(|b| b.get("kind")).and_then(|v| v.as_str()).unwrap_or("");
            let size = buffer.and_then(|b| b.get("element_size")).and_then(|v| v.as_u64()).unwrap_or(0);
            let capacity = buffer.and_then(|b| b.get("capacity_bytes")).and_then(|v| v.as_u64()).unwrap_or(0);
            // Textures and the synthetic demo entry have no SceneDB buffer
            // key to read. Real buffers use a viewport range until a cell is
            // explicitly selected in the central list.
            if kind != "texture" && !name.contains("[synthetic]") && size != 0 && capacity != 0 {
                let max_cell = capacity.checked_div(size).unwrap_or(0).saturating_sub(1);
                let cell = self.selected_gpu_cell.map(|v| v as u64).filter(|v| *v <= max_cell).map(|v| v as u32);
                let (byte_offset, byte_len) = if let Some(cell) = cell {
                    (cell as u64 * size, size.max(4))
                } else {
                    let start = (self.gpu_visible_start as u64).min(max_cell);
                    let count = (self.gpu_visible_count.max(1) as u64).min(max_cell.saturating_sub(start) + 1);
                    (start * size, count * size)
                };
                gpu = Some(serde_json::json!({ "buffer": name, "byte_offset": byte_offset, "byte_len": byte_len, "cell": cell }));
            }
        }}
        if cpu_ranges.is_empty() && gpu.is_none() { return; }
        let key = serde_json::json!({ "cpu_ranges": cpu_ranges, "gpu": gpu }).to_string();
        if self.last_request_key.as_deref() == Some(key.as_str()) { return; }
        let request = serde_json::json!({ "request_id": self.next_request_id, "cpu": null, "cpu_ranges": cpu_ranges, "gpu": gpu });
        let Ok(bytes) = serde_json::to_vec(&request) else { return; };
        if let Some(view) = self.session.as_ref().and_then(|s| s.view.as_ref()) { if view.publish_request(self.request_slot, &bytes) {
            self.request_slot = (self.request_slot + 1) % view.slot_count();
            self.next_request_id = self.next_request_id.wrapping_add(1);
            self.last_request_key = Some(key);
        }}
    }
}

/// Splice one CPU range response back into the matching archetype column of
/// `snapshot`. Entities, reflected values, and raw hex are *merged* into the
/// existing arrays (grown to `row_count` first) rather than replacing them,
/// so a one-row request for the selected entity does not erase the viewport
/// window another range request just fetched for the same column.
fn merge_cpu_response(snapshot: &mut serde_json::Value, cpu: &serde_json::Value) {
    let aid = cpu.get("archetype_id").and_then(|v| v.as_u64());
    let cid = cpu.get("component_id").and_then(|v| v.as_u64());
    let (Some(aid), Some(cid)) = (aid, cid) else { return; };
    let Some(arch) = snapshot
        .get_mut("archetypes")
        .and_then(|v| v.as_array_mut())
        .and_then(|a| a.iter_mut().find(|a| a.get("id").and_then(|v| v.as_u64()) == Some(aid)))
    else {
        return;
    };
    let Some(col) = arch
        .get_mut("columns")
        .and_then(|v| v.as_array_mut())
        .and_then(|c| c.iter_mut().find(|c| c.get("component_id").and_then(|v| v.as_u64()) == Some(cid)))
    else {
        return;
    };
    let start = cpu.get("row_start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let total = col.get("row_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let entities = cpu.get("entities").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    merge_range_into(&mut col["entities"], total, start, &entities, serde_json::Value::Null);
    if let Some(values) = cpu.get("rows_reflected").and_then(|v| v.as_array()).cloned() {
        merge_range_into(&mut col["rows_reflected"], total, start, &values, serde_json::Value::Null);
    } else if let Some(values) = cpu.get("rows_hex").and_then(|v| v.as_array()).cloned() {
        merge_range_into(
            &mut col["rows_hex"],
            total,
            start,
            &values,
            serde_json::Value::String(String::new()),
        );
    }
}

/// Write `values[i]` into `slot[start + i]`, growing `slot` to `total`
/// entries (padded with `pad`) if needed. Values outside `start..` are left
/// untouched.
fn merge_range_into(
    slot: &mut serde_json::Value,
    total: usize,
    start: usize,
    values: &[serde_json::Value],
    pad: serde_json::Value,
) {
    if !slot.is_array() {
        *slot = serde_json::Value::Array(Vec::new());
    }
    let Some(arr) = slot.as_array_mut() else { return; };
    if arr.len() < total {
        arr.resize(total, pad);
    }
    for (i, value) in values.iter().enumerate() {
        if let Some(slot) = arr.get_mut(start + i) {
            *slot = value.clone();
        }
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

/// Split a command-line string into arguments the way a shell would for the
/// common cases: whitespace separates arguments, and single or double quotes
/// group text (and are removed), including mid-token as in
/// `--project-path="C:\My Projects\Game"`. There is no shell here -- the
/// target is spawned directly -- so without this the quotes would reach the
/// target verbatim. Backslashes are literal, so Windows paths work unescaped.
fn split_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_token = false;
    let mut quote: Option<char> = None;
    for c in input.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => current.push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                in_token = true;
            }
            None if c.is_whitespace() => {
                if in_token {
                    args.push(std::mem::take(&mut current));
                    in_token = false;
                }
            }
            None => {
                current.push(c);
                in_token = true;
            }
        }
    }
    if in_token {
        args.push(current);
    }
    args
}

#[cfg(test)]
mod split_args_tests {
    use super::split_args;

    #[test]
    fn quotes_group_and_are_stripped() {
        assert_eq!(
            split_args(r#"--project-path="C:\Users\me\My Game" -v"#),
            vec![r"--project-path=C:\Users\me\My Game", "-v"]
        );
        assert_eq!(split_args("  a   'b c' "), vec!["a", "b c"]);
        assert_eq!(split_args(r#"x "" y"#), vec!["x", "", "y"]);
        assert!(split_args("   ").is_empty());
    }
}
