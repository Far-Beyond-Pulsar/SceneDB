mod proxy;

use axum::body::Body;
use axum::http::{HeaderValue, Method, Uri};
use axum::response::Response;
use axum::routing::{any, get};
use axum::Router;
use clap::Parser;
use rust_embed::RustEmbed;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::EnvFilter;

#[derive(RustEmbed)]
#[folder = "dashboard/out/"]
struct DashboardAssets;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "3456")]
    port: u16,
    #[arg(long, default_value = "8081")]
    telemetry_port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    telemetry_host: String,
}

#[derive(Clone)]
struct AppState {
    telemetry_base: String,
    proxy_client: reqwest::Client,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let args = Args::parse();
    let telemetry_base = format!("http://{}:{}", args.telemetry_host, args.telemetry_port);

    tracing::info!(
        "SceneDB Dashboard starting — http://127.0.0.1:{}  (proxying telemetry from {})",
        args.port,
        telemetry_base,
    );

    let state = AppState {
        telemetry_base,
        proxy_client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("reqwest client"),
    };

    let cors = CorsLayer::new()
        .allow_origin(HeaderValue::from_static("http://localhost:3456"))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/*path", any(proxy::proxy_handler))
        .route("/api", any(proxy::proxy_handler))
        .route("/", get(static_handler))
        .route("/*path", get(static_handler))
        .layer(cors)
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind failed");

    axum::serve(listener, app)
        .await
        .expect("server error");
}

async fn static_handler(
    uri: Uri,
) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.is_empty() || path.starts_with("api/") {
        return serve_exact("index.html");
    }
    // Try exact, then .html, then /index.html
    if let Some(resp) = try_serve(path) {
        return resp;
    }
    if let Some(resp) = try_serve(&format!("{}.html", path)) {
        return resp;
    }
    if let Some(resp) = try_serve(&format!("{}/index.html", path)) {
        return resp;
    }
    // SPA fallback
    serve_exact("index.html")
}

fn try_serve(path: &str) -> Option<Response> {
    let content = DashboardAssets::get(path)?;
    Some(serve(path, content))
}

fn serve(path: &str, content: rust_embed::EmbeddedFile) -> Response {
    Response::builder()
        .header("content-type", mime_for(path))
        .header("cache-control", "no-cache, no-store, must-revalidate")
        .body(Body::from(content.data.to_vec()))
        .unwrap()
}

fn serve_exact(path: &str) -> Response {
    let content = DashboardAssets::get(path).unwrap();
    Response::builder()
        .header("content-type", mime_for(path))
        .header("cache-control", "no-cache, no-store, must-revalidate")
        .body(Body::from(content.data.to_vec()))
        .unwrap()
}

fn mime_for(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript"
    } else if path.ends_with(".json") {
        "application/json"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".ico") {
        "image/x-icon"
    } else if path.ends_with(".woff2") {
        "font/woff2"
    } else if path.ends_with(".woff") {
        "font/woff"
    } else if path.ends_with(".ttf") {
        "font/ttf"
    } else {
        "application/octet-stream"
    }
}
