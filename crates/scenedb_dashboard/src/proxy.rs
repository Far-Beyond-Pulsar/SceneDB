use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use crate::AppState;

pub async fn proxy_handler(
    State(state): State<AppState>,
    path: Option<Path<String>>,
    query: Option<Query<HashMap<String, String>>>,
) -> Response {
    let path = path.map(|p| p.0).unwrap_or_default();
    let path = path.trim_start_matches('/');

    let target_url = if !query.is_some() || query.as_ref().map(|q| q.is_empty()).unwrap_or(true) {
        format!("{}/{}", state.telemetry_base, path)
    } else {
        let qs = query.unwrap();
        let params: Vec<String> = qs.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        format!("{}/{}?{}", state.telemetry_base, path, params.join("&"))
    };

    match state.proxy_client.get(&target_url).send().await {
        Ok(resp) => {
            let status = resp.status();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            let body = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return (StatusCode::BAD_GATEWAY, format!("proxy read error: {}", e)).into_response();
                }
            };
            Response::builder()
                .status(status)
                .header("content-type", content_type)
                .header("access-control-allow-origin", "*")
                .body(Body::from(body.to_vec()))
                .unwrap()
        }
        Err(e) => {
            if e.is_connect() {
                (
                    StatusCode::BAD_GATEWAY,
                    format!(
                        "{{\"error\":\"cannot connect to SceneDB telemetry at {}\"}}",
                        state.telemetry_base
                    ),
                )
                    .into_response()
            } else if e.is_timeout() {
                (
                    StatusCode::GATEWAY_TIMEOUT,
                    "{\"error\":\"telemetry server timed out\"}".to_string(),
                )
                    .into_response()
            } else {
                (
                    StatusCode::BAD_GATEWAY,
                    format!("{{\"error\":\"proxy error: {}\"}}", e),
                )
                    .into_response()
            }
        }
    }
}
