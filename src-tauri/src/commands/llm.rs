//! LLM service Tauri commands.
//!
//! These commands allow the frontend to query LLM service status and models
//! through the Python sidecar.

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::state::AppState;

/// LLM service status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMStatus {
    /// Whether LLM service is reachable.
    pub available: bool,
    /// LLM service API URL.
    pub url: String,
    /// Error message if unavailable.
    pub error: Option<String>,
}

/// Information about an LLM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMModel {
    /// Model name (e.g., "qwen3:14b").
    pub name: String,
    /// Model size on disk.
    pub size: String,
    /// Last modified timestamp.
    pub modified: String,
}

/// Response containing available LLM models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMModelsResponse {
    /// List of available models.
    pub models: Vec<LLMModel>,
    /// Configured default model.
    pub default_model: String,
}

/// Check LLM service status via sidecar.
///
/// Returns whether LLM service is available and reachable.
#[tauri::command]
pub async fn get_llm_status(state: State<'_, AppState>) -> Result<LLMStatus, String> {
    state
        .sidecar
        .get_async::<LLMStatus>("/llm/status")
        .await
        .map_err(|e| format!("Failed to get LLM status: {}", e))
}

/// List available LLM models via sidecar.
///
/// Returns the list of models available in the LLM service.
#[tauri::command]
pub async fn list_llm_models(state: State<'_, AppState>) -> Result<LLMModelsResponse, String> {
    state
        .sidecar
        .get_async::<LLMModelsResponse>("/llm/models")
        .await
        .map_err(|e| format!("Failed to list LLM models: {}", e))
}
