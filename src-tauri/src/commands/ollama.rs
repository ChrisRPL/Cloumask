//! Ollama-related Tauri commands.
//!
//! These commands allow the frontend to query Ollama LLM status and models
//! through the Python sidecar.

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::state::AppState;

/// Ollama service status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaStatus {
    /// Whether Ollama is reachable.
    pub available: bool,
    /// Ollama API URL.
    pub url: String,
    /// Error message if unavailable.
    pub error: Option<String>,
}

/// Information about an Ollama model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    /// Model name (e.g., "qwen3:14b").
    pub name: String,
    /// Model size on disk.
    pub size: String,
    /// Last modified timestamp.
    pub modified: String,
}

/// Response containing available Ollama models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelsResponse {
    /// List of available models.
    pub models: Vec<OllamaModel>,
    /// Configured default model.
    pub default_model: String,
}

/// Check Ollama service status via sidecar.
///
/// Returns whether Ollama is available and reachable.
#[tauri::command]
pub async fn get_ollama_status(state: State<'_, AppState>) -> Result<OllamaStatus, String> {
    state
        .sidecar
        .get_async::<OllamaStatus>("/ollama/status")
        .await
        .map_err(|e| format!("Failed to get Ollama status: {}", e))
}

/// List available Ollama models via sidecar.
///
/// Returns the list of models available in Ollama.
#[tauri::command]
pub async fn list_ollama_models(state: State<'_, AppState>) -> Result<OllamaModelsResponse, String> {
    state
        .sidecar
        .get_async::<OllamaModelsResponse>("/ollama/models")
        .await
        .map_err(|e| format!("Failed to list Ollama models: {}", e))
}
