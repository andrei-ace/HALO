variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "planner_model" {
  description = "Gemini model for the planner"
  type        = string
  default     = "gemini-3.1-flash-lite-preview"
}

variable "vlm_model" {
  description = "Gemini model for VLM scene analysis"
  type        = string
  default     = "gemini-3.1-flash-lite-preview"
}

variable "live_agent_model" {
  description = "Gemini model for the Live Agent (native audio)"
  type        = string
  default     = "gemini-2.5-flash-native-audio-preview-12-2025"
}

variable "invoker_impersonators" {
  description = "IAM members allowed to impersonate the invoker SA (e.g. [\"user:alice@example.com\"])"
  type        = list(string)
  default     = []
}

variable "deploy_service" {
  description = "Create the Cloud Run service. Set to false for initial bootstrap (registry + secrets only)."
  type        = bool
  default     = true
}
