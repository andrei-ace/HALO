#!/usr/bin/env bash
# Deploy the HALO cognitive service to Cloud Run.
#
# Usage:
#   PROJECT_ID=my-gcp-project REGION=us-central1 ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI authenticated with appropriate permissions
#   - Docker image pushed to gcr.io/${PROJECT_ID}/halo-cognitive:latest
#   - Secrets created:
#       gcloud secrets create google-api-key --data-file=-
#       gcloud secrets create halo-cloud-api-key --data-file=-

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your GCP project}"
: "${REGION:=us-central1}"

# Substitute PROJECT_ID in service.yaml and deploy
sed "s/\${PROJECT_ID}/${PROJECT_ID}/g" service.yaml | \
  gcloud run services replace /dev/stdin --region "${REGION}"

echo "Deployed to ${REGION}. Service URL:"
gcloud run services describe halo-cognitive --region "${REGION}" --format='value(status.url)'
