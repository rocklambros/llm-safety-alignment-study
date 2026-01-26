#!/bin/bash
set -e

echo "=============================================="
echo "LLM Safety Study - Model Import Automation"
echo "=============================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Source config if exists
if [ -f "$SCRIPT_DIR/config_output.sh" ]; then
    source "$SCRIPT_DIR/config_output.sh"
    echo "Loaded config: S3_BUCKET=$S3_BUCKET, AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID"
else
    echo "ERROR: config_output.sh not found. Run setup_all.sh first."
    exit 1
fi

# AWS profile
AWS_PROFILE="${AWS_PROFILE:-llm-safety}"
AWS_CMD="aws --profile $AWS_PROFILE"

echo "Using AWS profile: $AWS_PROFILE"
echo "Models directory: $MODELS_DIR"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# Model definitions: name|huggingface_id
MODELS=(
    "qwen3-8b-base|Qwen/Qwen3-8B-Base"
    "qwen3-8b-instruct|Qwen/Qwen3-8B"
    "mistral-7b-base|mistralai/Mistral-7B-v0.3"
    "mistral-7b-instruct|mistralai/Mistral-7B-Instruct-v0.3"
    "llama31-8b-base|meta-llama/Llama-3.1-8B"
    "llama31-8b-instruct|meta-llama/Llama-3.1-8B-Instruct"
)

# Checkpoint file
CHECKPOINT_FILE="$SCRIPT_DIR/.import_checkpoint"

# Functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step$" "$CHECKPOINT_FILE" 2>/dev/null
        return $?
    fi
    return 1
}

mark_completed() {
    local step="$1"
    echo "$step" >> "$CHECKPOINT_FILE"
}

download_model() {
    local name="$1"
    local hf_id="$2"
    local step="download_$name"

    if is_completed "$step"; then
        log "SKIP: $name already downloaded"
        return 0
    fi

    log "Downloading $name from $hf_id..."

    # Use hf download (newer command)
    if hf download "$hf_id" --local-dir "$MODELS_DIR/$name" 2>/dev/null; then
        mark_completed "$step"
        log "SUCCESS: Downloaded $name"
        return 0
    fi

    # Fallback to huggingface-cli
    if huggingface-cli download "$hf_id" --local-dir "$MODELS_DIR/$name" 2>/dev/null; then
        mark_completed "$step"
        log "SUCCESS: Downloaded $name"
        return 0
    fi

    log "ERROR: Failed to download $name"
    return 1
}

upload_to_s3() {
    local name="$1"
    local step="upload_$name"

    if is_completed "$step"; then
        log "SKIP: $name already uploaded to S3"
        return 0
    fi

    if [ ! -d "$MODELS_DIR/$name" ]; then
        log "ERROR: Model directory not found: $MODELS_DIR/$name"
        return 1
    fi

    log "Uploading $name to S3..."

    if $AWS_CMD s3 sync "$MODELS_DIR/$name" "s3://$S3_BUCKET/$name/" --quiet; then
        mark_completed "$step"
        log "SUCCESS: Uploaded $name to s3://$S3_BUCKET/$name/"
        return 0
    fi

    log "ERROR: Failed to upload $name"
    return 1
}

create_import_job() {
    local name="$1"
    local step="import_$name"

    if is_completed "$step"; then
        log "SKIP: Import job for $name already created"
        return 0
    fi

    log "Creating Bedrock import job for $name..."

    local role_arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockModelImportRole"
    local s3_uri="s3://$S3_BUCKET/$name/"

    if $AWS_CMD bedrock create-model-import-job \
        --job-name "${name}-import" \
        --imported-model-name "$name" \
        --role-arn "$role_arn" \
        --model-data-source "s3DataSource={s3Uri=$s3_uri}" 2>/dev/null; then
        mark_completed "$step"
        log "SUCCESS: Created import job for $name"
        return 0
    fi

    # Check if job already exists
    local status=$($AWS_CMD bedrock list-model-import-jobs \
        --query "modelImportJobSummaries[?jobName=='${name}-import'].status" \
        --output text 2>/dev/null)

    if [ -n "$status" ]; then
        log "Import job already exists with status: $status"
        mark_completed "$step"
        return 0
    fi

    log "ERROR: Failed to create import job for $name"
    return 1
}

check_import_status() {
    log "Checking import job status..."
    echo ""
    $AWS_CMD bedrock list-model-import-jobs \
        --query 'modelImportJobSummaries[*].[jobName,status,creationTime]' \
        --output table 2>/dev/null || echo "No import jobs found"
    echo ""
}

list_imported_models() {
    log "Listing imported models..."
    echo ""
    $AWS_CMD bedrock list-imported-models \
        --query 'modelSummaries[*].[modelName,modelArn]' \
        --output table 2>/dev/null || echo "No imported models found"
    echo ""
}

# Main execution
case "${1:-all}" in
    download)
        log "=== DOWNLOAD PHASE ==="
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            download_model "$name" "$hf_id" || true
        done
        ;;

    upload)
        log "=== UPLOAD PHASE ==="
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            upload_to_s3 "$name" || true
        done
        ;;

    import)
        log "=== IMPORT PHASE ==="
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            create_import_job "$name" || true
        done
        ;;

    status)
        check_import_status
        list_imported_models
        ;;

    all)
        log "=== FULL PIPELINE ==="
        echo ""
        echo "This will:"
        echo "  1. Download 6 models from HuggingFace (~90GB)"
        echo "  2. Upload them to S3"
        echo "  3. Create Bedrock import jobs"
        echo ""
        echo "Estimated time: 2-4 hours (depending on network speed)"
        echo "Note: Llama models require Meta approval on HuggingFace"
        echo ""
        read -p "Continue? (yes/no): " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            echo "Aborted."
            exit 0
        fi

        log "=== DOWNLOAD PHASE ==="
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            download_model "$name" "$hf_id" || log "WARNING: Skipping $name download"
        done

        log "=== UPLOAD PHASE ==="
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            if [ -d "$MODELS_DIR/$name" ]; then
                upload_to_s3 "$name" || log "WARNING: Skipping $name upload"
            fi
        done

        log "=== IMPORT PHASE ==="
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            # Check if uploaded
            if is_completed "upload_$name"; then
                create_import_job "$name" || log "WARNING: Skipping $name import"
            fi
        done

        echo ""
        log "=== PIPELINE COMPLETE ==="
        check_import_status
        ;;

    reset)
        log "Resetting checkpoint file..."
        rm -f "$CHECKPOINT_FILE"
        echo "Checkpoint cleared. Run again to restart from beginning."
        ;;

    single)
        if [ -z "$2" ]; then
            echo "Usage: $0 single <model-name>"
            echo "Available models: qwen3-8b-base, qwen3-8b-instruct, mistral-7b-base, mistral-7b-instruct, llama31-8b-base, llama31-8b-instruct"
            exit 1
        fi
        MODEL_NAME="$2"
        for model_def in "${MODELS[@]}"; do
            IFS='|' read -r name hf_id <<< "$model_def"
            if [ "$name" == "$MODEL_NAME" ]; then
                log "Processing single model: $name"
                download_model "$name" "$hf_id" && \
                upload_to_s3 "$name" && \
                create_import_job "$name"
                exit $?
            fi
        done
        echo "Model not found: $MODEL_NAME"
        exit 1
        ;;

    *)
        echo "Usage: $0 {all|download|upload|import|status|reset|single <model>}"
        echo ""
        echo "Commands:"
        echo "  all      - Run full pipeline (download, upload, import)"
        echo "  download - Download models from HuggingFace only"
        echo "  upload   - Upload downloaded models to S3 only"
        echo "  import   - Create Bedrock import jobs only"
        echo "  status   - Check import job and model status"
        echo "  reset    - Clear checkpoint file to restart"
        echo "  single   - Process a single model end-to-end"
        exit 1
        ;;
esac
