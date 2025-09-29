#!/bin/bash

# --- Configuration ---
# Set the core paths and parameters for the generation run.

# Directory containing the project subfolders with PDFs.
INPUT_DIR="papers"
# Directory where the final posts will be saved.
OUTPUT_DIR="output"
# Directory to cache reusable assets like extracted figures.
CACHE_DIR="pragent_cache"

# API and Model Settings
TEXT_API_KEY="YOUR_API_KEY" # Replace with your key, or leave empty to use .env
TEXT_API_BASE="https://api.deepseek.com/v1" # Example: DeepSeek API
TEXT_MODEL="deepseek-chat"
VISION_MODEL="deepseek-vl-chat"

# Maximum number of concurrent projects to process.
CONCURRENCY=5

# --- Advanced Modes ---
# To run a baseline or ablation study, uncomment one of the following lines.
# For details, see the paper's experiments section.

# Run a baseline generation process instead of the full pipeline.
# Options: "original", "fewshot", "with_figure"
# BASELINE_MODE="fewshot"

# Run an ablation study by disabling a specific module.
# Options: "no_logical_draft", "no_visual_analysis", "no_platform_adaptation", etc.
# ABLATION_MODE="no_platform_adaptation"

# --- Execution ---
# This section builds and runs the command. Do not modify.
echo "--- Starting PRAgent Generation ---"
CMD_ARGS=(
    "--input-dir" "$INPUT_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--text-api-base" "$TEXT_API_BASE"
    "--text-model" "$TEXT_MODEL"
    "--vision-model" "$VISION_MODEL"
    "--concurrency" "$CONCURRENCY"
)

# Add optional arguments if they are set
[ -n "$TEXT_API_KEY" ] && CMD_ARGS+=("--text-api-key" "$TEXT_API_KEY")
[ -n "$CACHE_DIR" ] && CMD_ARGS+=("--cache-dir" "$CACHE_DIR")
[ -n "$BASELINE_MODE" ] && CMD_ARGS+=("--baseline-mode" "$BASELINE_MODE")
[ -n "$ABLATION_MODE" ] && CMD_ARGS+=("--ablation" "$ABLATION_MODE")

# Run the Python script
python run.py "${CMD_ARGS[@]}"

echo "--- Generation Complete ---"