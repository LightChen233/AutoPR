#!/bin/bash

set -e

DATA_PREP_DIR="data_preparation"
EVAL_DATA_DIR="eval/data"

mkdir -p "$EVAL_DATA_DIR"

unzip -q "$DATA_PREP_DIR/Fine_grained_evaluation.zip" -d "$DATA_PREP_DIR"


unzip -q "$DATA_PREP_DIR/twitter_figure.zip" -d "$DATA_PREP_DIR"


unzip -q "$DATA_PREP_DIR/xhs_figure.zip" -d "$DATA_PREP_DIR"


python3 "$DATA_PREP_DIR/download_pdfs.py"


find "$DATA_PREP_DIR" -maxdepth 1 -type f ! -name "*.zip" -exec mv {} "$EVAL_DATA_DIR" \;

echo "Script execution completed!"