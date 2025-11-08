#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="gazefollow/auto_phrase_grounding"
VAL_REF="training_datasets/20251007_165818_sgl_conversation_data_20251007_171523/val.json"
PYTHON="/galitylab/students/alonmardi/llava/bin/python"

$PYTHON gazefollow/auto_phrase_grounding/combine_conversations.py "$ROOT_DIR"
$PYTHON gazefollow/auto_phrase_grounding/split_combined_conversations.py --reference-val "$VAL_REF"
