#!/usr/bin/env bash
# Pre-download benchmark datasets to HuggingFace cache
set -euo pipefail

python - <<'EOF'
from datasets import load_dataset

datasets_to_cache = [
    ("trivia_qa", "rc"),
    ("allenai/openbookqa", "main"),
    ("allenai/qasper", None),
]

for name, config in datasets_to_cache:
    try:
        print(f"Caching {name} ({config})...")
        ds = load_dataset(name, config, split="validation[:100]", trust_remote_code=True)
        print(f"  -> {len(ds)} examples cached")
    except Exception as e:
        print(f"  -> Failed: {e}")
EOF
echo "Dataset caching complete."
