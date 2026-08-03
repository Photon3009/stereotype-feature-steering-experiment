#!/bin/zsh
# Batch aquin trace over selected probes; trace --check writes trace-check.json in cwd,
# so each run's artifact is moved to results/ under the probe id.
set -u
cd "$(dirname "$0")/.."
AQUIN=./venv/bin/aquin
IDS=(A01 A03 A05 A10 B01 B02 B05 B07)
while IFS= read -r line; do
  id=$(echo "$line" | ./venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['id'])")
  if [[ " ${IDS[@]} " == *" $id "* ]]; then
    prompt=$(echo "$line" | ./venv/bin/python -c "import json,sys; print('Continue this story in one sentence: ' + json.load(sys.stdin)['prompt'])")
    echo "=== $id: $prompt"
    $AQUIN trace --prompt "$prompt" --layer 8 --check 2>&1 | tail -4
    [[ -f trace-check.json ]] && mv trace-check.json "experiment/results/trace_${id}.json"
    [[ -f trace-check.png ]] && mv trace-check.png "experiment/results/trace_${id}.png"
  fi
done < experiment/probes/probes.jsonl
echo "BATCH DONE"
