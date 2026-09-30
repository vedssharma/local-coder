#!/bin/bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/app/models/model.gguf}"
if [ "$#" -eq 0 ]; then
    set -- chat
fi
# Configure through the supported API; never rewrite application source.
python - "$APP_DIR" "$MODEL_PATH" "$1" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import config
profile = config.get_model_config()
if profile['backend'] == 'embedded':
    path = Path(sys.argv[2])
    if path.is_file():
        if not config.set_model_path(str(path.resolve())):
            raise SystemExit('MODEL_PATH must point to a GGUF model')
    elif sys.argv[3] in ('ask', 'chat', 'edit') and not Path(profile['model_path']).is_file():
        raise SystemExit('Model missing. Mount a GGUF model and set MODEL_PATH, or configure a local OpenAI-compatible backend.')
PY
exec python "$APP_DIR/main.py" "$@"
