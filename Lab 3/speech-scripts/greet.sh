#!/usr/bin/env bash
# Greets the user by name using Piper neural TTS.
# Usage: ./greet.sh            (uses the default name)
#        ./greet.sh "Someone"  (greets someone else)

set -euo pipefail
VOICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/voices"
NAME="${1:-Jacey}"

python3 -m piper \
  --model en_US-lessac-medium \
  --data-dir "$VOICES_DIR" \
  --output-raw \
  -- "Hi ${NAME}! Welcome back. It's nice to hear from you." \
  | aplay -q -r 22050 -f S16_LE -t raw -
