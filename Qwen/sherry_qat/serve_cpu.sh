#!/bin/bash
# Start the llama.cpp CPU translation server (OpenAI-compatible API on :8080).
# Then open translate.html in a browser.
#   ./serve_cpu.sh                       # serves the default Q4_K_M model
#   MODEL=/path/to/x.gguf ./serve_cpu.sh # serve a different GGUF
MODEL=${MODEL:-/home/tfbao/Shiyu/Interpreter/Qwen/sherry_qat/gguf/grpo_full_q4km.gguf}
WEBUI="$(cd "$(dirname "$0")" && pwd)/webui"
HOST=${HOST:-0.0.0.0}   # 0.0.0.0 = LAN-accessible; set HOST=127.0.0.1 for local-only
PORT=${PORT:-8080}
echo "serving $MODEL"
echo "local:  http://127.0.0.1:$PORT/   LAN:  http://<this-machine-ip>:$PORT/"
exec /home/tfbao/new/llama.cpp/build/bin/llama-server \
  -m "$MODEL" --host "$HOST" --port "$PORT" -t "${THREADS:-12}" -c 2048 \
  --path "$WEBUI"
