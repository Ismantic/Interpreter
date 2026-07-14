#!/bin/bash
# CPU translation via llama.cpp.
#   ./translate_cpu.sh zh-en "人工智能正在改变世界。"
#   ./translate_cpu.sh en-zh "Artificial intelligence is changing the world."
# Override the model with:  MODEL=/path/to/x.gguf ./translate_cpu.sh ...
set -e
DIR=${1:?usage: translate_cpu.sh zh-en|en-zh "text"}
TEXT=${2:?usage: translate_cpu.sh zh-en|en-zh "text"}
BIN=/home/tfbao/new/llama.cpp/build/bin/llama-completion
MODEL=${MODEL:-/home/tfbao/Shiyu/Interpreter/Qwen/sherry_qat/gguf/grpo_full_q4km.gguf}
THREADS=${THREADS:-12}

if [ "$DIR" = "zh-en" ]; then
  INSTR="Translate the following text from Chinese to English.\nChinese: ${TEXT}\nEnglish:"
elif [ "$DIR" = "en-zh" ]; then
  INSTR="Translate the following text from English to Chinese.\nEnglish: ${TEXT}\nChinese:"
else
  echo "direction must be zh-en or en-zh" >&2; exit 1
fi

PROMPT="<|im_start|>user\n${INSTR}<|im_end|>\n<|im_start|>assistant\n"
"$BIN" -m "$MODEL" -n 256 --temp 0 -t "$THREADS" --no-display-prompt \
       -p "$(printf '%b' "$PROMPT")" 2>/dev/null \
  | sed 's/<|im_end|>.*//' | sed '/^> EOF/d' | sed '/^[[:space:]]*$/d'
