# 运行最优 ReTok 翻译模型

记录 ReTok（PieceTokenizer + Qwen3-1.7B）训练流水线的最优 checkpoint 在新机器上的启动方式。

## 模型

- **路径**：`Translator_piece/output_v18_tie_grpo_full/`（约 3.0 GB）
- **架构**：Qwen3ForCausalLM（1.7B，hidden=2048，bf16），自定义 PieceTokenizer（vocab=81903）
- **训练流水线**：SFT → CPO (LoRA, merged) → GRPO (full-param)，base 来自 `/home/tfbao/Shiyu/Summer/output/phase2_ckpt_v18`
- **WMT23 成绩**：zh→en 18.43 / COMET 0.7967；en→zh 31.79 / COMET 0.8511

模型目录是**自包含**的，里面就有 `piece.model`、`dict.txt`、`token_mapping.json`、`special_tokens_map.json`、`tokenizer_config.json`、`model.safetensors`、`generation_config.json`。

### 关键 token ID

| Token   | ID    |
|---------|-------|
| `<unk>` | 0     |
| `<s>`   | 1     |
| `</s>`  | 2     |
| `<pad>` | 81899 |
| `<user>` | 81900 |
| `<assistant>` | 81901 |
| `<system>` | 81902 |

### 对话格式（不是 ChatML）

```
<s> <user> {prompt-pieces} <assistant> {response-pieces} </s>
```

生成时输入 `<s> <user> {prompt} <assistant>`，模型续写直到 `</s>`。

### Prompt 模板（必须和训练时一致）

```python
PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"
```

## 依赖

| 依赖 | 用途 |
|------|------|
| `/home/tfbao/new/HY-MT/.venv/` | venv（torch / transformers / vllm / sacrebleu / unbabel-comet） |
| `piece_tokenizer`（C++ 扩展，装在 venv 里） | PieceTokenizer 编解码；源码在 `/home/tfbao/Shiyu/Tokenizer/` |
| `Interpreter/tokenizer_wrapper.py` | `PieceTokenizerWrapper` —— HF 风格的薄封装 |
| `Translator_piece/eval_vllm_piece.py` | WMT 评测脚本（vLLM） |

新机器路径若不是 `/home/tfbao/...`，需要修改：
- 各 `run_*.sh` 顶部的 `PY=`
- `eval_vllm_piece.py` 里的 `COMET_CKPT` 路径（默认 `~/.cache/comet/...`，相对 home 一般不用动）

## 三级启动验证

### Level 1 — venv + piece_tokenizer 加载

```bash
/home/tfbao/new/HY-MT/.venv/bin/python -c "import torch, transformers, vllm, piece_tokenizer; print('ok')"
```

若 `piece_tokenizer` 报错，进 `/home/tfbao/Shiyu/Tokenizer/` 按那边 README 重编译并 `pip install -e .`。

### Level 2 — Tokenizer 编解码

```bash
cd /home/tfbao/Shiyu/Interpreter
/home/tfbao/new/HY-MT/.venv/bin/python -c "
from tokenizer_wrapper import PieceTokenizerWrapper
t = PieceTokenizerWrapper('Translator_piece/output_v18_tie_grpo_full')
print('vocab:', t.vocab_size, 'bos/eos/pad:', t.bos_token_id, t.eos_token_id, t.pad_token_id)
ids = t.apply_chat_template([{'role':'user','content':'Hello world'}], add_generation_prompt=True)
print('chat ids:', ids[:20])
print('decode:', t.decode(ids))
"
```

期望输出：`vocab: 81903 bos/eos/pad: 1 2 81899`，chat ids 第一个是 1（bos），后面有 81900（user）和 81901（assistant）。

### Level 3 — 最小端到端推理（不依赖 vLLM）

下面是一段独立 Python，把它存成 `Translator_piece/quick_infer.py` 直接跑：

```python
"""Minimal greedy inference for the ReTok GRPO model — no vLLM, just transformers."""
import sys, os
sys.path.insert(0, "/home/tfbao/Shiyu/Interpreter")
import torch
from transformers import AutoModelForCausalLM
from tokenizer_wrapper import PieceTokenizerWrapper

MODEL_DIR = "/home/tfbao/Shiyu/Interpreter/Translator_piece/output_v18_tie_grpo_full"

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def translate(text, direction="zh-en", max_new_tokens=256):
    tpl = PROMPT_ZH2EN if direction == "zh-en" else PROMPT_EN2ZH
    prompt = tpl.format(src=text)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    input_ids = torch.tensor([ids], device=model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0, input_ids.shape[1]:].tolist()
    if gen and gen[-1] == tok.eos_token_id:
        gen = gen[:-1]
    return tok.decode(gen, skip_special_tokens=True).strip()


if __name__ == "__main__":
    tok = PieceTokenizerWrapper(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print(translate("今天天气真好。", "zh-en"))
    print(translate("The quick brown fox jumps over the lazy dog.", "en-zh"))
```

运行：

```bash
cd /home/tfbao/Shiyu/Interpreter/Translator_piece
/home/tfbao/new/HY-MT/.venv/bin/python quick_infer.py
```

## 完整 WMT 评测（vLLM）

仓库自带的 `eval_vllm_piece.py` 已经对接好 sacrebleu BLEU + Unbabel COMET。

```bash
cd /home/tfbao/Shiyu/Interpreter/Translator_piece
PY=/home/tfbao/new/HY-MT/.venv/bin/python

# 完整 WMT23 双向 + BLEU + COMET
$PY -u eval_vllm_piece.py \
    --model_path ./output_v18_tie_grpo_full \
    --testset wmt23 \
    --direction both

# 跳过 COMET（COMET ckpt 未下载时）
$PY -u eval_vllm_piece.py \
    --model_path ./output_v18_tie_grpo_full \
    --testset wmt23 \
    --direction both \
    --no_comet

# 单向
$PY -u eval_vllm_piece.py \
    --model_path ./output_v18_tie_grpo_full \
    --testset wmt24 \
    --direction zh-en
```

### COMET checkpoint

`eval_vllm_piece.py` 写死从本地缓存读：
`~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/.../checkpoints/model.ckpt`

若新机器没有，跑一次 `from comet import download_model; download_model("Unbabel/wmt22-comet-da")`，或加 `--no_comet` 跳过。

## 排错

| 现象 | 原因 / 处理 |
|------|------------|
| `ImportError: piece_tokenizer` | venv 没装好。`cd /home/tfbao/Shiyu/Tokenizer && pip install -e .` |
| `No piece model found in ...` | model_path 写错了，或者 `piece.model` 没拷过来 |
| vLLM 报 tokenizer 相关错误 | 确认 `LLM(..., skip_tokenizer_init=True)`（`eval_vllm_piece.py` 已正确设置） |
| 输出乱码 / 一直停不下来 | `eos_token_id=2` 没传；prompt 模板和训练不一致；或 `apply_chat_template(..., add_generation_prompt=True)` 漏了 |
| COMET ckpt 找不到 | 用 `--no_comet`，或下载到 `~/.cache/comet/...` |

## 顺手记录：其它 checkpoint

`output_v18_tie_grpo_full` 是流水线终点。中间产物也都在 `Translator_piece/`：

- `output_v18_tie_sft/` —— Phase 1 SFT，3.0 GB
- `output_v18_tie_cpo_v3_plus_7b/` —— Phase 2 CPO 的 LoRA adapter，74 MB
- `output_v18_tie_cpo_v3_plus_7b_merged/` —— LoRA 合并版，3.0 GB（GRPO 的起点）

如果只想跑最强模型，**只需要保留** `output_v18_tie_grpo_full/`，其它都可以删。
