# RUN_PIPELINE.md — 跑一遍 SFT → CPO → GRPO

从 `Qwen3-1.7B-Base` 训一个中英互译模型的完整流程。最终模型
`checkpoints/output_1.7b_grpo_full`,WMT23 zh→en COMET 0.8053(追平 1.8B 的
HY-MT1.5-1.8B 0.8052)、en→zh 0.8540。

> 详细设计与每次实验的 keep/discard 见 `program.md` / `results.tsv`;数据清单见
> `DATA_MANIFEST.md`;仓库结构见 `CLAUDE.md`。

## 全景图

```
输入模型                 输入数据                     脚本                     产出
──────────────────────────────────────────────────────────────────────────────────
models/Qwen3-1.7B-Base   data/alma_combined_          train/train.py       →  checkpoints/
                         sft_clean.jsonl (36.8K)      (SFT 全参)              output_1.7b_base_v2
    │
    ▼ (SFT 模型)
checkpoints/output_      data/cpo_v3_plus_7b.jsonl    train/train_cpo.py   →  checkpoints/
1.7b_base_v2             (44K 偏好对)                 (CPO, LoRA)            output_1.7b_cpo_v3_plus_7b
                                                                             (LoRA adapter)
    │                                                train/merge_lora_    →  ..._cpo_v3_plus_7b
    ▼ (base + adapter)                               qwen3.py               _merged
checkpoints/output_      data/grpo_data.jsonl         train/train_grpo.py  →  checkpoints/
1.7b_cpo_v3_plus_7b_     (WMT17-21 prompt 池)         (GRPO, 全参,           output_1.7b_grpo_full
merged                                               COMET reward)          ★最终模型
```

## 前置(三阶段共用)

- **venv**：`PY=/home/tfbao/new/HY-MT/.venv/bin/python`（含 torch/transformers/peft/trl/vllm/sacrebleu/unbabel-comet）
- **所有命令从 `Qwen/` 根目录运行**——脚本、默认值、下面的命令里的路径都相对它。
- **GPU**：单卡 RTX 4090 (24GB)。配置都按 batch_size 1–2 + 梯度累积 + gradient checkpointing；GRPO 期间 vLLM 以 0.2 显存占比 colocate。
- **COMET 模型**（GRPO 奖励 + 评测用）：本地
  `~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/.../checkpoints/model.ckpt`，
  在 `train/train_grpo.py`、`eval/eval_vllm.py`、`eval/eval_multi.py` 里写死，从不下载。

三份采用数据已在 `data/`，可直接跑四步；要从头重造见文末「数据来源链」。

## 四步命令

```bash
PY=/home/tfbao/new/HY-MT/.venv/bin/python
cd /home/tfbao/Shiyu/Interpreter/Qwen

# ① SFT：Qwen3-1.7B-Base 全参微调（ChatML，loss 只算 assistant 段，1 epoch）
$PY -u train/train.py --model_path ./models/Qwen3-1.7B-Base \
  --train_data ./data/alma_combined_sft_clean.jsonl \
  --output_dir ./checkpoints/output_1.7b_base_v2 \
  --num_epochs 1 --lr 2e-5 --lr_scheduler inverse_sqrt \
  --batch_size 2 --gradient_accumulation_steps 8 --gradient_checkpointing

# ② CPO：在 SFT 模型上做 LoRA 偏好训练（DPO loss + NLL；LoRA r=16, all-linear）
$PY -u train/train_cpo.py --model_path ./checkpoints/output_1.7b_base_v2 \
  --data_path ./data/cpo_v3_plus_7b.jsonl \
  --output_dir ./checkpoints/output_1.7b_cpo_v3_plus_7b \
  --lr 1e-4 --beta 0.05 --nll_weight 1.0 \
  --batch_size 1 --gradient_accumulation_steps 16 --max_length 512

# ③ 合并 LoRA（GRPO / vLLM 前必须；base 必须是 ② 用的同一个 SFT 模型）
$PY -u train/merge_lora_qwen3.py \
  --base_model_path ./checkpoints/output_1.7b_base_v2 \
  --adapter_model_path ./checkpoints/output_1.7b_cpo_v3_plus_7b \
  --output_path ./checkpoints/output_1.7b_cpo_v3_plus_7b_merged --save_dtype bf16

# ④ GRPO：全参 RL，奖励 = wmt22-comet-da COMET(1.0) + 4-gram 重复惩罚(0.3)
$PY -u train/train_grpo.py \
  --model_path ./checkpoints/output_1.7b_cpo_v3_plus_7b_merged \
  --data_path ./data/grpo_data.jsonl \
  --output_dir ./checkpoints/output_1.7b_grpo_full --full_finetune
```

## 评测（每阶段之后）

```bash
# WMT23 主指标（BLEU + COMET）
$PY -u eval/eval_vllm.py  --model_path ./checkpoints/<该阶段输出> --testset wmt23 --direction both
# 多测试集（WMT23/24 + Flores），确认涨分真实、非 COMET 刷分
$PY -u eval/eval_multi.py --model_path ./checkpoints/<该阶段输出>
```

**评测纪律**：WMT22 已污染（~17.5% 泄漏进 ALMA SFT），最终数字报 **WMT23**。WMT17–21 是
CPO/GRPO 的训练源，所以用 `eval_multi.py` 的 WMT23/24 + Flores 做独立交叉验证。

体验最终模型（交互式，中↔英自动判向）：

```bash
$PY -u eval/translate.py --model_path ./checkpoints/output_1.7b_grpo_full
```

## 关键约束（踩坑点）

- **顺序依赖**：CPO 的 `--model_path` 与合并的 `--base_model_path` 必须是**同一个 SFT 模型**；GRPO 吃的是**合并后**模型，不是 LoRA adapter。
- **CPO 只能用 LoRA**：`train_cpo.py --full_finetune` 存在，但全参 CPO 会让模型崩溃（results.tsv 多次验证），别用。**全参只用于 GRPO**。
- **GRPO 慢**：TRL GRPOTrainer + colocated vLLM，`num_generations=8`，`loss_type="dapo"`。单 4090 上 1500 步耗时长；最优其实在 **step 600** 附近（`grpo_full_s600`）。
- **提示词/eos**：所有脚本内联同一 prompt 模板；ChatML，`<|im_end|>` 作 eos（训练与生成 stop 一致）。改模板需重跑 SFT。

## 可选：跳过 CPO（SFT → GRPO）

results.tsv 有个发现：调参后的 **SFT→GRPO**（`output_1.7b_grpo_sft_tuned`，beta=0.02
lr=2e-6，6000 prompts）能打平 SFT→CPO→GRPO——**CPO 步可省**。验证方式：把 SFT 模型直接
喂 GRPO：

```bash
$PY -u train/train_grpo.py --model_path ./checkpoints/output_1.7b_base_v2 \
  --data_path ./data/grpo_data.jsonl \
  --output_dir ./checkpoints/output_1.7b_grpo_sft_tuned --full_finetune \
  --beta 0.02 --lr 2e-6
```

## 数据来源链（data/ 已是成品，从头重造才需要）

- **SFT** `alma_combined_sft_clean.jsonl` ← `datasets/ALMA-Human-Parallel` + `datasets/X-ALMA-Preference` 合并去 WMT22/23 泄漏。
- **CPO** `cpo_v3_plus_7b.jsonl`（链最长）：
  1. `data_build/generate_candidates.py` — 用 SFT 模型自生成 5 候选 → `data/cpo_candidates.jsonl`
  2. `data_build/build_cpo_data.py` — COMET 选 best/worst → `data/cpo_preference.jsonl`
  3. `data_build/hymt7b_translate_cpo_sources.py` — 用 `models/HY-MT1.5-7B` 翻译源句 → `data/hymt7b_cpov3_translations.json`
  4. `data_build/build_cpo_v3_plus_7b.py` — 25.7% chosen 换成 7B 译文 → `data/cpo_v3_plus_7b.jsonl`
- **GRPO** `grpo_data.jsonl` ← `data_build/build_grpo_data.py` 拉 WMT17–21 源句+参考译文（sacrebleu）。

## 涉及资产清单

| 类别 | 具体 |
|---|---|
| 模型 | `models/Qwen3-1.7B-Base`（SFT 起点）；`models/HY-MT1.5-7B`（仅重造 CPO 数据用）；产出 3 个 checkpoint |
| 数据 | `data/alma_combined_sft_clean.jsonl`、`data/cpo_v3_plus_7b.jsonl`、`data/grpo_data.jsonl`（+ 重造时的中间 jsonl） |
| 训练脚本 | `train/train.py`、`train/train_cpo.py`、`train/merge_lora_qwen3.py`、`train/train_grpo.py` |
| 评测脚本 | `eval/eval_vllm.py`、`eval/eval_multi.py`、`eval/translate.py` |
| 数据脚本 | `data_build/generate_candidates.py`、`build_cpo_data.py`、`hymt7b_translate_cpo_sources.py`、`build_cpo_v3_plus_7b.py`、`build_grpo_data.py` |
