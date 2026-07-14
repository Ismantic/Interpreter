# REPRODUCE.md — 在另一台机器上复现 Qwen 翻译项目

代码在 git 里（`git clone` 即得）；模型、数据集、checkpoint 都是 **git-ignored、只在磁盘上**
（`models/ ~19G`、`checkpoints/ ~14G`、`datasets/ ~97M`、`data/ ~131M`）。所以复现的核心
是：**装环境 + 把外部资产弄齐**。两种路线——A 复制现成产物（快），B 从零重训（真复现）。

## 0. 硬件 / 系统前提

- GPU：单卡 **24GB（RTX 4090 级别）**。所有配置按 bs 1–2 + 梯度累积 + gradient checkpointing。
- Linux + CUDA；Python 3.10。

## 1. 拉代码

```bash
git clone git@github.com:Ismantic/Interpreter.git
cd Interpreter/Qwen          # 本项目在 Qwen/ 子目录；所有命令都从这里跑
```

## 2. 建环境

```bash
python3.10 -m venv .venv && source .venv/bin/activate
# torch / vllm 先按你的 CUDA 版本单独装，再装其余
pip install torch==2.11.0            # 或按 pytorch.org 选 CUDA 轮子
pip install vllm==0.20.2
pip install -r requirements.txt
```

> 脚本里用 `PY=<python>` 调用，默认写的是原机器的
> `/home/tfbao/new/HY-MT/.venv/bin/python`。**换机器后把 `PY` 指到你自己的 venv 即可**
> （命令行里覆盖，或改 `run_experiments.sh` / `eval/eval_cpo_v3_plus_7b.sh` / `RUN_PIPELINE.md` 顶部）。

## 3. COMET 奖励/评测模型（必需）

GRPO 奖励和所有评测都用 `Unbabel/wmt22-comet-da`，脚本里**写死了本地路径、不联网下载**：

```bash
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
```

下载后确认路径与脚本一致：
`~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/<hash>/checkpoints/model.ckpt`。
若 `<hash>` 不同，改这三处的常量：`train/train_grpo.py`、`eval/eval_vllm.py`、`eval/eval_multi.py`。

## 4. 机器相关路径（只在换了仓库绝对路径时才需要改）

- **核心 train/eval/data_build 脚本全用相对路径**（`./models ./data ./checkpoints`），只要
  `cd Qwen/` 再跑就行，无需改动。
- 只有 **sherry_qat 量化子实验**和两个 helper 脚本写了绝对路径
  `/home/tfbao/Shiyu/Interpreter/Qwen`（`sherry_qat/*.py` 的 `T=`、`run_experiments.sh` /
  `eval/eval_cpo_v3_plus_7b.sh` 的 `cd`）。跑主线不受影响；要跑 sherry_qat 就把这些改成你的路径。

---

## 路线 A：复制现成产物（最快，直接跑/评测已训好的模型）

在原机器上 rsync 需要的目录过来（放回 `Qwen/` 下同名位置）：

```bash
# 最小：只想跑最终模型（自包含,含 tokenizer）
rsync -av <原机>:.../Qwen/checkpoints/output_1.7b_grpo_full ./checkpoints/

# 完整：能评测 + 重训任意阶段（约 33G）
rsync -av <原机>:.../Qwen/{models,datasets,data,checkpoints} ./
```

跑起来：

```bash
PY=./.venv/bin/python
# 交互体验
$PY -u eval/translate.py --model_path ./checkpoints/output_1.7b_grpo_full
# 复现 WMT23 数字（应得 zh→en 0.8053 / en→zh 0.8540）
$PY -u eval/eval_vllm.py --model_path ./checkpoints/output_1.7b_grpo_full --testset wmt23 --direction both
```

## 路线 B：从零重训（真复现）

### B1. 下载底座模型 → `models/`

```bash
huggingface-cli download Qwen/Qwen3-1.7B-Base --local-dir ./models/Qwen3-1.7B-Base
# 可选（只为 CPO 的 7B 增强,贡献 ~+0.004 COMET；不要可跳过）
huggingface-cli download tencent/HY-MT1.5-7B  --local-dir ./models/HY-MT1.5-7B
```

### B2. 下载原始语料 → `datasets/`

```bash
huggingface-cli download haoranxu/ALMA-Human-Parallel --repo-type dataset --local-dir ./datasets/ALMA-Human-Parallel
huggingface-cli download haoranxu/X-ALMA-Preference   --repo-type dataset --local-dir ./datasets/X-ALMA-Preference
# flores 仅 eval_multi.py 用（可选）；WMT 测试集由 sacrebleu 自动下载,无需手动
```

### B3. 重建训练数据 → `data/`（详见 `DATA_MANIFEST.md` 的构建链）

```bash
PY=./.venv/bin/python
# SFT 数据：由 ALMA+X-ALMA 合并去污染（原始合并脚本见 program.md；产物 data/alma_combined_sft_clean.jsonl）
# CPO 数据链：
$PY -u data_build/generate_candidates.py --model_path ./checkpoints/output_1.7b_base_v2 --output ./data/cpo_candidates.jsonl  # 需先有 SFT 模型
$PY -u data_build/build_cpo_data.py --input ./data/cpo_candidates.jsonl --output ./data/cpo_preference.jsonl
$PY -u data_build/hymt7b_translate_cpo_sources.py     # 用 models/HY-MT1.5-7B
$PY -u data_build/build_cpo_v3_plus_7b.py             # → data/cpo_v3_plus_7b.jsonl
# GRPO 数据（WMT17-21，sacrebleu 自动拉）：
$PY -u data_build/build_grpo_data.py                  # → data/grpo_data.jsonl
```

### B4. 跑三阶段流程

见 **`RUN_PIPELINE.md`** 的四步命令（SFT → CPO → merge → GRPO）+ 每步评测。
期望终点：`checkpoints/output_1.7b_grpo_full`，WMT23 zh→en COMET 0.8053 / en→zh 0.8540。

> 注意 CPO 数据依赖 SFT 模型（B3 的候选生成用 SFT 模型自生成），所以顺序是
> **先 SFT → 再造 CPO 数据 → CPO → GRPO**，不是先把数据全造好。

---

## 已知坑

- **无联网时 COMET 会失败**：必须先完成第 3 步。
- **COMET snapshot hash**：不同下载可能 hash 不同,对不上就改那 3 个脚本里的常量。
- **绝对路径**：换仓库位置后 sherry_qat 的 `T=` 和 run 脚本的 `cd` 要同步改（见第 4 节）。
- **版本敏感**：vllm/trl/transformers 跨大版本 API 常变,尽量按 `requirements.txt` 的 pin。
