# DATA_MANIFEST.md — ReTok 实验结果 / 依赖的模型与数据

ReTok 是 `../Qwen/` 的 **A/B 对照**:同一条 SFT→CPO→GRPO 流水线、同一批数据、同一套超参
与奖励,**只换两样**——底座换成 PieceTokenizer 的 `phase2_ckpt_v18_tie`、tokenizer 换成
PieceTokenizer。核心问题:**换了 tokenizer 的底座,走完整流水线后能不能追上原生 HF tokenizer 的 Qwen?**

最优模型:`checkpoints/output_v18_tie_grpo_full`(WMT23 zh→en 18.43/0.7967,en→zh 31.79/0.8511)。

---

## 一、实验结果(WMT23,COMET = `Unbabel/wmt22-comet-da`)

| 阶段 | 模型 | zh→en BLEU/COMET | en→zh BLEU/COMET |
|---|---|---|---|
| base 5-shot | `models/phase2_ckpt_v18_tie` | 19.60 / 0.7834 | 40.99 / 0.8377 |
| **SFT** | `checkpoints/output_v18_tie_sft` | 19.34 / 0.7762 | 40.09 / 0.8392 |
| **CPO** | `..._cpo_v3_plus_7b_merged` | 18.11 / 0.7941 | 31.38 / 0.8480 |
| **GRPO** ★ | `..._grpo_full` | 18.43 / **0.7967** | 31.79 / **0.8511** |
| _(对照)_ Qwen SFT | `../Qwen/…/output_1.7b_base_v2` | 21.58 / 0.7924 | 39.82 / 0.8556 |
| _(对照)_ Qwen CPO | `../Qwen/…/cpo_v3_plus_7b_merged` | 19.16 / 0.8017 | 32.69 / 0.8507 |
| _(对照)_ Qwen GRPO | `../Qwen/…/grpo_sft_tuned` | 22.85 / 0.8003 | 41.97 / 0.8540 |

### 关键结论

- **piece tokenizer 的差距随流水线推进而收窄**(相对 Qwen 基线的 COMET 差,**两向都看**):

  | 阶段 | zh→en COMET Δ | en→zh COMET Δ |
  |---|---|---|
  | SFT  | −0.0162 | −0.0164 |
  | CPO  | −0.0076 | −0.0027 |
  | GRPO | **−0.0036** | **−0.0029** |

  两个方向一致:**换 tokenizer 在 SFT 阶段损失最大(约 −0.016),CPO/GRPO 把大部分差距补回,
  到 GRPO 时两向都收窄到 −0.003 ~ −0.004。**

- **BLEU 上差距不同步收窄**(相对 Qwen 基线的 BLEU 差):

  | 阶段 | zh→en BLEU Δ | en→zh BLEU Δ |
  |---|---|---|
  | SFT  | −2.24 | **+0.27** |
  | CPO  | −1.05 | −1.31 |
  | GRPO | −4.42 | −10.18 |

  - SFT 阶段 piece 的 en→zh BLEU 反而**略超** Qwen(+0.27)。
  - CPO 阶段是最干净的同路径对比(两边都 SFT→CPO),tokenizer 效应约 zh→en −1.05 / en→zh −1.31。
  - **GRPO 那行的大差距是路径差、非 tokenizer 差**:Qwen 基线取的是 **SFT→GRPO 的高 BLEU 变体
    (`grpo_sft_tuned` 22.85/41.97)**,而 ReTok grpo 走 SFT→CPO→GRPO(CPO 已把 en→zh BLEU 砸到 ~31)。
    要纯比 tokenizer,看 **CPO 行**;COMET 因为 CPO/GRPO 路径下都接近,故上表的收窄结论仍成立。
- **GRPO 相对自身 base 5-shot:两向 COMET 各 +0.013**(0.7834→0.7967、0.8377→0.8511)——
  流水线把 piece 底座显著抬到 few-shot 之上。
- **2nd-round GRPO(`_r2`)和双奖励 kiwi(`_grpo_kiwi`)= 噪声内**(≤+0.0006 COMET),
  已删;三者视为同一档,`grpo_full` 是存活代表。
- BLEU 的 en→zh 在 CPO 大跌(40→31),与 Qwen 同款 COMET/BLEU 背离,非 piece 独有。

完整逐行记录见 `results.tsv`。

---

## 二、依赖的模型

| 模型 | 位置 | 角色 | 来源 |
|---|---|---|---|
| **phase2_ckpt_v18_tie**(3.1G) | `models/` | **SFT 的起点**(PieceTokenizer 底座) | 由 `HYMT/` 那条 ReTok 换-tokenizer 线产出(Summer phase2 v18_tie),已复制进来自包含 |

- 运行时还依赖 venv 里的 C++ `piece_tokenizer` 扩展(非模型文件)。
- ReTok **不直接依赖 HY-MT-7B**:CPO 数据里的 7B 增强是 Qwen 侧上游做好的,ReTok 只复用成品数据。

---

## 三、依赖的数据(`data/`,均 git-ignored)

三份都是**从 `../Qwen/` 复制来的同一批数据**(A/B 契约:数据必须一致,只变 tokenizer/底座)。
构建来源链见 **`../Qwen/DATA_MANIFEST.md`**,这里不重复。

| 阶段 | 文件 | 规模 | 说明 |
|---|---|---|---|
| SFT | `data/alma_combined_sft_clean.jsonl` | 16M | ALMA+X-ALMA,去 WMT22/23 泄漏 |
| CPO | `data/cpo_v3_plus_7b.jsonl` | 29M | 自生成偏好对,25.7% chosen 换 HY-MT-7B 译文 |
| GRPO | `data/grpo_data.jsonl` | 13M | WMT17-21 源句+参考译文(prompt 池) |

> 注意:数据虽同,但**训练时用 PieceTokenizer 重新编码**(`<bos><user>…<assistant>…<eos>`),
> 与 Qwen 的 ChatML 不同——这正是 A/B 的变量所在。

---

## 四、产出的 checkpoint(`checkpoints/`,均 git-ignored)

| 目录 | 阶段 | 大小 |
|---|---|---|
| `output_v18_tie_sft` | SFT | 3.0G |
| `output_v18_tie_cpo_v3_plus_7b` | CPO LoRA adapter | 74M |
| `output_v18_tie_cpo_v3_plus_7b_merged` | CPO 合并(GRPO 起点) | 3.0G |
| `output_v18_tie_grpo_full` ★ | GRPO(最优,部署用) | 3.0G |

每个 checkpoint 目录**自包含**:`model.safetensors` + 5 个 piece 文件
(`piece.model`、`dict.txt`、`token_mapping.json`、`special_tokens_map.json`、
`tokenizer_config.json`)。部署/推理见 `RUN_BEST_MODEL.md`。
