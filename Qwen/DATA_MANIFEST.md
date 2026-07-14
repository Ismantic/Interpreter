# 数据清单 — Qwen3-1.7B 中英翻译模型项目（完整记录）

目录：`/home/tfbao/Shiyu/Interpreter/Qwen/`
最佳模型：`checkpoints/output_1.7b_grpo_full`（SFT→CPO→GRPO，WMT23 0.8054/0.8542）

> 目录重排后：下表中所有 `*.jsonl`/`*.json` 现位于 **`data/`**，checkpoint 位于
> **`checkpoints/`**,数据集(ALMA/X-ALMA/flores/Qwen3-*)位于 **`datasets/`**。
> 大量早期/弃用数据与 checkpoint 已按 results.tsv 判定清理,仅保留采用/基底/参考项。

---

## 一、最佳模型实际用到的数据

| 阶段 | 文件 | 规模 | 说明 |
|------|------|------|------|
| SFT | `alma_combined_sft_clean.jsonl` | 36800 | ALMA+X-ALMA 平行语料，去WMT22/23泄漏 |
| CPO | `cpo_v3_plus_7b.jsonl` | 44195 | 自生成偏好对，25.7%chosen换成HY-MT 7B更优译文 |
| GRPO | `grpo_data.jsonl`（前3000） | 22120 | WMT17-21源句+参考译文，COMET reward打分 |

---

## 二、SFT 阶段全部数据

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `alma_combined_sft.jsonl` | 44624 | ALMA+X-ALMA 合并，未去污染 | 中间 |
| `alma_combined_sft_clean.jsonl` | 36800 | 去WMT22/23泄漏 | **采用** |
| `alma_sources.jsonl` | 44624 | 上述数据的源句提取 | 工具 |
| `sft_balanced.jsonl` | 36800 | 翻译方向50:50平衡 | 弃用（zh-en掉） |
| `sft_enhanced.jsonl` | 111800 | ALMA + 蒸馏数据112K | 弃用（zh-en掉） |
| `sft_with_general.jsonl` | 52571 | 70%翻译+30%通用 | 弃用 |
| `sft_with_cqia.jsonl` | 46000 | 80%翻译+20%COIG-CQIA（按样本） | 弃用 |
| `sft_with_cqia_v2.jsonl` | 40012 | 80/20按token平衡，CQIA<256 | 弃用 |
| `sft_with_oasst.jsonl` | 46000 | 80%翻译+20%OpenAssistant | 弃用 |
| `sft_with_oasst_v2.jsonl` | 41472 | 80/20按token平衡 | 弃用 |
| `sft_with_smoltalk.jsonl` | 46000 | 80%翻译+20%SmolTalk-Chinese | 弃用 |
| `xalma_sft.jsonl` | 13812 | 纯X-ALMA SFT数据 | 早期 |

**结论**：通用数据混合（cqia/oasst/smoltalk/general）全部损害翻译质量，弃用。最终采用纯净ALMA+X-ALMA。

---

## 三、CPO / 偏好数据

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `cpo_preference.jsonl` | 44195 | 自生成偏好对（5候选COMET选best/worst），CPO v3基底 | 基底 |
| `cpo_v3_plus_7b.jsonl` | 44195 | v3 + HY-MT 7B增强（25.7%替换） | **采用** |
| `cpo_candidates.jsonl` | 22329 | 候选翻译池 | 中间 |
| `cpo_alma_combined.jsonl` | 35246 | ALMA合并偏好 | 早期 |
| `cpo_clean_final.jsonl` | 73357 | 干净版偏好（最终） | 早期 |
| `cpo_clean_full.jsonl` | 67210 | 干净版偏好（全量） | 早期 |
| `cpo_clean_no_selfgen.jsonl` | 36681 | 去自生成的干净偏好 | 早期 |
| `cpo_combined_preference.jsonl` | 79441 | 多源合并偏好 | 早期 |
| `cpo_exp_c.jsonl` | 36685 | 实验C：ALMA-R+X-ALMA chosen | 弃用 |
| `cpo_exp_d.jsonl` | 6127 | 实验D：GPT-4候选池 | 弃用 |
| `cpo_exp_d_candidates.jsonl` | 6130 | 实验D候选 | 中间 |
| `cpo_exp_e.jsonl` | 6128 | 实验E：+Codex候选 | 弃用 |
| `cpo_exp_e_candidates.jsonl` | 6130 | 实验E候选 | 中间 |
| `cpo_exp_f.jsonl` | 6127 | 实验F：+Claude候选 | 弃用 |
| `cpo_gpt4_vs_ours.jsonl` | 6130 | GPT-4 chosen + 我们greedy（Exp B） | keep |
| `cpo_hymt7b_vs_ours.jsonl` | 6085 | HY-MT 7B chosen + 我们greedy | 弃用（风格不匹配） |
| `cpo_mixed_best.jsonl` | 50325 | 混合最优偏好 | 早期 |
| `cpo_mixed_chosen.jsonl` | 4649 | 多源chosen混合 | 弃用（评分偏置） |
| `cpo_v3_7b_plus_gpt4.jsonl` | 50325 | v3+7B 44K + GPT-4 6K | 弃用（加数据稀释） |
| `alma_r_preference.jsonl` | 4691 | ALMA-R 原始偏好数据 | 参考 |
| `xalma_preference.jsonl` | 30555 | X-ALMA 原始偏好数据 | 参考 |
| `xalma_5cand.jsonl` | 30555 | X-ALMA 5候选数据 | 参考 |
| `xalma_selfgen_scored.jsonl` | 30531 | X-ALMA自生成+COMET打分 | 早期 |

### CPO 数据是怎么构建的（self-gen 偏好对）

CPO 数据**不是外部数据集，是模型自生成（on-policy）的偏好对**：

1. **源句池**：`data_build/generate_candidates.py` 用 sacrebleu 拉 **WMT17/18/19/20/21**
   五届测试集的源句（zh→en + en→zh 双向）。WMT22/23 留作评测，不进训练。
2. **生成候选**：对每个源句，用 **SFT 模型**（`checkpoints/output_1.7b_base_v2`）带
   temperature 采样生成 **5 个候选译文** → `cpo_candidates.jsonl`。
3. **选优**：`data_build/build_cpo_data.py` 用 **COMET 给 5 候选打分**，best→chosen、
   worst→rejected → `cpo_preference.jsonl`（= CPO v3 基底）。
4. **7B 增强**：`hymt7b_translate_cpo_sources.py` 用 `models/HY-MT1.5-7B` 翻译同一批源句，
   `build_cpo_v3_plus_7b.py` 把其中 **COMET 更优的 25.7% chosen 替换成 7B 译文** →
   `cpo_v3_plus_7b.jsonl`（**采用**）。

### 关键结论：HY-MT1.5-7B 数据的收益是**有条件**的

7B 数据**用对了才有小收益，裸用反而有害**（见 `results.tsv`）：

| 版本 | zh→en COMET | en→zh COMET | 结论 |
|------|-----------|-----------|------|
| `cpo_v3`（纯自生成） | 0.8005 | 0.8463 | 基底 |
| `cpo_v3_plus_7b`（COMET 过滤后 7B 替换 25.7% chosen） | **0.8017** | **0.8507** | **采用**，+0.0044 en-zh，best balanced |
| `cpo_7b_lora`（7B chosen，**无 COMET 过滤**） | 0.7422 | 0.7495 | 弃用：长度/风格不匹配，模型被带偏 |
| `cpo_v3_7b_gpt4`（v3+7B 再加 GPT-4 6K） | 0.7966 | 0.8478 | 弃用：稀释信号，质量>数量 |

- 收益依赖两个条件：(a) 必须 **COMET 过滤**；(b) 只**替换一部分 chosen** 融进自生成池。
- 幅度不大（~+0.004 COMET）；真正的大提升在后续 **GRPO**。
- 因此 `models/HY-MT1.5-7B` 是**可选增强**——只用纯自生成 `cpo_v3` 也能训出接近的模型。

---

## 四、GRPO 数据

| 文件 | 行数 | 说明 |
|------|------|------|
| `grpo_data.jsonl` | 22120 | WMT17-21 源句+参考译文（prompt池）。最佳模型用前3000；调参版SFT→GRPO用6000 |

---

## 五、HY-MT 7B 生成的翻译（蒸馏/增强用）

| 文件 | 说明 |
|------|------|
| `hymt7b_cpov3_translations.json` | HY-MT 7B 翻译 WMT17-21 源句（CPO v3+7B增强用） |
| `hymt7b_almar_translations.json` | HY-MT 7B 翻译 ALMA-R 源句 |
| `hymt_alma_r_translations.json` | 同上（早期版本） |
| `hymt7b_wmt23_translations.json` | HY-MT 7B 翻译 WMT23（基线eval产物） |
| `alma_r_sources.json` | ALMA-R 源句提取 |

---

## 六、原始语料（子目录）

| 目录 | 内容 |
|------|------|
| `ALMA-Human-Parallel/` | ALMA 人工平行语料（cs/de/is/ru/zh-en），WMT17-20 |
| `ALMA-R-Preference/` | ALMA-R 偏好数据（cs/de/is/ru/zh-en） |
| `X-ALMA-Preference/` | X-ALMA 偏好数据 |
| `flores200_dataset/` | Flores-200 评测集（204语言文件） |

---

## 七、评测集（held-out，不参与训练）

| 数据 | 说明 |
|------|------|
| WMT23 zh-en / en-zh | sacrebleu，主评测集 |
| WMT24 en-zh | sacrebleu，最新最干净（无zh-en方向） |
| `flores200_dataset/devtest/` | Flores-200，维基域 |
| ~~WMT22~~ | 排除：17.5%泄漏进ALMA SFT |

---

## 八、项目更早期收集的平行语料（broader project，多在 private/ 目录）

记录自 memory `reference_parallel_data.md`：

**S级（高质量）**
- X-ALMA Human Parallel：6906 zh-en对
- ALMA Human Parallel：HuggingFace `haoranxu/ALMA-Human-Parallel`
- WMT24++ Post-edits：998对，`private/parallel_data/wmt24pp_en_zh.jsonl`
- MSR Human Parity：2001句

**A级（良好）**
- HY-MT 自蒸馏：~150K对（SkyPile/Fineweb等），`private/distill_*.txt`
- FineTranslations（Gemma3过滤）：40K zh→en；全量未过滤~660K对

**B级（中等）**
- SFT过滤数据：44K对 `private/sft_filtered.jsonl`
- SFT COMET≥0.90：42K对
- WMT17-19 平行、News-Commentary v18、TED Talks

**C级（大但噪声多）**
- ParaCrawl 1.3GB、OpenSubtitles 346MB、WikiMatrix

**单语（蒸馏/CPT用）**
- Fineweb-edu 20万+英文句、SkyPile 中文、CNN/DailyMail 等

**预打包**
- `sft_distill_ft.jsonl`：186K对（蒸馏+FineTranslations）

注：第八节为项目更早阶段（含HY-MT tokenizer替换阶段）收集的语料，多数未进入最终最佳模型，记录备查。
