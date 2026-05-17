# Low-bit quantization of the Qwen3-1.7B translator

Compress `output_1.7b_grpo_full` (the FP zh↔en translator, WMT23 0.8054/0.8546)
for smaller / on-device deployment. Two routes were explored: **PTQ** (llama.cpp
k-quant, no training) and **QAT** (a Sherry/SEQ port, training).

## Final results — WMT23, same eval protocol

| 档 | 文件/体积 | zh-en COMET | en-zh COMET | vs FP | 训练 |
|----|-----------|-------------|-------------|-------|------|
| FP 原版 | 3.3 GB | 0.8054 | 0.8546 | — | — |
| **4-bit** Q4_K_M | **1.03 GB** | 0.8015 | 0.8519 | −0.004 / −0.003 | 不用 |
| **3-bit** Q3_K_M + imatrix | **0.90 GB** | 0.7889 | 0.8334 | −0.017 / −0.021 | 不用 |
| **2-bit** SEQ-QAT (v3) | ~0.6 GB* | 0.7349 | 0.7666 | −0.071 / −0.088 | 11h QAT |

\* 2-bit 没有可部署文件（需自写 SEQ kernel）；只有 baked 的 3.4GB bf16 检查点。

**结论:部署用 4-bit(实质无损)或 3-bit(小损失、更小)。2-bit 损失明显且无部署路径。**

## 两条路线

- **PTQ(推荐,免训练)** — `convert_hf_to_gguf.py` → `llama-imatrix` → `llama-quantize`。
  4-bit / 3-bit 都是现成 GGUF,llama.cpp / 手机直接跑。3-bit **必须配 imatrix**
  (不加 imatrix 的 Q3_K 在难句上跑飞)。
- **QAT(我们的 SEQ 移植)** — 2-bit。Sherry/ParetoQ 的 StretchedElasticQuant
  伪量化 + KD 蒸馏训练。质量比 2-bit PTQ 好,但没有 llama.cpp 部署 kernel。

## 文件

| 文件 | 作用 |
|------|------|
| quant.py | 量化层:NMQuant+Arenas(1.25-bit 三值)、StretchedElasticQuant+SEQLinear(2/3-bit) |
| quantize.py | 模块替换 FP→量化;`bake_quantized` 固化回普通 Qwen3 |
| train_qat.py | QAT 训练循环(`--method seq|sherry`),凸退火 + KD |
| build_qat_data*.py | teacher 蒸馏构建 KD 数据 |
| ptq.py | 用 SEQ 量化器做 PTQ(实验用) |
| materialize.py | 把量化 checkpoint 固化成 vLLM 可加载的普通 Qwen3 |
| eval_qat.py / eval_q4_cpu.py / run_eval.sh | 评测(后者走 llama.cpp/GGUF) |
| serve_cpu.sh / translate_cpu.sh / webui/ | CPU 部署:llama-server + 翻译网页 |
| smoke_test.py / check_seq.py / check_servers.py | 各阶段 sanity check |

## 关键发现 / 踩过的坑

- **量化质量**:4-bit 实质无损(−0.004);3-bit −0.018;2-bit −0.08。bit 数对质量是
  强非线性,2-3 bit 之间是断崖(ParetoQ 的"learning transition")。
- **3-bit 必须 imatrix**:plain Q3_K 在难/残缺输入上重复跑飞;imatrix 修好。但
  imatrix 校准样本量 400→1500→2000 **无差别**(v1/v2/v3 COMET 全等)—— imatrix
  在 ~400 例就饱和。
- **Q3_K_L 比 Q3_K_M 更差**:更大的变体反而 en→zh 翻车(18% 输出成英文)。别用。
- **embedding 是固定成本**:词表 151936 → embedding 311M 参数 / 255MB,在 3-bit 和
  4-bit GGUF 里**完全一样**(都 Q6_K)。bit 数动不了它。要再缩只能**裁词表**。
- **3-bit vs 4-bit 速度基本持平**:Q3_K_M 生成快 ~11%(内存瓶颈),但 prompt 处理慢
  ~33%(算力瓶颈,3-bit 解包更费)。选 3-bit 是为体积,不是为速度。
- **2-bit QAT**:1 epoch 不够(eps 退火的纯量化收尾段会落在没见过的新数据上,
  收敛不了);3 epoch 才行。数据从 43K 扩到 133K 有效。Arenas 残差用**凸组合**
  `(1-eps)·quant + eps·fp`(论文是相加式,会爆),适合"保住已训好的模型"。
- **评测坑**:① 走 llama-server 的 chat 接口会套错聊天模板 → 用裸 `/completion`;
  ② CUDA 版 llama.cpp 在 CUDA 13.2 下 flash-attention kernel 有 bug → 必须 `-fa off`。

## 复现

**PTQ(4-bit / 3-bit)**
```
python <llama.cpp>/convert_hf_to_gguf.py output_1.7b_grpo_full --outfile f16.gguf --outtype f16
llama-imatrix -m f16.gguf -f calib.txt -o imatrix.dat        # 3-bit 才需要
llama-quantize [--imatrix imatrix.dat] f16.gguf out.gguf Q4_K_M|Q3_K_M
```
**QAT(2-bit)**
```
python build_qat_data.py            # teacher 蒸馏 -> qat_kd.jsonl
python train_qat.py --method seq --w_bits 2 --num_epochs 3   # -> baked 普通 Qwen3
python ../eval_multi.py --model_path output_qat_2bit_v3
```

## 还能改进的(均未做)

- **裁词表** — 省 ~130MB,正交、不掉质量。最划算。
- **logit-KD** — 提 2-bit 质量(KD 样本效率更高)。
- **Q2_K + imatrix** — 补一个能部署的 2-bit。
- `--tensor-type` 自定义混合精度 / 3-bit QAT — 3-bit 质量,提升小或要训练。
- 抬天花板要改进 FP 模型本身(translator 项目的事)。
