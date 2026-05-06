# Interpreter

HY-MT1.5-1.8B 翻译模型 Tokenizer 替换实验。将原始 HuggingFace tokenizer (vocab 120K) 替换为自定义 piece tokenizer (vocab 65K)，通过 embedding 映射 + 两阶段训练恢复翻译能力。

## 依赖

- Python 3.10+, PyTorch 2.x, transformers (仅模型加载)
- [piece_tokenizer](https://github.com/user/Tokenizer) — 自定义 BPE tokenizer (需编译安装 Python binding)
- sacrebleu, unbabel-comet (评测)

## 工作流

```bash
# 1. 给 tokenizer 添加特殊 token
python add_special_tokens.py \
    --input /path/to/piece.model \
    --output piece_mt.model \
    --tokens '<pad>,<user>,<assistant>,<system>'

# 2. 替换 tokenizer，初始化 embedding
python replace_tokenizer.py \
    --old_model_path ./HY-MT1.5-1.8B \
    --new_tokenizer_path ./piece_mt.model \
    --output_path ./HY-MT1.5-1.8B-new-tok

# 3. 预处理训练数据 (packing 成固定长度 chunks)
make tokenize

# 4. Phase 1: 冻结 Transformer, 训练 Embedding
make train

# 5. 评测
make eval
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `finetune_muon.py` | 训练脚本，支持 SFT/CLM 模式，Muon+Adam 优化器，冻结 Transformer |
| `muon.py` | Muon 优化器实现 (SingleDeviceMuonWithAuxAdam) |
| `replace_tokenizer.py` | Tokenizer 替换 + embedding 初始化 (copy-average 映射) |
| `add_special_tokens.py` | 向 piece .model 文件追加 CONTROL token |
| `pretokenize.py` | 预处理文本数据为 packed token chunks (.pt) |
| `tokenizer_wrapper.py` | piece_tokenizer 的 HuggingFace 兼容 wrapper |
| `eval.py` | WMT 评测脚本 (BLEU + COMET) |
| `piece_mt.model` | 含特殊 token 的 piece tokenizer (vocab 65007) |
| `Makefile` | tokenize / train / eval 一键命令 |

## Embedding 映射策略

对新词表每个 token，用旧 tokenizer encode 其文本，取旧 embedding 的均值初始化：
- 一对一映射: 73.5% (直接复制)
- 多对一映射: 25.5% (平均 2.1 个旧 token)
- 字节 fallback: 1.0% (全局均值)

## 评测结果 (WMT22, 200 句)

| 模型 | zh→en BLEU | en→zh BLEU |
|------|-----------|-----------|
| 原始 HY-MT1.5-1.8B | 16.38 | 31.66 |
| 替换 tokenizer (未训练) | 16.88 | 29.81 |
| Phase 1 训练中 | - | - |
