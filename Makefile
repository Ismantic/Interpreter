PYTHON = /home/tfbao/new/HY-MT/.venv/bin/python -u
DATA_CN = /home/tfbao/Shiyu/Tokenizer/data/cn_sentences.txt
DATA_EN = /home/tfbao/Shiyu/Tokenizer/data/en_sentences.txt
TOK_MODEL = ./piece_mt.model
CN_DICT = /home/tfbao/Shiyu/Tokenizer/scripts/dict.txt
OLD_MODEL_PATH = ./HY-MT1.5-1.8B

# Phase 1: embedding training
P1_TRAIN_DATA = ./private/train_data.pt
P1_MODEL_PATH = ./HY-MT1.5-1.8B-new-tok
P1_OUTPUT_DIR = ./private/phase1_v7
P1_SEQ_LEN = 384
P1_BATCH = 32
P1_ACCUM = 8
P1_STEPS = 2000
P1_WARMUP = 100
P1_LR = 1e-4
P1_MAX_CHUNKS = 600000

# Phase 2: full fine-tuning with translation data
P2_TRAIN_DATA = ./private/news_commentary_sft.jsonl
P2_MODEL_PATH = ./private/phase1_v5
P2_OUTPUT_DIR = ./private/phase2
P2_SEQ_LEN = 512
P2_BATCH = 2
P2_ACCUM = 16
P2_STEPS = 500
P2_WARMUP = 50
P2_MUON_LR = 0.0003
P2_ADAM_LR = 3e-5

tokenize:
	$(PYTHON) pretokenize.py \
		--input $(DATA_CN),$(DATA_EN) \
		--tokenizer_model $(TOK_MODEL) \
		--output $(P1_TRAIN_DATA) \
		--seq_length $(P1_SEQ_LEN) \
		--max_chunks $(P1_MAX_CHUNKS) \
		--cn_dict $(CN_DICT)

phase1:
	$(PYTHON) finetune_muon.py \
		--model_path $(P1_MODEL_PATH) \
		--train_data $(P1_TRAIN_DATA) \
		--output_dir $(P1_OUTPUT_DIR) \
		--freeze_transformer \
		--gradient_checkpointing \
		--max_seq_length $(P1_SEQ_LEN) \
		--batch_size $(P1_BATCH) \
		--gradient_accumulation_steps $(P1_ACCUM) \
		--max_steps $(P1_STEPS) \
		--warmup_steps $(P1_WARMUP) \
		--adam_lr $(P1_LR) \
		--logging_steps 10 \
		--save_steps 500

phase2:
	$(PYTHON) finetune_muon.py \
		--model_path $(P2_MODEL_PATH) \
		--train_data $(P2_TRAIN_DATA) \
		--mode sft \
		--output_dir $(P2_OUTPUT_DIR) \
		--gradient_checkpointing \
		--max_seq_length $(P2_SEQ_LEN) \
		--batch_size $(P2_BATCH) \
		--gradient_accumulation_steps $(P2_ACCUM) \
		--max_steps $(P2_STEPS) \
		--warmup_steps $(P2_WARMUP) \
		--muon_lr $(P2_MUON_LR) \
		--adam_lr $(P2_ADAM_LR) \
		--logging_steps 10 \
		--save_steps 500

eval:
	$(PYTHON) eval.py \
		--model_path $(P2_OUTPUT_DIR) \
		--testset wmt22 \
		--direction both \
		--max_samples 200 \
		--no_comet \
		--batch_size 8

eval-p1:
	$(PYTHON) eval.py \
		--model_path $(P1_OUTPUT_DIR) \
		--testset wmt22 \
		--direction both \
		--max_samples 200 \
		--no_comet \
		--batch_size 8

.PHONY: tokenize phase1 phase2 eval eval-p1
