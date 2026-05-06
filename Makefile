PYTHON = /home/tfbao/new/HY-MT/.venv/bin/python -u
DATA_CN = /home/tfbao/Shiyu/Tokenizer/data/cn_sentences.txt
DATA_EN = /home/tfbao/Shiyu/Tokenizer/data/en_sentences.txt
TOK_MODEL = ./piece_mt.model
CN_DICT = /home/tfbao/Shiyu/Tokenizer/scripts/dict.txt
TRAIN_DATA = ./private/train_data.pt
MODEL_PATH = ./HY-MT1.5-1.8B-new-tok
OUTPUT_DIR = ./private/phase1_v5

SEQ_LEN = 384
BATCH_SIZE = 32
GRAD_ACCUM = 8
MAX_STEPS = 2000
WARMUP = 100
LR = 1e-4
MAX_CHUNKS = 600000

tokenize:
	$(PYTHON) pretokenize.py \
		--input $(DATA_CN),$(DATA_EN) \
		--tokenizer_model $(TOK_MODEL) \
		--output $(TRAIN_DATA) \
		--seq_length $(SEQ_LEN) \
		--max_chunks $(MAX_CHUNKS) \
		--cn_dict $(CN_DICT)

train:
	$(PYTHON) finetune_muon.py \
		--model_path $(MODEL_PATH) \
		--train_data $(TRAIN_DATA) \
		--output_dir $(OUTPUT_DIR) \
		--freeze_transformer \
		--gradient_checkpointing \
		--max_seq_length $(SEQ_LEN) \
		--batch_size $(BATCH_SIZE) \
		--gradient_accumulation_steps $(GRAD_ACCUM) \
		--max_steps $(MAX_STEPS) \
		--warmup_steps $(WARMUP) \
		--adam_lr $(LR) \
		--logging_steps 10 \
		--save_steps 500

eval:
	$(PYTHON) eval.py \
		--model_path $(OUTPUT_DIR) \
		--testset wmt22 \
		--direction both \
		--max_samples 200 \
		--no_comet \
		--batch_size 8

.PHONY: tokenize train eval
