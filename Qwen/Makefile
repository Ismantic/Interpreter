PYTHON      = /home/tfbao/.venv/bin/python -u
QWEN_BASE   = ./Qwen3-0.6B-Base
QWEN_NEW    = ./Qwen3-0.6B-Base-new-tok
PIECE_MODEL = ./piece_mt.model
RESULTS     = ./eval_results

# Paper Table 2 benchmark recipe (task : num_fewshot)
EVAL_TASKS = piqa:5 arc_challenge:25 hellaswag:10 mmlu:5 cmmlu:5 \
             agieval:0 bbh:3 humaneval:0 gsm8k:5

HF_CLI = /home/tfbao/.venv/bin/huggingface-cli
LM_EVAL = $(PYTHON) -m lm_eval run --model hf --batch_size auto

download:
	$(HF_CLI) download Qwen/Qwen3-0.6B-Base --local-dir $(QWEN_BASE)

replace:
	$(PYTHON) replace_tokenizer.py \
	    --old_model_path $(QWEN_BASE) \
	    --new_tokenizer_path $(PIECE_MODEL) \
	    --output_path $(QWEN_NEW)

# Baseline eval — Qwen3-0.6B-Base with its native BBPE tokenizer
eval-base:
	@for spec in $(EVAL_TASKS); do \
	    task=$${spec%%:*}; shots=$${spec##*:}; \
	    echo "=== $$task ($$shots-shot) ==="; \
	    extra=""; \
	    if [ "$$task" = "humaneval" ]; then extra="--confirm_run_unsafe_code"; fi; \
	    $(LM_EVAL) \
	        --model_args pretrained=$(QWEN_BASE),trust_remote_code=True,dtype=bfloat16 \
	        --tasks $$task --num_fewshot $$shots \
	        --output_path $(RESULTS)/base/$$task $$extra; \
	done

# Replaced model eval — needs piece-aware adapter; see eval_with_piece.py
eval-new:
	@for spec in $(EVAL_TASKS); do \
	    task=$${spec%%:*}; shots=$${spec##*:}; \
	    echo "=== $$task ($$shots-shot) ==="; \
	    $(PYTHON) eval_with_piece.py \
	        --model_path $(QWEN_NEW) \
	        --task $$task --num_fewshot $$shots \
	        --output_path $(RESULTS)/new-tok/$$task; \
	done

.PHONY: download replace eval-base eval-new
