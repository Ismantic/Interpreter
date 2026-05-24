"""MacBERT-large + CRF training for CWS on PD-1998."""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402


def boundary_f1(pred_words, gold_words):
    def spans(ws):
        out, pos = set(), 0
        for w in ws:
            out.add((pos, pos + len(w)))
            pos += len(w)
        return out
    P, G = spans(pred_words), spans(gold_words)
    if not P or not G:
        return 0.0
    tp = len(P & G)
    if tp == 0:
        return 0.0
    p = tp / len(P)
    r = tp / len(G)
    return 2 * p * r / (p + r)


@torch.no_grad()
def evaluate(model, dataset, collator, device, batch_size=64):
    """Iterate dataset in order, decode with CRF, compute boundary F1."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=2)
    f1s = []
    item_idx = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model.decode(batch["input_ids"], batch["attention_mask"])
        for pred_tags in preds:
            item = dataset.items[item_idx]
            chars = item["chars"]
            n = min(len(chars), len(pred_tags))
            pred_words = bies_to_words(chars[:n], pred_tags[:n])
            gold_words = bies_to_words(chars[:n], item["tags"][:n])
            f1s.append(boundary_f1(pred_words, gold_words))
            item_idx += 1
    model.train()
    return sum(f1s) / max(1, len(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--train_jsonl", default="./data/cws.jsonl")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_chars", type=int, default=254)
    ap.add_argument("--bert_lr", type=float, default=2e-5)
    ap.add_argument("--crf_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_dev_limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    print(f"Tokenizer: vocab={tokenizer.vocab_size} pad={tokenizer.pad_token_id} "
          f"unk={tokenizer.unk_token_id}")

    print(f"\nLoading train: {args.train_jsonl}")
    t0 = time.time()
    train_ds = CWSDataset(args.train_jsonl, max_chars=args.max_chars)
    print(f"  {len(train_ds)} samples  ({time.time()-t0:.1f}s)")

    print(f"Loading dev: {args.dev_jsonl}")
    t0 = time.time()
    full_dev_ds = CWSDataset(args.dev_jsonl, max_chars=args.max_chars)
    print(f"  {len(full_dev_ds)} samples  ({time.time()-t0:.1f}s)")

    class DevSubset:
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]
    dev_subset = DevSubset(full_dev_ds.items[:args.eval_dev_limit])
    print(f"  eval subset for per-epoch monitor: {len(dev_subset)}")

    collator = Collator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2, pin_memory=True,
                              drop_last=False)

    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)

    # Param groups: BERT (decoupled lr from CRF head)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    bert_params = list(model.bert.named_parameters())
    head_params = list(model.classifier.named_parameters()) + list(model.crf.named_parameters())
    grouped = [
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": 0.0},
        {"params": [p for _, p in head_params],
         "lr": args.crf_lr, "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"\nTotal steps: {total_steps}  warmup: {warmup_steps}\n")

    model.train()
    global_step = 0
    best_f1 = 0.0
    t_start = time.time()

    for epoch in range(args.epochs):
        ep_loss, ep_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ep_loss += loss.item()
            ep_n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                el = time.time() - t_start
                sps = global_step / el
                eta_min = (total_steps - global_step) / sps / 60
                print(f"  ep{epoch+1} step {global_step}/{total_steps}  "
                      f"loss {loss.item():.3f}  "
                      f"lr_bert {optimizer.param_groups[0]['lr']:.2e}  "
                      f"{sps:.1f} step/s  ETA {eta_min:.1f}m", flush=True)

        dev_f1 = evaluate(model, dev_subset, collator, device)
        avg_loss = ep_loss / max(1, ep_n)
        print(f"\n=== Epoch {epoch+1}/{args.epochs}  avg_loss {avg_loss:.4f}  "
              f"dev_F1(subset {len(dev_subset)}) {dev_f1:.4f} ===", flush=True)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"    ↑ saved best.pt", flush=True)
        print(flush=True)

    torch.save(model.state_dict(), out_dir / "final.pt")
    tokenizer.save_pretrained(out_dir)
    print(f"\nDone. Best dev_F1(subset): {best_f1:.4f}")
    print(f"Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
