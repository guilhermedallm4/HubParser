"""Hyperparameter search with the dissertation protocol: 10 configurations x 5-fold CV
over train+val, 40 epochs, batch 16, early stopping (patience 5), best epoch by
validation LAS. Dynamic padding (same metrics as padding to 512, ~4x faster).

The 10 configurations are fixed in data/search_configs.json: they are the first 10
trials of Optuna's TPESampler(seed=42), which are random samples (TPE only kicks in
after 10 trials), so every job evaluates exactly the same configurations.

Resumable: every finished fold is appended to results/<job>/folds.jsonl and is never
re-trained; a restart continues with the next missing (configuration, fold).

Usage: python -m hubparser_ml.search --encoder beto --head linear --corpus multilingual
"""
import argparse
import gc
import json
import shutil
import time

import numpy as np
import torch
from datasets import concatenate_datasets
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from .common import (ENCODERS, HEADS, ML_ROOT, build_model, compute_metrics, load_corpus, make_collator,
                     preprocess_logits_for_metrics, seed_everything, tokenize)

N_FOLDS = 5


def job_name(encoder, head, corpus):
    return f"{encoder}__{head}__{corpus}"


def run_fold(encoder, head, corpus, full, train_idx, val_idx, hp, tokenizer, out_dir):
    seed_everything(42)
    model, _ = build_model(encoder, head, corpus)
    train = tokenize(full.select(train_idx.tolist()), tokenizer, head, pad_to_512=False)
    val = tokenize(full.select(val_idx.tolist()), tokenizer, head, pad_to_512=False)
    args = TrainingArguments(
        output_dir=str(out_dir), seed=42,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"], warmup_ratio=hp["warmup_ratio"],
        num_train_epochs=hp["num_train_epochs"], lr_scheduler_type="linear", max_grad_norm=1.0,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        save_total_limit=1, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model="las", greater_is_better=True,
        label_names=["deprel_label", "upos_label", "head_label"], remove_unused_columns=False,
        report_to="none", dataloader_num_workers=4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      data_collator=make_collator(tokenizer.pad_token_id),
                      callbacks=[EarlyStoppingCallback(5)])
    t0 = time.time()
    trainer.train()
    res = trainer.evaluate()
    best_epoch = next((h["epoch"] for h in trainer.state.log_history
                       if "eval_las" in h and abs(h["eval_las"] - trainer.state.best_metric) < 1e-12), None)
    out = dict(las=res["eval_las"], uas=res["eval_uas"], upos=res["eval_upos_accuracy"],
               best_epoch=best_epoch, epochs_run=trainer.state.epoch, minutes=round((time.time() - t0) / 60, 1))
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return out


CONFIGS = json.loads((ML_ROOT / "data" / "search_configs.json").read_text())["configs"]


def config_index(hp):
    """Index of hp in the fixed list of 10 configurations (None if not in it)."""
    for i, c in enumerate(CONFIGS):
        if all(c[k] == hp.get(k) for k in ("learning_rate", "weight_decay", "warmup_ratio")):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--head", required=True, choices=list(HEADS))
    ap.add_argument("--corpus", default="multilingual")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    args = ap.parse_args()

    name = job_name(args.encoder, args.head, args.corpus) + ("__smoke" if args.smoke else "")
    res_dir = ML_ROOT / "results" / name
    res_dir.mkdir(parents=True, exist_ok=True)
    folds_file = res_dir / "folds.jsonl"

    # finished folds, keyed by (configuration index, fold); repeated runs of the same
    # configuration (from older restarts) are kept in the file but counted once
    done = {}
    if folds_file.exists():
        for line in folds_file.read_text().splitlines():
            r = json.loads(line)
            ci = config_index(r["hyperparameters"])
            if ci is not None:
                done.setdefault((ci, r["fold"]), r)

    ds = load_corpus(args.corpus)
    full = concatenate_datasets([ds["train"], ds["val"]])
    configs = [dict(c) for c in CONFIGS]
    if args.smoke:
        full = full.shuffle(seed=0).select(range(200))
        configs = [dict(c, num_train_epochs=2) for c in configs[:2]]
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[args.encoder])
    splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(np.arange(len(full))))

    for ci, hp in enumerate(configs):
        for fold, (tr, va) in enumerate(splits):
            if (ci, fold) in done:
                continue
            print(f"[{name}] config {ci} fold {fold} {hp}", flush=True)
            r = run_fold(args.encoder, args.head, args.corpus, full, tr, va, hp, tokenizer, res_dir / "tmp_run")
            rec = dict(job=name, encoder=ENCODERS[args.encoder], head=args.head, corpus=args.corpus,
                       trial=ci, config=ci, fold=fold, hyperparameters=hp, **r)
            with open(folds_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[(ci, fold)] = rec
            print(f"[{name}] config {ci} fold {fold} LAS {r['las']:.4f} ({r['minutes']} min)", flush=True)

    # best configuration by mean CV LAS (same rule as the dissertation)
    ranking = []
    for ci, hp in enumerate(configs):
        las = [done[(ci, f)]["las"] for f in range(N_FOLDS)]
        ranking.append({"trial": ci, "config": ci, "hyperparameters": hp, "mean_las": float(np.mean(las)),
                        "std_las": float(np.std(las, ddof=1)), "folds": N_FOLDS})
    ranking.sort(key=lambda x: -x["mean_las"])
    (res_dir / "best.json").write_text(json.dumps({"job": name, "n_configs": len(configs), "best": ranking[0],
                                                   "ranking": ranking}, indent=2))
    print(f"[{name}] BEST config {ranking[0]['config']} mean LAS {ranking[0]['mean_las']:.4f} {ranking[0]['hyperparameters']}", flush=True)


if __name__ == "__main__":
    main()
