"""Final training with the best hyperparameters of a finished search (dissertation
recipe: seed 42, batch 8, 40 epochs, padding to 512). The exported model is the one at
the end of epoch 40 (no checkpoint selection); the best-validation checkpoint is also
evaluated on test for reference. Test metrics are reported per language with greedy,
Eisner and MST (Chu-Liu/Edmonds) decoding.

Usage: python -m hubparser_ml.final_train --encoder beto --head linear --corpus multilingual
"""
import argparse
import json
import shutil
import time

from transformers import AutoTokenizer, Trainer, TrainingArguments

from .common import (ENCODERS, HEADS, ML_ROOT, build_model, compute_metrics, evaluate_test, load_corpus,
                     make_collator, preprocess_logits_for_metrics, seed_everything, tokenize)
from .search import job_name

DECODINGS = ("greedy", "eisner", "mst")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--head", required=True, choices=list(HEADS))
    ap.add_argument("--corpus", default="multilingual")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    name = job_name(args.encoder, args.head, args.corpus) + ("__smoke" if args.smoke else "")
    res_dir = ML_ROOT / "results" / name
    best = json.loads((res_dir / "best.json").read_text())["best"]
    hp = dict(best["hyperparameters"])

    seed_everything(42)
    ds = load_corpus(args.corpus)
    if args.smoke:
        ds["train"] = ds["train"].shuffle(seed=0).select(range(64))
        ds["val"] = ds["val"].shuffle(seed=0).select(range(64))
        ds["test"] = ds["test"].shuffle(seed=0).select(range(60))
        hp["num_train_epochs"] = 2
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[args.encoder])
    model, cls = build_model(args.encoder, args.head, args.corpus)
    train = tokenize(ds["train"], tokenizer, args.head, pad_to_512=True)
    val = tokenize(ds["val"], tokenizer, args.head, pad_to_512=True)

    run_dir = ML_ROOT / "runs" / name
    targs = TrainingArguments(
        output_dir=str(run_dir), seed=42,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"], warmup_ratio=hp["warmup_ratio"],
        num_train_epochs=hp["num_train_epochs"], lr_scheduler_type="linear", max_grad_norm=1.0,
        per_device_train_batch_size=8, per_device_eval_batch_size=32,
        save_total_limit=1, save_only_model=True, load_best_model_at_end=False,
        metric_for_best_model="las", greater_is_better=True,
        label_names=["deprel_label", "upos_label", "head_label"], remove_unused_columns=False, report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      data_collator=make_collator(tokenizer.pad_token_id))
    t0 = time.time()
    trainer.train()
    minutes = round((time.time() - t0) / 60, 1)

    final_model = trainer.model.eval()
    test = {dec: evaluate_test(final_model, tokenizer, ds["test"], dec) for dec in DECODINGS}
    best_ckpt = trainer.state.best_model_checkpoint
    best_model = cls.from_pretrained(best_ckpt).to(final_model.device).eval()
    test_best = {dec: evaluate_test(best_model, tokenizer, ds["test"], dec) for dec in DECODINGS}
    best_epoch = next(h["epoch"] for h in trainer.state.log_history
                      if "eval_las" in h and abs(h["eval_las"] - trainer.state.best_metric) < 1e-12)

    export = ML_ROOT / "models" / name
    if export.exists():
        shutil.rmtree(export)
    final_model.config.auto_map = {"AutoModel": f"modeling_hubparser.{cls.__name__}"}
    final_model.config.architectures = [cls.__name__]
    final_model.config.base_model = ENCODERS[args.encoder]
    final_model.save_pretrained(export)
    tokenizer.save_pretrained(export)
    shutil.copy(ML_ROOT / "hubparser_ml" / "modeling_hubparser.py", export)

    record = dict(job=name, encoder=ENCODERS[args.encoder], head=args.head, corpus=args.corpus,
                  search_best=best, hyperparameters=hp, selection="final_epoch", train_minutes=minutes,
                  test=test, test_best_val_checkpoint=test_best, best_val_epoch=best_epoch,
                  val_history=[{"epoch": h["epoch"], "las": h["eval_las"]} for h in trainer.state.log_history if "eval_las" in h])
    (res_dir / "final.json").write_text(json.dumps(record, indent=2))
    (export / "hubparser_results.json").write_text(json.dumps(record, indent=2))
    shutil.rmtree(run_dir, ignore_errors=True)
    for dec in DECODINGS:
        print(f"[{name}] TEST {dec}: " + " | ".join(f"{l} UAS {m['uas']:.2f} LAS {m['las']:.2f}" for l, m in test[dec].items()), flush=True)


if __name__ == "__main__":
    main()
